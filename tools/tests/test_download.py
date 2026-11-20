"""Pruebas de download.py con un `rclone` falso (no toca Google Drive).

Corre con:  python3 -m unittest discover -s tools/tests -p 'test_download.py'
       o:   cd tools && uv run pytest tests/test_download.py
"""

import csv
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import download  # noqa: E402

# rclone falso: `rclone backend copyid remote: ID DEST/ ...`  →  $4 = ID, $5 = DEST
# El comportamiento depende del ID.
FAKE_RCLONE = textwrap.dedent("""\
    #!/usr/bin/env bash
    id="$4"; dest="$5"
    case "$id" in
      okpdf*)    cp "$FIXTURE_PDF" "${dest}constancia 201 posesión.pdf" ;;
      encpdf*)   cp "$FIXTURE_ENC" "${dest}cifrado.pdf" ;;
      notpdf*)   echo "hola" > "${dest}nota.txt" ;;
      missing*)  printf '2026/11/20 15:42:37 Failed to backend: command "copyid" failed: failed copying "%s" to "%s%0300d": couldn'"'"'t find id: googleapi: Error 404: File not found: %s., notFound\\n' "$id" "$dest" 0 "$id" >&2; exit 1 ;;
      flaky*)    n=$(cat "$COUNTER" 2>/dev/null || echo 0); echo $((n+1)) > "$COUNTER"
                 if [ "$n" -lt 1 ]; then echo 'ERROR : 500 backend error' >&2; exit 1; fi
                 cp "$FIXTURE_PDF" "${dest}tarde.pdf" ;;
      broken*)   echo "%PDF-1.7 basura" > "${dest}roto.pdf" ;;
    esac
""")


@unittest.skipIf(shutil.which("qpdf") is None, "requiere qpdf")
class FetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.raw = self.tmp / "raw"
        self.raw.mkdir()
        fixture = self.tmp / "fixture.pdf"
        subprocess.run(["qpdf", "--empty", str(fixture)], check=True)
        enc = self.tmp / "enc.pdf"
        subprocess.run(["qpdf", "--encrypt", "", "owner", "256", "--", str(fixture), str(enc)], check=True)
        self.rclone = self.tmp / "rclone"
        self.rclone.write_text(FAKE_RCLONE)
        self.rclone.chmod(0o755)
        import os
        os.environ.update(FIXTURE_PDF=str(fixture), FIXTURE_ENC=str(enc), COUNTER=str(self.tmp / "counter"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def fetch(self, file_id):
        return download.fetch_one(file_id, self.raw, attempts=3, backoff=0, rclone=str(self.rclone))

    def test_ok_pdf_renamed_to_id_and_keeps_original_name(self):
        row = self.fetch("okpdf123456")
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["original_name"], "constancia 201 posesión.pdf")
        self.assertEqual(row["encrypted"], "no")
        self.assertTrue((self.raw / "okpdf123456.pdf").exists())
        self.assertFalse((self.raw / ".tmp" / "okpdf123456").exists())

    def test_encrypted_pdf_is_decrypted(self):
        row = self.fetch("encpdf123456")
        self.assertEqual((row["status"], row["encrypted"]), ("ok", "yes"))
        out = subprocess.run(["qpdf", "--is-encrypted", str(self.raw / "encpdf123456.pdf")])
        self.assertEqual(out.returncode, 2)  # 2 = no cifrado

    def test_not_pdf_goes_to_other(self):
        row = self.fetch("notpdf123456")
        self.assertEqual(row["status"], "not_pdf")
        self.assertTrue((self.raw / "other" / "notpdf123456.txt").exists())

    def test_not_found_is_not_retried(self):
        row = self.fetch("missing12345")
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["attempts"], 1)
        self.assertTrue(row["error"].startswith("couldn't find id: googleapi: Error 404"), row["error"])

    def test_transient_error_is_retried(self):
        row = self.fetch("flaky1234567")
        self.assertEqual((row["status"], row["attempts"]), ("ok", 2))

    def test_qpdf_failure_keeps_original_in_failed(self):
        row = self.fetch("broken123456")
        self.assertEqual(row["status"], "error")
        self.assertTrue(row["error"].startswith("qpdf:"))
        self.assertTrue((self.raw / "failed" / "broken123456.pdf").exists())
        self.assertFalse((self.raw / "broken123456.pdf").exists())


class ParseTests(unittest.TestCase):
    def test_extract_id_formats(self):
        self.assertEqual(download.extract_id("https://drive.google.com/open?id=120NgNEdva-uTJf7ueCaNOpnhaKHr-4Z0"),
                         "120NgNEdva-uTJf7ueCaNOpnhaKHr-4Z0")
        self.assertEqual(download.extract_id(
            "https://drive.google.com/file/d/1Hc7gmUYcwY4m_QkIXXzWeFzhJUDagW_N/view?usp=drive_link"),
            "1Hc7gmUYcwY4m_QkIXXzWeFzhJUDagW_N")
        self.assertIsNone(download.extract_id("https://example.com/algo"))

    def test_read_ids_dedupes_and_reports_bad_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "urls.txt"
            p.write_text("https://drive.google.com/open?id=AAAAAAAAAA1\n\n"
                         "https://drive.google.com/file/d/BBBBBBBBBB2/view\n"
                         "https://drive.google.com/open?id=AAAAAAAAAA1\n"
                         "basura\n")
            ids, bad = download.read_ids(p)
        self.assertEqual(ids, ["AAAAAAAAAA1", "BBBBBBBBBB2"])
        self.assertEqual(bad, ["basura"])

    def test_load_done_uses_last_row_and_requires_file(self):
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d)
            (raw / "aaa.pdf").write_bytes(b"%PDF")
            m = raw / "manifest.csv"
            with m.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=download.FIELDS)
                w.writeheader()
                for file_id, status in [("aaa", "error"), ("aaa", "ok"),   # reintento exitoso
                                        ("bbb", "ok"), ("bbb", "error"),   # último falló
                                        ("ccc", "ok")]:                    # ok pero sin archivo
                    w.writerow({"id": file_id, "status": status})
            self.assertEqual(download.load_done(m, raw), {"aaa"})


if __name__ == "__main__":
    unittest.main()
