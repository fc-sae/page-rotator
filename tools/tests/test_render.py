import csv
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image

import render


def close(size, expected):
    """pdfium redondea hacia arriba: se acepta ±1 px."""
    return all(abs(a - b) <= 1 for a, b in zip(size, expected))


def make_pdf(path: Path, rotations: list[int], size=(612, 792)) -> Path:
    doc = pdfium.PdfDocument.new()
    for rot in rotations:
        page = doc.new_page(*size)
        page.set_rotation(rot)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def dirs(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "pages"
    raw.mkdir()
    return raw, out


def test_render_pages_at_150dpi_grayscale(dirs):
    raw, out = dirs
    rows = render.render_pdf(make_pdf(raw / "abc.pdf", [0, 0]), out, dpi=150)
    assert [r["status"] for r in rows] == ["ok", "ok"]
    assert [r["page"] for r in rows] == [1, 2]
    img = Image.open(out / "abc" / "001.png")
    assert img.mode == "L"
    assert close(img.size, (1275, 1650))  # carta a 150 dpi
    assert rows[0]["path"] == "abc/001.png"


def test_existing_rotate_is_applied_and_recorded(dirs):
    raw, out = dirs
    rows = render.render_pdf(make_pdf(raw / "rot.pdf", [0, 90]), out, dpi=150)
    assert rows[0]["rotate"] == 0 and close((rows[0]["width_px"], rows[0]["height_px"]), (1275, 1650))
    # Página con /Rotate 90: se renderiza como se ve (horizontal)
    assert rows[1]["rotate"] == 90 and close((rows[1]["width_px"], rows[1]["height_px"]), (1650, 1275))
    assert (rows[1]["width_pt"], rows[1]["height_pt"]) == (792, 612)


def test_broken_pdf_returns_error_row(dirs):
    raw, out = dirs
    bad = raw / "bad.pdf"
    bad.write_bytes(b"%PDF-1.7 basura")
    rows = render.render_pdf(bad, out)
    assert len(rows) == 1 and rows[0]["status"] == "error" and rows[0]["error"]


def test_main_is_resumable_and_skips_partial_downloads(dirs):
    raw, out = dirs
    make_pdf(raw / "a.pdf", [0])
    make_pdf(raw / "b.pdf", [0, 180])
    (raw / "c.pdf.part").write_bytes(b"descarga en curso")
    assert render.main(["--raw", str(raw), "--out", str(out), "--workers", "2"]) == 0

    make_pdf(raw / "d.pdf", [0])
    assert render.main(["--raw", str(raw), "--out", str(out), "--workers", "2"]) == 0

    with (out / "manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert sorted((r["pdf_id"], r["page"]) for r in rows) == [("a", "1"), ("b", "1"), ("b", "2"), ("d", "1")]
