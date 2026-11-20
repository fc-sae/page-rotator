#!/usr/bin/env python3
"""Descarga los PDFs listados en data/urls.txt desde Google Drive con rclone.

Por cada URL:
  1. extrae el ID de Drive (formatos open?id=<ID> y /file/d/<ID>/view),
  2. lo descarga con `rclone backend copyid`,
  3. le quita el cifrado con `qpdf --decrypt` (los no cifrados quedan igual),
  4. lo guarda como data/raw/<ID>.pdf y agrega una fila a data/raw/manifest.csv.

Es reanudable: al volver a correrlo salta los IDs que ya están bien y reintenta los que fallaron.
Se puede interrumpir con Ctrl-C en cualquier momento.

Solo usa la biblioteca estándar:  python3 tools/download.py [--workers 4] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ID_RE = re.compile(r"(?:[?&]id=|/file/d/)([A-Za-z0-9_-]{10,})")
FIELDS = ["id", "status", "original_name", "bytes", "encrypted", "attempts", "error", "finished_at"]
DONE_STATUSES = {"ok", "not_pdf"}
PERMANENT_ERRORS = re.compile(r"404|not ?found|notFound|insufficientFilePermissions|cannotDownloadFile", re.I)


def extract_id(url: str) -> str | None:
    m = ID_RE.search(url)
    return m.group(1) if m else None


def read_ids(path: Path) -> tuple[list[str], list[str]]:
    """IDs únicos en el orden del archivo, y las líneas no vacías sin ID reconocible."""
    ids: list[str] = []
    seen: set[str] = set()
    bad: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        file_id = extract_id(line)
        if file_id is None:
            bad.append(line)
        elif file_id not in seen:
            seen.add(file_id)
            ids.append(file_id)
    return ids, bad


def load_done(manifest: Path, raw_dir: Path) -> set[str]:
    """IDs cuya última fila es exitosa y cuyo archivo sigue en disco."""
    if not manifest.exists():
        return set()
    last: dict[str, dict] = {}
    with manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            last[row["id"]] = row
    done = set()
    for file_id, row in last.items():
        if row["status"] == "ok" and (raw_dir / f"{file_id}.pdf").exists():
            done.add(file_id)
        elif row["status"] == "not_pdf" and any((raw_dir / "other").glob(f"{file_id}*")):
            done.add(file_id)
    return done


def _last_line(text: str) -> str:
    """Última línea, conservando el final (donde rclone pone la causa: 'Error 404: ...')."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return ""
    line = lines[-1]
    # rclone: '... failed copying "ID" to "/ruta/larga/": couldn't find id: googleapi: Error 404: ...'
    if "failed copying" in line and '": ' in line:
        line = line.rsplit('": ', 1)[1]
    return line[-300:]


def fetch_one(
    file_id: str,
    raw_dir: Path,
    remote: str = "gdrive",
    attempts: int = 3,
    backoff: float = 5.0,
    rclone: str = "rclone",
    qpdf: str = "qpdf",
) -> dict:
    row = {k: "" for k in FIELDS}
    row["id"] = file_id
    tmp = raw_dir / ".tmp" / file_id
    try:
        error = ""
        files: list[Path] = []
        for attempt in range(1, attempts + 1):
            row["attempts"] = attempt
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            proc = subprocess.run(
                [rclone, "backend", "copyid", f"{remote}:", file_id, f"{tmp}/",
                 "--retries", "1", "--low-level-retries", "5"],
                capture_output=True, text=True, timeout=600,
            )
            files = [p for p in tmp.iterdir() if p.is_file()]
            if proc.returncode == 0 and len(files) == 1:
                break
            error = _last_line(proc.stderr) or f"rclone rc={proc.returncode}, {len(files)} archivos"
            files = []
            if PERMANENT_ERRORS.search(proc.stderr) or attempt == attempts:
                break
            time.sleep(backoff * 2 ** (attempt - 1))

        if not files:
            row.update(status="error", error=error)
            return row

        src = files[0]
        row["original_name"] = src.name
        with src.open("rb") as f:
            is_pdf = f.read(1024).lstrip().startswith(b"%PDF")
        if not is_pdf:
            other = raw_dir / "other"
            other.mkdir(exist_ok=True)
            dst = other / f"{file_id}{src.suffix}"
            shutil.move(src, dst)
            row.update(status="not_pdf", bytes=dst.stat().st_size)
            return row

        # qpdf --is-encrypted: 0 = cifrado, 2 = no cifrado
        enc = subprocess.run([qpdf, "--is-encrypted", str(src)], capture_output=True)
        row["encrypted"] = {0: "yes", 2: "no"}.get(enc.returncode, "?")

        part = raw_dir / f"{file_id}.pdf.part"
        dec = subprocess.run([qpdf, "--decrypt", str(src), str(part)], capture_output=True, text=True)
        # qpdf: 0 = ok, 3 = ok con advertencias (el archivo sí se escribe)
        if dec.returncode in (0, 3) and part.exists():
            dst = raw_dir / f"{file_id}.pdf"
            part.replace(dst)
            row.update(status="ok", bytes=dst.stat().st_size,
                       error=_last_line(dec.stderr) if dec.returncode == 3 else "")
        else:
            part.unlink(missing_ok=True)
            failed = raw_dir / "failed"
            failed.mkdir(exist_ok=True)
            shutil.move(src, failed / f"{file_id}.pdf")
            row.update(status="error", error="qpdf: " + (_last_line(dec.stderr) or f"rc={dec.returncode}"))
        return row
    except Exception as e:  # noqa: BLE001 - un archivo malo no debe detener la descarga
        row.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
        return row
    finally:
        row["finished_at"] = datetime.now().isoformat(timespec="seconds")
        shutil.rmtree(tmp, ignore_errors=True)


def preflight(remote: str, rclone: str, qpdf: str) -> None:
    for tool in (rclone, qpdf):
        if shutil.which(tool) is None:
            sys.exit(f"No se encontró `{tool}` en el PATH.")
    remotes = subprocess.run([rclone, "listremotes"], capture_output=True, text=True).stdout.split()
    if f"{remote}:" not in remotes:
        sys.exit(f"El remote `{remote}:` no existe. Remotes: {' '.join(remotes) or '(ninguno)'}")
    # Valida la autorización (y refresca el token una vez) antes de lanzar los workers.
    about = subprocess.run([rclone, "about", f"{remote}:"], capture_output=True, text=True, timeout=120)
    if about.returncode != 0:
        sys.exit(f"`rclone about {remote}:` falló:\n{about.stderr.strip()}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urls", type=Path, default=REPO / "data" / "urls.txt")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--remote", default="gdrive")
    ap.add_argument("--workers", type=int, default=4, help="descargas en paralelo (default 4)")
    ap.add_argument("--attempts", type=int, default=3, help="intentos por archivo (default 3)")
    ap.add_argument("--limit", type=int, help="descargar solo los primeros N pendientes (prueba)")
    ap.add_argument("--rclone", default="rclone")
    ap.add_argument("--qpdf", default="qpdf")
    args = ap.parse_args(argv)

    ids, bad = read_ids(args.urls)
    for line in bad:
        print(f"AVISO: línea sin ID de Drive: {line}", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(args.out / ".tmp", ignore_errors=True)
    manifest = args.out / "manifest.csv"
    done = load_done(manifest, args.out)
    pending = [i for i in ids if i not in done]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"{len(ids)} IDs únicos, {len(done)} ya descargados, {len(pending)} por descargar "
          f"({args.workers} en paralelo).")
    if not pending:
        return 0
    preflight(args.remote, args.rclone, args.qpdf)

    new_file = not manifest.exists()
    counts: Counter[str] = Counter()
    start = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=args.workers)
    with manifest.open("a", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        futures = [pool.submit(fetch_one, i, args.out, args.remote, args.attempts,
                               rclone=args.rclone, qpdf=args.qpdf) for i in pending]
        try:
            for n, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                writer.writerow(row)
                mf.flush()
                counts[row["status"]] += 1
                elapsed = time.monotonic() - start
                eta = elapsed / n * (len(pending) - n)
                detail = row["original_name"] if row["status"] == "ok" else row["error"] or row["original_name"]
                print(f"[{n:>5}/{len(pending)}] {row['status']:<7} {row['id']}  {detail}"
                      f"  (ETA {eta / 60:.0f} min)", flush=True)
        except KeyboardInterrupt:
            print("\nInterrumpido. Vuelve a correr el mismo comando para continuar.", file=sys.stderr)
            pool.shutdown(wait=False, cancel_futures=True)
            os._exit(130)
    pool.shutdown()

    summary = ", ".join(f"{k}: {v}" for k, v in counts.most_common())
    print(f"\nListo en {(time.monotonic() - start) / 60:.1f} min. {summary}")
    if counts["error"]:
        print(f"Errores en {manifest} (status=error). Vuelve a correr para reintentarlos.")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
