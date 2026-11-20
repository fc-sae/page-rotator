#!/usr/bin/env python3
"""Rasteriza cada página de data/raw/<ID>.pdf a PNG en escala de grises.

Salida:
  data/pages/<ID>/<NNN>.png      página NNN (1-based) a 150 dpi, tal como se ve (con su /Rotate aplicado)
  data/pages/manifest.csv        una fila por página: pdf_id, page, rotate, tamaño en pt y px, ruta

La etiqueta de orientación se refiere a la página *como se ve*: la corrección que se aplique después
será (rotate + corrección) mod 360, igual que en la CLI.

Es reanudable: salta los PDFs que ya tienen páginas en el manifiesto, así que se puede correr
varias veces mientras download.py sigue bajando archivos.

    uv run python render.py [--dpi 150] [--workers N]
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIELDS = ["pdf_id", "page", "status", "rotate", "width_pt", "height_pt", "width_px", "height_px", "path", "error"]


def load_done(manifest: Path) -> set[str]:
    """PDFs con al menos una página renderizada (los que fallaron se reintentan)."""
    if not manifest.exists():
        return set()
    with manifest.open(newline="", encoding="utf-8") as f:
        return {row["pdf_id"] for row in csv.DictReader(f) if row["status"] == "ok"}


def render_pdf(pdf_path: Path, out_root: Path, dpi: int = 150) -> list[dict]:
    """Renderiza todas las páginas de un PDF. Devuelve una fila por página, o una fila de error."""
    import pypdfium2 as pdfium  # importado aquí: cada proceso hijo carga su propia pdfium

    pdf_id = pdf_path.stem
    try:
        doc = pdfium.PdfDocument(pdf_path)
    except Exception as e:  # noqa: BLE001
        return [{"pdf_id": pdf_id, "page": "", "status": "error", "error": f"{type(e).__name__}: {e}"[:300]}]

    out_dir = out_root / pdf_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        if len(doc) == 0:
            return [{"pdf_id": pdf_id, "page": "", "status": "error", "error": "PDF sin páginas"}]
        for i in range(len(doc)):
            row = {"pdf_id": pdf_id, "page": i + 1}
            try:
                page = doc[i]
                w_pt, h_pt = page.get_size()  # ya considera /Rotate
                img = page.render(scale=dpi / 72, grayscale=True).to_pil().convert("L")
                dst = out_dir / f"{i + 1:03d}.png"
                tmp = dst.with_suffix(".png.part")
                img.save(tmp, format="PNG", optimize=False)
                tmp.replace(dst)
                row.update(status="ok", rotate=page.get_rotation(), width_pt=round(w_pt, 1),
                           height_pt=round(h_pt, 1), width_px=img.width, height_px=img.height,
                           path=str(dst.relative_to(out_root)), error="")
                page.close()
            except Exception as e:  # noqa: BLE001 - una página mala no tumba el PDF
                row.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
            rows.append(row)
    finally:
        doc.close()
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "pages")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, help="procesar solo los primeros N PDFs pendientes (prueba)")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.csv"
    done = load_done(manifest)
    pdfs = sorted(args.raw.glob("*.pdf"))  # .pdf.part (descargas en curso) no coincide
    pending = [p for p in pdfs if p.stem not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"{len(pdfs)} PDFs en {args.raw}, {len(done)} ya renderizados, {len(pending)} pendientes "
          f"({args.workers} procesos, {args.dpi} dpi).")
    if not pending:
        return 0

    new_file = not manifest.exists()
    counts: Counter[str] = Counter()
    pages = 0
    start = time.monotonic()
    with manifest.open("a", newline="", encoding="utf-8") as mf, \
            ProcessPoolExecutor(max_workers=args.workers,
                                mp_context=multiprocessing.get_context("spawn")) as pool:
        writer = csv.DictWriter(mf, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        futures = {pool.submit(render_pdf, p, args.out, args.dpi): p for p in pending}
        for n, fut in enumerate(as_completed(futures), 1):
            rows = fut.result()
            writer.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
            mf.flush()
            ok = sum(r["status"] == "ok" for r in rows)
            pages += ok
            counts["ok" if ok == len(rows) else "error"] += 1
            if ok < len(rows):
                errs = "; ".join(r["error"] for r in rows if r["status"] != "ok")
                print(f"ERROR {futures[fut].name}: {errs}", file=sys.stderr)
            if n % 100 == 0 or n == len(pending):
                elapsed = time.monotonic() - start
                print(f"[{n:>5}/{len(pending)}] {pages} páginas, {n / elapsed:.1f} PDF/s", flush=True)

    print(f"\nListo en {(time.monotonic() - start) / 60:.1f} min: {pages} páginas. "
          f"PDFs ok: {counts['ok']}, con errores: {counts['error']}.")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
