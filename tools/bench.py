"""Etapa 6: compara page-rotator contra ocrmypdf en el test dorado (errores de orientación y tiempo).

Por cada PDF de la partición (por defecto `test`) corre, uno tras otro:
  page-rotator -q --threshold T IN OUT
  ocrmypdf -q --rotate-pages --rotate-pages-threshold 0.9 --skip-text IN OUT   (línea de producción)

El giro aplicado a cada página se mide visualmente y no leyendo /Rotate, porque ocrmypdf reescribe
el PDF y gira el contenido: se rasteriza la página de entrada y la de salida tal como se ven, y gana
el giro horario (0/90/180/270) de la entrada que más se parece a la salida. Así ambas herramientas
se miden igual. Cada página etiquetada `ok` se clasifica como
  bien        giro aplicado == etiqueta
  mal         se giró, pero a otra orientación (el peor caso: estropea una página)
  sin corregir  la etiqueta pedía giro y no se aplicó
  dudoso      no se pudo medir (página casi en blanco o simétrica)

Salida en data/bench/: pages.csv (por página), pdfs.csv (tiempos) y report.md.

    uv run python bench.py [--split test] [--threshold 0.9] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import os
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
ROTATIONS = (0, 90, 180, 270)
CMP_SIDE = 128       # lado del cuadrado en el que se comparan las páginas
RENDER_SIDE = 400    # lado largo del rasterizado para medir
MIN_GAP = 1.5        # diferencia mínima (niveles de gris) entre el mejor giro y el segundo
OCRMYPDF = ["ocrmypdf", "-q", "--rotate-pages", "--rotate-pages-threshold", "0.9", "--skip-text"]


# --- medición del giro ------------------------------------------------------------------------

def _square(a: np.ndarray) -> np.ndarray:
    img = Image.fromarray(a).resize((CMP_SIDE, CMP_SIDE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def applied_rotation(before: np.ndarray, after: np.ndarray) -> int | None:
    """Giro horario que lleva `before` a `after` (imágenes en gris tal como se ven), o None si es dudoso.

    Los giros que cambian la forma (vertical ↔ horizontal) se descartan cuando la página no es casi
    cuadrada; entre los restantes gana la menor diferencia media, y debe ganar con margen.
    """
    ha, wa = after.shape
    aspect = wa / ha
    target = _square(after)
    scores = {}
    for rot in ROTATIONS:
        cand = np.rot90(before, k=-rot // 90)  # k negativo = sentido horario
        hc, wc = cand.shape
        if abs(np.log((wc / hc) / aspect)) > 0.1:
            continue
        scores[rot] = float(np.abs(_square(np.ascontiguousarray(cand)) - target).mean())
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    if len(ranked) > 1 and ranked[1][1] - ranked[0][1] < MIN_GAP:
        return None
    return ranked[0][0]


def render_pages(path: Path) -> list[np.ndarray]:
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(path))
    try:
        out = []
        for i in range(len(doc)):
            page = doc[i]
            w, h = page.get_size()  # ya considera /Rotate
            bmp = page.render(scale=RENDER_SIDE / max(w, h, 1), grayscale=True)
            out.append(np.asarray(bmp.to_pil().convert("L")))
        return out
    finally:
        doc.close()


def classify(label: int, applied: int | None) -> str:
    if applied is None:
        return "dudoso"
    if applied == label:
        return "bien"
    if applied == 0:
        return "sin_corregir"
    return "mal"


# --- ejecución --------------------------------------------------------------------------------

def run_timed(cmd: list[str], env: dict | None = None, timeout: float = 600) -> tuple[int, float, float, str]:
    """(código, segundos de reloj, segundos de CPU de los procesos hijos, stderr)."""
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        code, err = p.returncode, p.stderr
    except subprocess.TimeoutExpired:
        code, err = -1, "timeout"
    wall = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    return code, wall, cpu, err


def load_golden(labels_csv: Path, splits_csv: Path, split: str) -> dict[str, dict[int, dict]]:
    with splits_csv.open(newline="", encoding="utf-8") as f:
        ids = {r["pdf_id"] for r in csv.DictReader(f) if r["split"] == split}
    last: dict[tuple[str, int], dict] = {}
    with labels_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["pdf_id"] in ids:
                last[(r["pdf_id"], int(r["page"]))] = r
    golden: dict[str, dict[int, dict]] = defaultdict(dict)
    for (pid, page), r in last.items():
        golden[pid][page] = r
    return dict(sorted(golden.items()))


def find_pdfium() -> str | None:
    if os.environ.get("PDFIUM_LIB"):
        return os.environ["PDFIUM_LIB"]
    import pypdfium2_raw
    lib = Path(pypdfium2_raw.__file__).parent / "libpdfium.so"
    return str(lib) if lib.exists() else None


def summarize(rows: list[dict], pdfs: list[dict], tools: list[str]) -> str:
    lines = ["| herramienta | páginas | bien | mal | sin corregir | dudoso | PDFs con error |"
             " reloj mediana | reloj p95 | reloj total | CPU total |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in tools:
        c = Counter(r[t] for r in rows)
        n = len(rows)
        walls = sorted(p[f"{t}_wall"] for p in pdfs)
        cpu = sum(p[f"{t}_cpu"] for p in pdfs)
        errs = sum(1 for p in pdfs if p[f"{t}_code"] != 0)
        pct = lambda k: f"{c[k]} ({100 * c[k] / n:.1f} %)" if n else "0"
        p95 = walls[min(len(walls) - 1, int(len(walls) * 0.95))] if walls else 0
        lines.append(f"| {t} | {n} | {pct('bien')} | {pct('mal')} | {pct('sin_corregir')} | {pct('dudoso')} "
                     f"| {errs} | {statistics.median(walls) * 1000 if walls else 0:.0f} ms | {p95 * 1000:.0f} ms "
                     f"| {sum(walls):.0f} s | {cpu:.0f} s |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--labels", type=Path, default=REPO / "data" / "labels.csv")
    ap.add_argument("--splits", type=Path, default=REPO / "data" / "splits" / "split.csv")
    ap.add_argument("--raw", type=Path, default=REPO / "data" / "raw")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "bench")
    ap.add_argument("--rotator", type=Path, default=REPO / "rotator" / "target" / "release" / "page-rotator")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0, help="solo los primeros N PDFs (prueba rápida)")
    ap.add_argument("--no-ocrmypdf", action="store_true", help="medir solo page-rotator")
    args = ap.parse_args(argv)

    if not args.rotator.exists():
        sys.exit(f"No existe {args.rotator}; compila con: cd rotator && cargo build --release")
    tools = ["page_rotator"] + ([] if args.no_ocrmypdf else ["ocrmypdf"])
    if "ocrmypdf" in tools and shutil.which("ocrmypdf") is None:
        sys.exit("ocrmypdf no está instalado (usa --no-ocrmypdf)")
    env = dict(os.environ)
    if (lib := find_pdfium()) is not None:
        env.setdefault("PDFIUM_LIB", lib)

    golden = load_golden(args.labels, args.splits, args.split)
    ids = list(golden)[: args.limit or None]
    print(f"{len(ids)} PDFs de {args.split!r}, {sum(len(golden[i]) for i in ids)} páginas etiquetadas")
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    pdfs: list[dict] = []
    with tempfile.TemporaryDirectory(dir=args.out) as tmp:
        for n, pid in enumerate(ids, 1):
            src = args.raw / f"{pid}.pdf"
            before = render_pages(src)
            outs = {t: Path(tmp) / f"{t}.pdf" for t in tools}
            cmds = {"page_rotator": [str(args.rotator), "-q", "--threshold", str(args.threshold)],
                    "ocrmypdf": OCRMYPDF}
            prec = {"pdf_id": pid, "pages": len(before)}
            after: dict[str, list[np.ndarray] | None] = {}
            for t in tools:
                outs[t].unlink(missing_ok=True)
                code, wall, cpu, err = run_timed(cmds[t] + [str(src), str(outs[t])], env)
                prec.update({f"{t}_code": code, f"{t}_wall": wall, f"{t}_cpu": cpu,
                             f"{t}_error": err.strip().splitlines()[-1][:200] if code and err.strip() else ""})
                pages = render_pages(outs[t]) if code == 0 and outs[t].exists() else None
                if pages is not None and len(pages) != len(before):
                    prec[f"{t}_error"] = f"cambió el número de páginas ({len(before)} → {len(pages)})"
                    pages = None
                after[t] = pages
            pdfs.append(prec)
            for page, lab in sorted(golden[pid].items()):
                if lab["status"] != "ok" or page > len(before):
                    continue
                row = {"pdf_id": pid, "page": page, "label": int(lab["rotation"])}
                for t in tools:
                    applied = (applied_rotation(before[page - 1], after[t][page - 1])
                               if after[t] is not None else 0)  # si la herramienta falla, la página queda igual
                    row[f"{t}_applied"] = "" if applied is None else applied
                    row[t] = classify(row["label"], applied)
                rows.append(row)
            if n % 10 == 0 or n == len(ids):
                status = "  ".join(f"{t}: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r[t] for r in rows).items()))
                                   for t in tools)
                print(f"[{n}/{len(ids)}] {status}", flush=True)

    with (args.out / "pages.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["pdf_id"])
        w.writeheader()
        w.writerows(rows)
    with (args.out / "pdfs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pdfs[0]) if pdfs else ["pdf_id"])
        w.writeheader()
        w.writerows(pdfs)

    table = summarize(rows, pdfs, tools)
    report = [f"# Benchmark — partición {args.split}, umbral {args.threshold}", "",
              f"{len(pdfs)} PDFs, {len(rows)} páginas etiquetadas `ok`.", "", table, ""]
    for t in tools:
        bad = [r for r in rows if r[t] in ("mal", "sin_corregir", "dudoso")]
        if bad:
            report += [f"## {t}: páginas no resueltas ({len(bad)})", "",
                       "| pdf_id | página | etiqueta | aplicado | resultado |", "|---|---|---|---|---|"]
            report += [f"| {r['pdf_id']} | {r['page']} | {r['label']} | {r[f'{t}_applied']} | {r[t]} |" for r in bad]
            report.append("")
        errs = [p for p in pdfs if p[f"{t}_code"] != 0 or p[f"{t}_error"]]
        if errs:
            report += [f"## {t}: PDFs con error ({len(errs)})", ""]
            report += [f"- {p['pdf_id']}: código {p[f'{t}_code']} {p[f'{t}_error']}" for p in errs]
            report.append("")
    (args.out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print()
    print(table)
    print(f"\nReporte: {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
