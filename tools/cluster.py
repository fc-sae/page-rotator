#!/usr/bin/env python3
"""Agrupa las páginas por tipo de constancia para que se puedan nombrar a mano (etapa 3a).

1. Calcula un embedding por página con un modelo preentrenado (DINOv2 small, timm).
   El embedding es el promedio de las 4 rotaciones (0/90/180/270°), así que no depende de la
   orientación: una constancia derecha y la misma girada caen en el mismo grupo.
2. Agrupa con k-means (k≈30, algo más que los ~24 tipos, para que sobren y se fusionen a mano).
3. Genera una hoja de miniaturas por grupo y un index.html para revisarlas.

Salida en data/clusters/:
  embeddings.npz          caché de embeddings (solo se calculan las páginas nuevas)
  clusters.csv            pdf_id, page, cluster, dist (distancia al centro; alta = atípica)
  sheets/cluster_NN.jpg   miniaturas de típicas (arriba) a atípicas (abajo)
  index.html              todas las hojas en una página
  types.csv               cluster → type, para llenar a mano (no se sobrescribe si ya existe)

    uv run python cluster.py [--k 30]
"""

from __future__ import annotations

import argparse
import csv
import html
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "vit_small_patch14_dinov2.lvd142m"


def letterbox(img: Image.Image, size: int) -> Image.Image:
    """Escala el lado largo a `size` y centra en un cuadrado blanco (gris, modo L)."""
    img = img.convert("L")
    scale = size / max(img.size)
    w, h = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    img = img.resize((w, h), Image.BILINEAR)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(img, ((size - w) // 2, (size - h) // 2))
    return canvas


class Pages(Dataset):
    def __init__(self, paths: list[Path], size: int):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            arr = np.asarray(letterbox(img, self.size), dtype=np.float32) / 255.0
        return torch.from_numpy(arr)[None]  # 1×H×W


def load_model(name: str, size: int, pretrained: bool, device: str):
    import timm

    kwargs = {"img_size": size} if name.startswith(("vit", "deit", "eva")) else {}
    model = timm.create_model(name, pretrained=pretrained, num_classes=0, **kwargs).eval().to(device)
    cfg = timm.data.resolve_data_config(model=model)
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    return model, mean, std


@torch.inference_mode()
def embed(model, mean, std, x: torch.Tensor) -> torch.Tensor:
    """x: B×1×H×W en [0,1] → B×D normalizado, invariante a rotaciones de 90°."""
    x = (x.expand(-1, 3, -1, -1) - mean) / std
    with torch.autocast(x.device.type, enabled=x.device.type == "cuda"):
        feats = sum(model(torch.rot90(x, k, dims=(2, 3))).float() for k in range(4)) / 4
    return F.normalize(feats, dim=1)


def compute_embeddings(paths: list[Path], model_name: str, size: int, pretrained: bool,
                       device: str, batch: int, workers: int) -> np.ndarray:
    model, mean, std = load_model(model_name, size, pretrained, device)
    loader = DataLoader(Pages(paths, size), batch_size=batch, num_workers=workers)
    out = []
    for n, x in enumerate(loader, 1):
        out.append(embed(model, mean, std, x.to(device)).cpu().numpy())
        if n % 20 == 0:
            print(f"  {min(n * batch, len(paths))}/{len(paths)} páginas", flush=True)
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def read_pages(pages_dir: Path) -> list[dict]:
    with (pages_dir / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    seen, unique = set(), []
    for r in rows:  # por si una página quedó repetida en el manifiesto
        key = f"{r['pdf_id']}/{r['page']}"
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def update_cache(cache: Path, rows: list[dict], pages_dir: Path, args) -> tuple[list[str], np.ndarray]:
    keys = [f"{r['pdf_id']}/{r['page']}" for r in rows]
    old: dict[str, np.ndarray] = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["model"]) == args.model and int(z["size"]) == args.size:
            old = dict(zip(z["keys"].tolist(), z["emb"]))
        else:
            print("El modelo o el tamaño cambiaron: se recalculan todos los embeddings.")
    todo = [i for i, k in enumerate(keys) if k not in old]
    print(f"{len(keys)} páginas, {len(keys) - len(todo)} en caché, {len(todo)} por calcular "
          f"({args.model}, {args.size}px, {args.device}).")
    if todo:
        new = compute_embeddings([pages_dir / rows[i]["path"] for i in todo], args.model, args.size,
                                 not args.no_pretrained, args.device, args.batch, args.workers)
        for i, e in zip(todo, new):
            old[keys[i]] = e
    emb = np.stack([old[k] for k in keys]).astype(np.float32)
    np.savez(cache, keys=np.array(keys), emb=emb, model=args.model, size=args.size)
    return keys, emb


def kmeans(emb: np.ndarray, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(emb)
    centers = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    dist = 1.0 - np.sum(emb * centers[km.labels_], axis=1)  # distancia coseno al centro
    return km.labels_, dist


def _font(size: int) -> ImageFont.ImageFont:
    """DejaVu Sans si existe (la fuente por defecto de PIL no tiene acentos)."""
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def contact_sheet(paths: list[Path], captions: list[str], title: str, thumb: int = 180, cols: int = 8) -> Image.Image:
    rows = max(1, -(-len(paths) // cols))
    cell_h = thumb + 16
    sheet = Image.new("RGB", (cols * (thumb + 8) + 8, 34 + rows * (cell_h + 8)), "white")
    draw = ImageDraw.Draw(sheet)
    title_font, cap_font = _font(18), _font(11)
    draw.text((8, 8), title, fill="black", font=title_font)
    for i, (p, cap) in enumerate(zip(paths, captions)):
        x, y = 8 + (i % cols) * (thumb + 8), 34 + (i // cols) * (cell_h + 8)
        with Image.open(p) as img:
            img = img.convert("L")
            img.thumbnail((thumb, thumb))
        sheet.paste(img, (x + (thumb - img.width) // 2, y + (thumb - img.height) // 2))
        draw.rectangle((x - 1, y - 1, x + thumb, y + thumb), outline=(200, 200, 200))
        draw.text((x, y + thumb + 2), cap, fill=(90, 90, 90), font=cap_font)
    return sheet


def pick_samples(idx: np.ndarray, dist: np.ndarray, n: int) -> np.ndarray:
    """n muestras repartidas de la más típica a la más atípica."""
    order = idx[np.argsort(dist[idx])]
    if len(order) <= n:
        return order
    return order[np.unique(np.linspace(0, len(order) - 1, n).round().astype(int))]


def write_outputs(out: Path, rows: list[dict], labels: np.ndarray, dist: np.ndarray,
                  pages_dir: Path, per_sheet: int) -> None:
    with (out / "clusters.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pdf_id", "page", "cluster", "dist"])
        for r, c, d in zip(rows, labels, dist):
            w.writerow([r["pdf_id"], r["page"], int(c), f"{d:.4f}"])

    sheets = out / "sheets"
    sheets.mkdir(exist_ok=True)
    for old in sheets.glob("cluster_*.jpg"):
        old.unlink()
    by_cluster: dict[int, np.ndarray] = {}
    for c in np.unique(labels):
        by_cluster[int(c)] = np.flatnonzero(labels == c)
    order = sorted(by_cluster, key=lambda c: -len(by_cluster[c]))

    summary = []
    for c in order:
        idx = by_cluster[c]
        pdfs = len({rows[i]["pdf_id"] for i in idx})
        sample = pick_samples(idx, dist, per_sheet)
        title = f"Cluster {c:02d}: {len(idx)} páginas, {pdfs} PDFs (de típica a atípica)"
        caps = [f"{rows[i]['pdf_id'][:10]} p{rows[i]['page']}" for i in sample]
        sheet = contact_sheet([pages_dir / rows[i]["path"] for i in sample], caps, title)
        sheet.save(sheets / f"cluster_{c:02d}.jpg", quality=85)
        summary.append((c, len(idx), pdfs))

    parts = ["<!doctype html><meta charset=utf-8><title>Clusters</title>",
             "<style>body{font-family:sans-serif;margin:16px}img{max-width:100%;border:1px solid #ccc}"
             "nav a{margin-right:8px}</style>",
             "<h1>Clusters de páginas</h1><p>Llena <code>types.csv</code>: un nombre de tipo por cluster; "
             "usa el mismo nombre para fusionar clusters.</p><nav>"]
    parts += [f'<a href="#c{c}">{c:02d} ({n})</a>' for c, n, _ in summary]
    parts.append("</nav>")
    for c, n, pdfs in summary:
        parts.append(f'<h2 id="c{c}">Cluster {c:02d}: {n} páginas, {pdfs} PDFs</h2>'
                     f'<img src="sheets/cluster_{c:02d}.jpg" loading="lazy">')
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")

    types = out / "types.csv"
    if types.exists():
        print(f"AVISO: {types} ya existe y no se sobrescribió. Si cambiaste --k o hay páginas nuevas, "
              f"los números de cluster pueden no coincidir.", file=sys.stderr)
    else:
        with types.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["cluster", "pages", "pdfs", "type"])
            for c, n, pdfs in sorted(summary):
                w.writerow([c, n, pdfs, ""])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", type=Path, default=REPO / "data" / "pages")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "clusters")
    ap.add_argument("--k", type=int, default=30, help="número de clusters (default 30)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--size", type=int, default=224, help="lado de la imagen de entrada (default 224)")
    ap.add_argument("--no-pretrained", action="store_true", help="pesos aleatorios (solo pruebas)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8, help="procesos que leen PNGs")
    ap.add_argument("--per-sheet", type=int, default=48, help="miniaturas por hoja (default 48)")
    args = ap.parse_args(argv)

    start = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = read_pages(args.pages)
    if len(rows) < args.k:
        sys.exit(f"Hay {len(rows)} páginas, menos que k={args.k}.")
    _, emb = update_cache(args.out / "embeddings.npz", rows, args.pages, args)

    labels, dist = kmeans(emb, args.k)
    write_outputs(args.out, rows, labels, dist, args.pages, args.per_sheet)

    sizes = defaultdict(int)
    for c in labels:
        sizes[int(c)] += 1
    print(f"\n{args.k} clusters en {(time.monotonic() - start):.0f} s. Páginas por cluster: "
          + ", ".join(f"{c:02d}:{n}" for c, n in sorted(sizes.items(), key=lambda x: -x[1])))
    print(f"Revisa {args.out / 'index.html'} y llena {args.out / 'types.csv'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
