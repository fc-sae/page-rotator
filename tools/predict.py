#!/usr/bin/env python3
"""Predice la orientación de todas las páginas con el modelo entrenado (etapa 3c, bootstrap).

Cada página se evalúa tal como se ve y girada 180°, con la misma regla que la CLI (orient.decide).

Salida data/predictions.csv:
  pdf_id, page, rotation (corrección más probable), confidence, consistent, decided (lo que haría la CLI
  con --threshold), label (etiqueta de labels.csv si existe), split

Después:  uv run python label.py --predictions ../data/predictions.csv
revisa primero las páginas sin etiqueta de menor confianza, cada una ya con la rotación predicha.

    uv run python predict.py [--threshold 0.9]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import orient

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
FIELDS = ["pdf_id", "page", "rotation", "confidence", "consistent", "decided", "label", "split"]


class Shown(Dataset):
    """Páginas tal como se ven: devuelve (imagen, imagen girada 180°, índice)."""

    def __init__(self, paths: list[Path], size: int):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            x = orient.to_input(img, self.size)
        return torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(x[:, ::-1, ::-1])), i


@torch.no_grad()
def predict(model, paths: list[Path], size: int, device: str, batch: int = 64,
            workers: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidades [n, 4] sobre la imagen y sobre la imagen girada 180°."""
    model.eval()
    p = np.zeros((len(paths), 4), np.float32)
    p180 = np.zeros((len(paths), 4), np.float32)
    for x, x180, i in DataLoader(Shown(paths, size), batch_size=batch, num_workers=workers):
        both = torch.cat([x, x180]).to(device)
        with torch.autocast(device_type=device.split(":")[0], enabled=device.startswith("cuda")):
            probs = F.softmax(model(both).float(), dim=1).cpu().numpy()
        p[i.numpy()], p180[i.numpy()] = probs[: len(i)], probs[len(i):]
    return p, p180


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=DATA / "models" / "orientation.pt")
    ap.add_argument("--pages", type=Path, default=DATA / "pages")
    ap.add_argument("--labels", type=Path, default=DATA / "labels.csv")
    ap.add_argument("--splits", type=Path, default=DATA / "splits" / "split.csv")
    ap.add_argument("--out", type=Path, default=DATA / "predictions.csv")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    if not args.ckpt.exists():
        print(f"No existe {args.ckpt}: entrena primero con train.py", file=sys.stderr)
        return 1

    rows = orient.read_manifest(args.pages)
    labels: dict[tuple[str, int], dict] = {}
    if args.labels.exists():
        with args.labels.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                labels[(r["pdf_id"], int(r["page"]))] = r
    splits: dict[str, str] = {}
    if args.splits.exists():
        with args.splits.open(newline="", encoding="utf-8") as f:
            splits = {r["pdf_id"]: r["split"] for r in csv.DictReader(f)}

    model, ckpt = orient.load_checkpoint(args.ckpt, args.device)
    p, p180 = predict(model, [args.pages / r["path"] for r in rows], int(ckpt["size"]), args.device,
                      args.batch, args.workers)

    out_rows = []
    for r, a, b in zip(rows, p, p180):
        key = (r["pdf_id"], int(r["page"]))
        decided, conf, consistent = orient.decide(a, b, args.threshold)
        lab = labels.get(key)
        out_rows.append({"pdf_id": key[0], "page": key[1], "rotation": orient.CLASSES[int(np.argmax(a))],
                         "confidence": f"{conf:.4f}", "consistent": int(consistent), "decided": decided,
                         "label": "" if lab is None else (lab["rotation"] if lab["status"] == "ok" else "skip"),
                         "split": splits.get(key[0], "")})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".part")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    tmp.replace(args.out)

    unl = [r for r in out_rows if r["label"] == ""]
    conf = np.array([float(r["confidence"]) for r in unl]) if unl else np.zeros(0)
    print(f"{len(out_rows)} páginas → {args.out}")
    print(f"sin etiqueta: {len(unl)}  rotación predicha: {dict(sorted(Counter(r['rotation'] for r in unl).items()))}")
    for lo, hi in ((0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.01)):
        print(f"  confianza {lo:.2f}–{min(hi, 1):.2f}: {int(((conf >= lo) & (conf < hi)).sum())}")
    for split in ("train", "val", "test"):
        lab = [r for r in out_rows if r["label"] not in ("", "skip") and r["split"] == split]
        bad = [r for r in lab if int(r["label"]) != r["rotation"]]
        if lab:
            print(f"etiquetadas en {split}: {len(lab)}, el modelo no coincide en {len(bad)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
