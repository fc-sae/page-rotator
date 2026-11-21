#!/usr/bin/env python3
"""Entrena el clasificador de orientación (etapas 3c y 4).

Usa las páginas `ok` de data/labels.csv: cada una se endereza con su etiqueta y se gira 90°·k para
producir las 4 clases. La partición es por PDF (data/splits/split.csv, se crea una vez).

Salida en data/models/:
  orientation.pt      mejor época por exactitud de validación (pesos + backbone + tamaño)
  metrics.json        métricas de validación de esa época (y de test con --test)

    uv run python train.py [--epochs 15] [--batch 64] [--test]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import orient

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"


class TrainSet(Dataset):
    """Página derecha + aumentos + giro aleatorio; devuelve (imagen, índice de clase)."""

    def __init__(self, items: list[orient.Item], size: int, augment: bool = True):
        self.items, self.size, self.augment = items, size, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
        with Image.open(it.path) as img:
            up = orient.straighten(img.convert("L"), it.rotation)
        if self.augment:
            up = orient.augment(up, rng)
        k = int(rng.integers(0, 4))
        return torch.from_numpy(orient.to_input(orient.turn(up, orient.CLASSES[k]), self.size)), k


class EvalSet(Dataset):
    """Cada página en sus 4 giros, sin aumentos; devuelve (imagen, clase, índice de página)."""

    def __init__(self, items: list[orient.Item], size: int):
        self.items, self.size = items, size

    def __len__(self):
        return 4 * len(self.items)

    def __getitem__(self, j):
        i, k = divmod(j, 4)
        it = self.items[i]
        with Image.open(it.path) as img:
            up = orient.straighten(img.convert("L"), it.rotation)
        return torch.from_numpy(orient.to_input(orient.turn(up, orient.CLASSES[k]), self.size)), k, i


def sample_weights(items: list[orient.Item]) -> list[float]:
    """Peso ∝ 1/√n_tipo para que los tipos raros no se pierdan."""
    n = Counter(it.type for it in items)
    return [1 / math.sqrt(n[it.type]) for it in items]


@torch.no_grad()
def predict_views(model, items: list[orient.Item], size: int, device: str, batch: int,
                  workers: int) -> np.ndarray:
    """Probabilidades [n_páginas, 4 giros, 4 clases]."""
    model.eval()
    probs = np.zeros((len(items), 4, 4), dtype=np.float32)
    loader = DataLoader(EvalSet(items, size), batch_size=batch, num_workers=workers)
    for x, k, i in loader:
        with torch.autocast(device_type=device.split(":")[0], enabled=device.startswith("cuda")):
            logits = model(x.to(device))
        probs[i.numpy(), k.numpy()] = F.softmax(logits.float(), dim=1).cpu().numpy()
    return probs


def _rule_counts(outcomes: list[tuple[int, int]]) -> dict:
    """outcomes = [(verdadera, decidida)] → tasas de acierto, giro equivocado y sin corregir."""
    n = len(outcomes)
    wrong = sum(1 for c, r in outcomes if r != c and r != 0)
    missed = sum(1 for c, r in outcomes if r == 0 and c != 0)
    return {"n": n, "correct": (n - wrong - missed) / n if n else None,
            "wrong": wrong / n if n else None, "missed": missed / n if n else None,
            "wrong_count": wrong, "missed_count": missed}


def metrics(items: list[orient.Item], probs: np.ndarray, threshold: float) -> dict:
    """Exactitud por giro, por clase y por tipo; regla de la CLI (vista + vista a 180°) con umbral.

    `real` usa solo la vista tal como llegó la página (su etiqueta): es la distribución de producción.
    """
    if not items:
        return {"pages": 0}
    pred = probs.argmax(axis=2)                       # [n, 4]
    truth = np.tile(np.arange(4), (len(items), 1))
    conf = np.zeros((4, 4), dtype=int)
    for t, p in zip(truth.ravel(), pred.ravel()):
        conf[t, p] += 1
    by_type: dict[str, list[bool]] = defaultdict(list)
    for i, it in enumerate(items):
        by_type[it.type].extend((pred[i] == np.arange(4)).tolist())
    all_out, real_out, real_by_type = [], [], defaultdict(list)
    for i, it in enumerate(items):
        for k in range(4):
            rot, _, _ = orient.decide(probs[i, k], probs[i, (k + 2) % 4], threshold)
            o = (orient.CLASSES[k], rot)
            all_out.append(o)
            if orient.CLASSES[k] == it.rotation:
                real_out.append(o)
                real_by_type[it.type].append(o)
    return {
        "pages": len(items),
        "accuracy": float((pred == truth).mean()),
        "per_class": {str(c): float(conf[k, k] / conf[k].sum()) for k, c in enumerate(orient.CLASSES)},
        "confusion": conf.tolist(),
        "per_type": {t: {"views": len(v), "accuracy": float(np.mean(v))} for t, v in sorted(by_type.items())},
        "rule": {"threshold": threshold, "all_views": _rule_counts(all_out), "real": _rule_counts(real_out),
                 "real_per_type": {t: _rule_counts(v) for t, v in sorted(real_by_type.items())}},
    }


def train(items_by_split: dict[str, list[orient.Item]], out_dir: Path, backbone: str, pretrained: bool,
          size: int, epochs: int, batch: int, lr: float, device: str, workers: int,
          threshold: float = 0.9, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_items, val_items = items_by_split["train"], items_by_split["val"]
    if not train_items:
        raise ValueError("no hay páginas etiquetadas en train")
    torch.manual_seed(0)
    model = orient.build_model(backbone, pretrained).to(device)
    sampler = WeightedRandomSampler(sample_weights(train_items), num_samples=len(train_items), replacement=True)
    loader = DataLoader(TrainSet(train_items, size), batch_size=batch, sampler=sampler, num_workers=workers,
                        drop_last=len(train_items) > batch, persistent_workers=workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * len(loader)
    warm = max(1, len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best, best_metrics = -1.0, {}
    for ep in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = F.cross_entropy(model(x), y, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, n = total + loss.item() * len(y), n + len(y)
        m = metrics(val_items, predict_views(model, val_items, size, device, batch, workers), threshold) \
            if val_items else {"pages": 0}
        acc = m.get("accuracy", ep)  # sin validación: se queda la última época
        rule = m.get("rule", {}).get("real", {})
        log(f"época {ep}/{epochs}  pérdida {total / max(n, 1):.4f}  val {acc if val_items else '-'}"
            f"  real: mal {rule.get('wrong_count', '-')} sin corregir {rule.get('missed_count', '-')}"
            f"  ({time.time() - t0:.0f}s)")
        if acc > best:
            best, best_metrics = acc, {"epoch": ep, "val": m}
            torch.save({"state_dict": model.state_dict(), "backbone": backbone, "size": size,
                        "classes": orient.CLASSES, "epoch": ep}, out_dir / "orientation.pt")
    return best_metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=DATA / "labels.csv")
    ap.add_argument("--pages", type=Path, default=DATA / "pages")
    ap.add_argument("--clusters", type=Path, default=DATA / "clusters" / "clusters.csv")
    ap.add_argument("--types", type=Path, default=DATA / "clusters" / "types.csv")
    ap.add_argument("--splits", type=Path, default=DATA / "splits" / "split.csv")
    ap.add_argument("--out", type=Path, default=DATA / "models")
    ap.add_argument("--backbone", default=orient.DEFAULT_BACKBONE)
    ap.add_argument("--no-pretrained", action="store_true", help="pesos aleatorios (solo pruebas)")
    ap.add_argument("--size", type=int, default=orient.SIZE)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--test", action="store_true", help="también evalúa en test (úsalo poco: es el test dorado)")
    args = ap.parse_args(argv)

    if not args.labels.exists():
        print(f"No existe {args.labels}: etiqueta primero con label.py", file=sys.stderr)
        return 1
    types = orient.page_types(args.clusters, args.types)
    items = orient.load_labeled(args.labels, args.pages, types)
    splits = orient.load_or_make_splits(args.splits, args.pages, types)
    by_split: dict[str, list[orient.Item]] = {"train": [], "val": [], "test": []}
    for it in items:
        by_split[splits.get(it.pdf_id, "train")].append(it)
    print(f"páginas etiquetadas: {len(items)}  train {len(by_split['train'])}  val {len(by_split['val'])}"
          f"  test {len(by_split['test'])}  dispositivo {args.device}")
    try:
        result = train(by_split, args.out, args.backbone, not args.no_pretrained, args.size, args.epochs,
                       args.batch, args.lr, args.device, args.workers, args.threshold)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    if args.test and by_split["test"]:
        model, _ = orient.load_checkpoint(args.out / "orientation.pt", args.device)
        result["test"] = metrics(by_split["test"], predict_views(model, by_split["test"], args.size, args.device,
                                                                 args.batch, args.workers), args.threshold)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    val = result.get("val", {})
    print(f"mejor época {result.get('epoch')}: exactitud val {val.get('accuracy')}  → {args.out / 'orientation.pt'}")
    if "rule" in val:
        print("regla CLI en val (vista real):", json.dumps(val["rule"]["real"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
