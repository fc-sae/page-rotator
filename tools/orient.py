"""Piezas comunes del clasificador de orientación (entrenamiento, export y predicción).

Convención (igual que labels.csv y la CLI): la clase es la corrección, en grados en sentido horario,
que hay que aplicar a la página *tal como se ve* para dejarla derecha: 0, 90, 180 o 270.

Entrada del modelo (contrato con la CLI en Rust): float32 [N, 1, SIZE, SIZE], valores 0–1 (1 = blanco).
La página se escala con el lado largo a SIZE (bilineal) y se centra sobre blanco con
offset (SIZE − lado) // 2. La normalización va dentro del modelo. Salida: logits [N, 4].
"""

from __future__ import annotations

import csv
import hashlib
import io
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

CLASSES = [0, 90, 180, 270]
SIZE = 384
DEFAULT_BACKBONE = "mobilenetv3_large_100"
SPLITS = (("train", 0.8), ("val", 0.1), ("test", 0.1))


# --- geometría ------------------------------------------------------------------------------

def straighten(img: Image.Image, rotation: int) -> Image.Image:
    """Aplica `rotation` grados en sentido horario (PIL gira en sentido antihorario)."""
    return img.rotate(-rotation, expand=True) if rotation % 360 else img.copy()


def turn(upright: Image.Image, rotation: int) -> Image.Image:
    """Inversa de `straighten`: la imagen que necesitaría la corrección `rotation`."""
    return upright.rotate(rotation, expand=True) if rotation % 360 else upright.copy()


def letterbox(img: Image.Image, size: int = SIZE) -> Image.Image:
    """Lado largo a `size` (bilineal) centrado en un cuadrado blanco, gris (modo L)."""
    img = img.convert("L")
    scale = size / max(img.size)
    w, h = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    img = img.resize((w, h), Image.BILINEAR)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(img, ((size - w) // 2, (size - h) // 2))
    return canvas


def to_input(img: Image.Image, size: int = SIZE) -> np.ndarray:
    """Imagen → arreglo [1, size, size] float32 en 0–1."""
    return (np.asarray(letterbox(img, size), dtype=np.float32) / 255.0)[None]


# --- aumentos -------------------------------------------------------------------------------

def augment(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Ruido de escaneo sobre la página derecha (antes del giro de 90°·k)."""
    img = img.convert("L")
    # reducir primero: los aumentos sobre 150 dpi completos son lentos e innecesarios
    scale = (SIZE * 1.5) / max(img.size)
    if scale < 1:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BILINEAR)
    if rng.random() < 0.8:  # inclinación ±5°
        img = img.rotate(float(rng.uniform(-5, 5)), resample=Image.BILINEAR, expand=True, fillcolor=255)
    if rng.random() < 0.5:  # recorte 85–100 %
        s = float(rng.uniform(0.85, 1.0))
        w, h = img.size
        cw, ch = max(1, int(w * s)), max(1, int(h * s))
        x, y = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
        img = img.crop((x, y, x + cw, y + ch))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 1.5))))
    arr = np.asarray(img, dtype=np.float32)
    if rng.random() < 0.5:  # contraste y brillo
        arr = (arr - 128) * float(rng.uniform(0.7, 1.3)) + 128 + float(rng.uniform(-30, 30))
    if rng.random() < 0.3:  # ruido
        arr = arr + rng.normal(0, float(rng.uniform(2, 12)), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L")
    if rng.random() < 0.5:  # JPEG
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(rng.integers(30, 91)))
        buf.seek(0)
        img = Image.open(buf).convert("L")
    return img


# --- modelo ---------------------------------------------------------------------------------

class Orientation(torch.nn.Module):
    """Backbone de timm con la normalización incluida: entrada 0–1, salida logits de 4 clases."""

    def __init__(self, backbone: torch.nn.Module, mean: float, std: float):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone((x - self.mean) / self.std)


def build_model(name: str = DEFAULT_BACKBONE, pretrained: bool = True) -> Orientation:
    import timm
    backbone = timm.create_model(name, pretrained=pretrained, num_classes=len(CLASSES), in_chans=1)
    cfg = backbone.pretrained_cfg or {}
    mean = float(np.mean(cfg.get("mean", (0.5,))))
    std = float(np.mean(cfg.get("std", (0.5,))))
    return Orientation(backbone, mean, std)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[Orientation, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


# --- regla de decisión (igual que la CLI) -----------------------------------------------------

def decide(p: np.ndarray, p180: np.ndarray, threshold: float) -> tuple[int, float, bool]:
    """`p` = probabilidades sobre la imagen; `p180` = sobre la imagen girada 180°.

    Devuelve (corrección, confianza, consistente). Si las dos vistas no coinciden o la confianza
    media es menor al umbral, la corrección es 0 (no tocar la página).
    """
    a, b = int(np.argmax(p)), int(np.argmax(p180))
    consistent = b == (a + 2) % 4
    conf = float((p[a] + p180[b]) / 2) if consistent else float(min(p[a], p180[b]))
    rot = CLASSES[a] if consistent and conf >= threshold else 0
    return rot, conf, consistent


# --- datos ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Item:
    pdf_id: str
    page: int
    path: Path
    rotation: int
    type: str


def read_manifest(pages_dir: Path) -> list[dict]:
    with (pages_dir / "manifest.csv").open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["status"] == "ok"]


def page_types(clusters_csv: Path | None, types_csv: Path | None) -> dict[tuple[str, int], str]:
    """(pdf_id, page) → tipo: nombre de types.csv si está lleno, si no `c<cluster>`."""
    if clusters_csv is None or not Path(clusters_csv).exists():
        return {}
    names: dict[str, str] = {}
    if types_csv is not None and Path(types_csv).exists():
        with Path(types_csv).open(newline="", encoding="utf-8") as f:
            names = {r["cluster"]: r["type"].strip() for r in csv.DictReader(f) if r.get("type", "").strip()}
    with Path(clusters_csv).open(newline="", encoding="utf-8") as f:
        return {(r["pdf_id"], int(r["page"])): names.get(r["cluster"], f"c{r['cluster']}")
                for r in csv.DictReader(f)}


def load_labeled(labels_csv: Path, pages_dir: Path, types: dict[tuple[str, int], str]) -> list[Item]:
    """Páginas etiquetadas `ok` (la última fila de cada página gana) con su PNG existente."""
    last: dict[tuple[str, int], dict] = {}
    with Path(labels_csv).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            last[(r["pdf_id"], int(r["page"]))] = r
    paths = {(r["pdf_id"], int(r["page"])): pages_dir / r["path"] for r in read_manifest(pages_dir)}
    items = []
    for key, r in sorted(last.items()):
        if r["status"] != "ok" or key not in paths or not paths[key].exists():
            continue
        items.append(Item(key[0], key[1], paths[key], int(r["rotation"]), types.get(key, "sin_tipo")))
    return items


def _hash_split(pdf_id: str) -> str:
    u = int(hashlib.sha1(pdf_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, frac in SPLITS:
        acc += frac
        if u < acc:
            return name
    return SPLITS[-1][0]


def load_or_make_splits(out: Path, pages_dir: Path, types: dict[tuple[str, int], str],
                        seed: int = 0) -> dict[str, str]:
    """pdf_id → train/val/test, por PDF y estratificado por tipo (el de su primera página).

    El archivo se crea una vez; los PDFs nuevos se agregan por hash del ID sin mover los existentes.
    """
    out = Path(out)
    splits: dict[str, str] = {}
    if out.exists():
        with out.open(newline="", encoding="utf-8") as f:
            splits = {r["pdf_id"]: r["split"] for r in csv.DictReader(f)}
    pdf_type: dict[str, str] = {}
    for r in read_manifest(pages_dir):
        pdf_type.setdefault(r["pdf_id"], types.get((r["pdf_id"], int(r["page"])), "sin_tipo"))
    new = [p for p in pdf_type if p not in splits]
    if not new:
        return splits
    if not splits:  # primera vez: estratificado
        by_type: dict[str, list[str]] = defaultdict(list)
        for p in sorted(new):
            by_type[pdf_type[p]].append(p)
        rng = random.Random(seed)
        offset = 0.0  # reparte los sobrantes entre tipos para no favorecer siempre a train
        for t in sorted(by_type):
            ids = by_type[t]
            rng.shuffle(ids)
            n = len(ids)
            for i, p in enumerate(ids):
                u = ((i + offset) % n) / n
                splits[p] = "train" if u < 0.8 else ("val" if u < 0.9 else "test")
            offset += 0.37
    else:
        for p in new:
            splits[p] = _hash_split(p)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pdf_id", "split"])
        w.writerows(sorted(splits.items()))
    tmp.replace(out)
    return splits
