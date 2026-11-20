"""Almacén de etiquetas (labels.csv) y cola de páginas por etiquetar.

labels.csv es solo-agregar: la última fila de cada (pdf_id, page) es la vigente.
La cola recorre los clústeres en round-robin para ver variedad de constancias desde el inicio.
"""

from __future__ import annotations

import csv
import os
import random
import re
import threading
from datetime import datetime
from pathlib import Path

from PIL import Image

FIELDS = ["pdf_id", "page", "rotation", "status", "labeled_at"]
ROTATIONS = {0, 90, 180, 270}
STATUSES = {"ok", "skip"}
ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,}$")
THUMB_SIDE = 320
THUMB_QUALITY = 80


class LabelStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.labels: dict[tuple[str, int], dict] = {}
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    row["page"] = int(row["page"])
                    row["rotation"] = int(row["rotation"])
                    self.labels[(row["pdf_id"], row["page"])] = row
        self._new = not (self.path.exists() and self.path.stat().st_size > 0)
        self._f = None

    def add(self, rows: list[dict]) -> None:
        """Valida todas las filas y luego las agrega; si alguna es inválida no escribe nada."""
        now = datetime.now().isoformat(timespec="seconds")
        clean = []
        for r in rows:
            rotation, status = r["rotation"], r["status"]
            if rotation not in ROTATIONS:
                raise ValueError(f"rotación inválida: {rotation!r}")
            if status not in STATUSES:
                raise ValueError(f"estado inválido: {status!r}")
            clean.append({"pdf_id": str(r["pdf_id"]), "page": int(r["page"]),
                          "rotation": int(rotation), "status": status, "labeled_at": now})
        if self._f is None:
            self._f = self.path.open("a", newline="", encoding="utf-8")
        w = csv.DictWriter(self._f, fieldnames=FIELDS)
        if self._new:
            w.writeheader()
            self._new = False
        for r in clean:
            w.writerow(r)
            self.labels[(r["pdf_id"], r["page"])] = r
        self._f.flush()

    def __contains__(self, key: tuple[str, int]) -> bool:
        return key in self.labels

    def __len__(self) -> int:
        return len(self.labels)


def build_queue(pages_manifest: Path, clusters_csv: Path | None, seed: int = 0) -> list[tuple[str, int]]:
    """Páginas `ok` en round-robin por clúster (orden aleatorio fijo dentro de cada uno)."""
    cluster_of: dict[tuple[str, int], int] = {}
    if clusters_csv is not None and Path(clusters_csv).exists():
        with Path(clusters_csv).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cluster_of[(row["pdf_id"], int(row["page"]))] = int(row["cluster"])

    groups: dict[int | str, list[tuple[str, int]]] = {}
    with Path(pages_manifest).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok":
                continue
            key = (row["pdf_id"], int(row["page"]))
            groups.setdefault(cluster_of.get(key, "sin_cluster"), []).append(key)

    rng = random.Random(seed)
    ordered = sorted((g for g in groups if g != "sin_cluster"))
    if "sin_cluster" in groups:
        ordered.append("sin_cluster")
    lists = []
    for g in ordered:
        items = sorted(groups[g])
        rng.shuffle(items)
        lists.append(items)

    queue: list[tuple[str, int]] = []
    for i in range(max((len(x) for x in lists), default=0)):
        queue.extend(x[i] for x in lists if i < len(x))
    return queue


def image_path(pages_dir: Path, thumbs_dir: Path, pdf_id: str, page: int, size: str) -> Path | None:
    """Ruta de la imagen de una página (`full` = PNG original, `thumb` = JPEG de 320 px).

    Es la frontera de seguridad del servidor: valida el id contra ID_RE (sin separadores
    ni `..`), así que nunca devuelve rutas fuera de pages_dir/thumbs_dir.
    Devuelve None si el id, la página o el tamaño son inválidos o el PNG no existe.
    """
    if size not in ("thumb", "full") or not isinstance(pdf_id, str) or not ID_RE.fullmatch(pdf_id):
        return None
    if not isinstance(page, int) or page < 1:
        return None
    png = Path(pages_dir) / pdf_id / f"{page:03d}.png"
    if not png.is_file():
        return None
    if size == "full":
        return png
    thumb = Path(thumbs_dir) / pdf_id / f"{page:03d}.jpg"
    if not thumb.exists():
        thumb.parent.mkdir(parents=True, exist_ok=True)
        tmp = thumb.with_suffix(f".jpg.{os.getpid()}.{threading.get_ident()}.part")
        try:
            with Image.open(png) as im:
                im = im.convert("L")
                im.thumbnail((THUMB_SIDE, THUMB_SIDE))
                im.save(tmp, format="JPEG", quality=THUMB_QUALITY)
            tmp.replace(thumb)
        finally:
            tmp.unlink(missing_ok=True)
    return thumb
