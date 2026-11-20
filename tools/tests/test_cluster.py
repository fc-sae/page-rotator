import csv

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

import cluster


def synthetic_page(kind: int, seed: int, size=(170, 220)) -> Image.Image:
    """Página falsa: cada `kind` tiene un diseño distinto (barra, bloques, círculo)."""
    rng = np.random.default_rng(seed)
    img = Image.new("L", size, 255)
    d = ImageDraw.Draw(img)
    w, h = size
    if kind == 0:
        d.rectangle((0, 0, w, 30), fill=0)
    elif kind == 1:
        for y in range(40, h - 20, 20):
            d.rectangle((15, y, w - 15, y + 6), fill=60)
    else:
        d.ellipse((w // 4, h // 4, 3 * w // 4, h // 2), fill=30)
    for _ in range(20):  # ruido para que no sean idénticas
        x, y = rng.integers(0, w - 4), rng.integers(0, h - 4)
        d.rectangle((x, y, x + 3, y + 3), fill=int(rng.integers(100, 200)))
    return img


def test_letterbox_square_white_padding():
    out = cluster.letterbox(Image.new("L", (100, 200), 0), 64)
    assert out.size == (64, 64) and out.mode == "L"
    assert out.getpixel((0, 32)) == 255  # relleno blanco a los lados
    assert out.getpixel((32, 32)) == 0


def test_embedding_is_rotation_invariant():
    model, mean, std = cluster.load_model("resnet18", 64, pretrained=False, device="cpu")
    page = synthetic_page(0, 1)
    xs = [torch.from_numpy(np.asarray(cluster.letterbox(page.rotate(a, expand=True), 64),
                                      dtype=np.float32) / 255)[None] for a in (0, 90, 180, 270)]
    e = cluster.embed(model, mean, std, torch.stack(xs))
    sims = (e @ e[0]).numpy()
    assert sims.min() > 0.99, sims


def test_pick_samples_spans_typical_to_atypical():
    dist = np.array([0.5, 0.1, 0.9, 0.3, 0.7])
    idx = np.arange(5)
    assert cluster.pick_samples(idx, dist, 3).tolist() == [1, 0, 2]
    assert cluster.pick_samples(idx, dist, 10).tolist() == [1, 3, 0, 4, 2]


@pytest.fixture
def pages_dir(tmp_path):
    pages = tmp_path / "pages"
    rows = []
    for n in range(12):
        pdf_id = f"pdf{n:02d}"
        (pages / pdf_id).mkdir(parents=True)
        img = synthetic_page(n % 3, n).rotate(90 * (n % 4), expand=True)
        img.save(pages / pdf_id / "001.png")
        rows.append({"pdf_id": pdf_id, "page": 1, "status": "ok", "path": f"{pdf_id}/001.png"})
    rows.append({"pdf_id": "roto", "page": "", "status": "error", "path": ""})
    with (pages / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdf_id", "page", "status", "path"])
        w.writeheader()
        w.writerows(rows)
    return pages


def run(pages, out, *extra):
    return cluster.main(["--pages", str(pages), "--out", str(out), "--k", "3", "--model", "resnet18",
                         "--no-pretrained", "--device", "cpu", "--size", "64", "--workers", "0", *extra])


def test_main_writes_outputs_and_reuses_cache(pages_dir, tmp_path, capsys):
    out = tmp_path / "clusters"
    assert run(pages_dir, out) == 0

    with (out / "clusters.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12 and {r["cluster"] for r in rows} <= {"0", "1", "2"}
    assert len(list((out / "sheets").glob("cluster_*.jpg"))) == len({r["cluster"] for r in rows})
    assert "sheets/cluster_" in (out / "index.html").read_text()
    with (out / "types.csv").open() as f:
        types = list(csv.DictReader(f))
    assert sum(int(t["pages"]) for t in types) == 12 and all(t["type"] == "" for t in types)

    # El usuario llena types.csv; la segunda corrida no recalcula ni lo sobrescribe.
    (out / "types.csv").write_text("cluster,pages,pdfs,type\n0,4,4,ielts\n")
    capsys.readouterr()
    assert run(pages_dir, out) == 0
    assert "12 en caché, 0 por calcular" in capsys.readouterr().out
    assert "ielts" in (out / "types.csv").read_text()
