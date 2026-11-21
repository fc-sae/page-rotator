import numpy as np
from PIL import Image, ImageDraw

import bench


def page(w=300, h=400):
    """Página vertical con texto simulado: barra arriba y bloque a la izquierda (no simétrica)."""
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 30], fill=0)
    d.rectangle([0, h // 2, 40, h - 20], fill=0)
    for y in range(60, h // 2, 20):
        d.rectangle([60, y, w - 30, y + 6], fill=60)
    return np.asarray(img)


def turned(a, rot):
    """Gira `rot` grados en sentido horario (como la corrección)."""
    return np.asarray(Image.fromarray(a).rotate(-rot, expand=True))


def test_applied_rotation_recovers_each_turn():
    a = page()
    for rot in bench.ROTATIONS:
        assert bench.applied_rotation(a, turned(a, rot)) == rot


def test_applied_rotation_tolerates_rescale_and_noise():
    a = page()
    b = turned(a, 90)
    b = np.asarray(Image.fromarray(b).resize((b.shape[1] * 3 // 2, b.shape[0] * 3 // 2)))
    rng = np.random.default_rng(0)
    b = np.clip(b + rng.normal(0, 8, b.shape), 0, 255).astype(np.uint8)
    assert bench.applied_rotation(a, b) == 90


def test_blank_page_is_doubtful():
    blank = np.full((400, 300), 255, np.uint8)
    assert bench.applied_rotation(blank, blank) is None


def test_classify():
    assert bench.classify(90, 90) == "bien"
    assert bench.classify(0, 0) == "bien"
    assert bench.classify(90, 0) == "sin_corregir"
    assert bench.classify(90, 270) == "mal"
    assert bench.classify(0, 180) == "mal"
    assert bench.classify(90, None) == "dudoso"
