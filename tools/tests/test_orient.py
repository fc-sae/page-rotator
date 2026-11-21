import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

import orient


def arrow_page(size=(170, 220)) -> Image.Image:
    """Página asimétrica: barra negra arriba y bloque a la izquierda."""
    img = Image.new("L", size, 255)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, size[0], 25), fill=0)
    d.rectangle((0, 0, 30, size[1] // 2), fill=0)
    return img


def test_to_input_shape_and_range():
    x = orient.to_input(Image.new("L", (100, 200), 0), size=64)
    assert x.shape == (1, 64, 64) and x.dtype == np.float32
    assert x[0, 32, 0] == 1.0 and x[0, 32, 32] == 0.0  # relleno blanco, contenido negro


def test_turn_and_straighten_are_inverse():
    page = arrow_page()
    for c in orient.CLASSES:
        shown = orient.turn(page, c)            # imagen que necesita corrección c
        back = orient.straighten(shown, c)      # aplicar c en sentido horario
        assert back.size == page.size
        assert np.array_equal(np.asarray(back), np.asarray(page))


def test_turn_90_needs_clockwise_correction():
    page = arrow_page()
    shown = orient.turn(page, 90)
    # necesita girar 90° a la derecha: la barra superior quedó a la izquierda
    arr = np.asarray(shown)
    assert arr.shape == (170, 220)  # (alto, ancho) tras girar
    assert (arr[:, :20] == 0).all()


def test_augment_keeps_mode_and_is_deterministic():
    page = arrow_page()
    a = orient.augment(page, np.random.default_rng(3))
    b = orient.augment(page, np.random.default_rng(3))
    assert a.mode == "L"
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_decide_rule():
    p = np.array([0.02, 0.95, 0.02, 0.01])          # 90
    q = np.array([0.01, 0.01, 0.02, 0.96])          # 270 = 90 + 180
    assert orient.decide(p, q, 0.9) == (90, pytest.approx(0.955), True)
    assert orient.decide(p, q, 0.99)[0] == 0         # confianza baja → no tocar
    bad = np.array([0.01, 0.97, 0.01, 0.01])        # 180° predice 90 → inconsistente
    rot, conf, ok = orient.decide(p, bad, 0.5)
    assert rot == 0 and not ok


def test_model_output_shape():
    m = orient.build_model("mobilenetv3_small_050", pretrained=False)
    out = m(torch.rand(2, 1, 64, 64))
    assert out.shape == (2, 4)


def _write(path: Path, header: list[str], rows: list[list]):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def data(tmp_path):
    pages = tmp_path / "pages"
    rows = []
    for i in range(20):
        pid = f"pdf{i:08d}xx"
        (pages / pid).mkdir(parents=True)
        arrow_page().save(pages / pid / "001.png")
        rows.append([pid, 1, "ok", 0, f"{pid}/001.png"])
    _write(pages / "manifest.csv", ["pdf_id", "page", "status", "rotate", "path"], rows)
    clusters = tmp_path / "clusters.csv"
    _write(clusters, ["pdf_id", "page", "cluster", "dist"], [[r[0], 1, i % 2, 0.1] for i, r in enumerate(rows)])
    types = tmp_path / "types.csv"
    _write(types, ["cluster", "pages", "pdfs", "type"], [[0, 10, 10, "toefl"], [1, 10, 10, ""]])
    labels = tmp_path / "labels.csv"
    _write(labels, ["pdf_id", "page", "rotation", "status", "labeled_at"],
           [[rows[0][0], 1, 90, "ok", "t"], [rows[0][0], 1, 0, "ok", "t"],  # la última gana
            [rows[1][0], 1, 270, "ok", "t"], [rows[2][0], 1, 0, "skip", "t"]])
    return tmp_path, pages, clusters, types, labels


def test_page_types(data):
    _, pages, clusters, types, _ = data
    t = orient.page_types(clusters, types)
    assert t[("pdf00000000xx", 1)] == "toefl"
    assert t[("pdf00000001xx", 1)] == "c1"
    assert orient.page_types(None, None) == {}


def test_load_labeled_last_wins_and_skips(data):
    _, pages, clusters, types, labels = data
    items = orient.load_labeled(labels, pages, orient.page_types(clusters, types))
    by = {i.pdf_id: i for i in items}
    assert set(by) == {"pdf00000000xx", "pdf00000001xx"}
    assert by["pdf00000000xx"].rotation == 0 and by["pdf00000000xx"].type == "toefl"
    assert by["pdf00000001xx"].rotation == 270 and by["pdf00000001xx"].path.exists()


def test_splits_stable_and_by_pdf(data, tmp_path):
    _, pages, clusters, types, _ = data
    out = tmp_path / "split.csv"
    s1 = orient.load_or_make_splits(out, pages, orient.page_types(clusters, types), seed=0)
    assert set(s1.values()) <= {"train", "val", "test"} and len(s1) == 20
    assert sum(v == "train" for v in s1.values()) >= 12
    out.write_text(out.read_text().replace(",val", ",test"))  # se respeta el archivo existente
    s2 = orient.load_or_make_splits(out, pages, {}, seed=99)
    assert "val" not in s2.values()


def test_new_pdfs_get_a_split(data, tmp_path):
    _, pages, clusters, types, _ = data
    out = tmp_path / "split.csv"
    orient.load_or_make_splits(out, pages, {}, seed=0)
    pid = "pdfNEW00000xx"
    (pages / pid).mkdir()
    with (pages / "manifest.csv").open("a", newline="") as f:
        csv.writer(f).writerow([pid, 1, "ok", 0, f"{pid}/001.png"])
    s = orient.load_or_make_splits(out, pages, {}, seed=0)
    assert s[pid] in {"train", "val", "test"}
    assert pid in out.read_text()
