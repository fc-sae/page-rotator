import json

import numpy as np
import pytest
import torch
from PIL import Image

import orient
import train
from test_orient import arrow_page, data  # noqa: F401  (fixture)


class EdgeModel(torch.nn.Module):
    """Modelo falso que 've' la barra oscura: logits = oscuridad de [arriba, izquierda, abajo, derecha]."""

    def forward(self, x):
        d = 1 - x[:, 0]
        h, w = d.shape[1] // 2, d.shape[2] // 2
        return 50 * torch.stack([d[:, :h].mean((1, 2)), d[:, :, :w].mean((1, 2)),
                                 d[:, h:].mean((1, 2)), d[:, :, w:].mean((1, 2))], 1)


def make_items(tmp_path, rotations, types=None):
    items = []
    for i, r in enumerate(rotations):
        p = tmp_path / f"{i}.png"
        orient.turn(arrow_page(), r).save(p)  # guardada tal como "llegó": necesita corrección r
        items.append(orient.Item(f"pdf{i:08d}xx", 1, p, r, (types or ["a"] * len(rotations))[i]))
    return items


def test_train_set_returns_rotated_upright(tmp_path):
    items = make_items(tmp_path, [90])
    ds = train.TrainSet(items, size=64, augment=False)
    for _ in range(8):
        x, k = ds[0]
        assert x.shape == (1, 64, 64)
        logits = EdgeModel()(x[None])
        assert int(logits.argmax()) == k  # la clase coincide con lo que se ve


def test_eval_and_metrics_perfect_model(tmp_path):
    items = make_items(tmp_path, [0, 90, 270], types=["a", "a", "b"])
    probs = train.predict_views(EdgeModel(), items, 64, "cpu", batch=4, workers=0)
    assert probs.shape == (3, 4, 4)
    m = train.metrics(items, probs, threshold=0.5)
    assert m["accuracy"] == 1.0
    assert m["per_type"]["b"]["views"] == 4
    assert m["rule"]["all_views"]["correct"] == 1.0
    assert m["rule"]["real"]["n"] == 3


def test_metrics_counts_wrong_and_missed():
    items = [orient.Item("x" * 12, 1, None, 0, "a")]
    probs = np.full((1, 4, 4), 0.01, dtype=np.float32)
    for k in range(4):
        probs[0, k, (k + 1) % 4] = 0.97  # siempre se equivoca por 90°, pero de forma consistente
    m = train.metrics(items, probs, 0.9)
    assert m["accuracy"] == 0.0
    # el giro 270 se predice como 0 → no se toca (sin corregir); los otros 3 se giran mal
    assert m["rule"]["all_views"]["wrong"] == 0.75 and m["rule"]["all_views"]["missed"] == 0.25
    probs[0, :, :] = 0.25  # sin confianza: nunca toca la página
    m = train.metrics(items, probs, 0.9)
    assert m["rule"]["all_views"]["missed"] == 0.75 and m["rule"]["all_views"]["wrong"] == 0.0


def test_sample_weights_inverse_sqrt():
    items = [orient.Item("x", 1, None, 0, t) for t in ["a"] * 4 + ["b"]]
    w = train.sample_weights(items)
    assert w[0] == pytest.approx(0.5) and w[-1] == pytest.approx(1.0)


def test_main_trains_and_writes_outputs(data, tmp_path):  # noqa: F811
    root, pages, clusters, types, labels = data
    # todas las páginas etiquetadas para tener train y val
    with labels.open("a") as f:
        for i in range(3, 20):
            f.write(f"pdf{i:08d}xx,1,{[0, 90, 180, 270][i % 4]},ok,t\n")
    out = tmp_path / "models"
    rc = train.main(["--labels", str(labels), "--pages", str(pages), "--clusters", str(clusters),
                     "--types", str(types), "--splits", str(tmp_path / "split.csv"), "--out", str(out),
                     "--backbone", "mobilenetv3_small_050", "--no-pretrained", "--size", "64",
                     "--epochs", "1", "--batch", "4", "--workers", "0", "--device", "cpu", "--test"])
    assert rc == 0
    m = json.loads((out / "metrics.json").read_text())
    assert m["epoch"] == 1 and "test" in m
    model, ckpt = orient.load_checkpoint(out / "orientation.pt")
    assert ckpt["size"] == 64 and model(torch.rand(1, 1, 64, 64)).shape == (1, 4)


def test_main_without_labels(tmp_path, capsys):
    assert train.main(["--labels", str(tmp_path / "nada.csv")]) == 1
    assert "label.py" in capsys.readouterr().err
