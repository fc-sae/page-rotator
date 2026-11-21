import csv

import numpy as np

import orient
import predict
from test_export import make_ckpt
from test_orient import arrow_page, data  # noqa: F401  (fixture)
from test_train import EdgeModel


def test_predict_views_are_consistent(tmp_path):
    paths = []
    for r in orient.CLASSES:
        paths.append(tmp_path / f"{r}.png")
        orient.turn(arrow_page(), r).save(paths[-1])
    p, p180 = predict.predict(EdgeModel(), paths, 64, "cpu", batch=3, workers=0)
    assert [orient.CLASSES[i] for i in p.argmax(1)] == orient.CLASSES
    for a, b in zip(p, p180):
        rot, conf, ok = orient.decide(a, b, 0.5)
        assert ok


def test_main_writes_predictions(data, tmp_path):  # noqa: F811
    _, pages, _, _, labels = data
    make_ckpt(tmp_path / "m.pt")
    out = tmp_path / "pred.csv"
    rc = predict.main(["--ckpt", str(tmp_path / "m.pt"), "--pages", str(pages), "--labels", str(labels),
                       "--splits", str(tmp_path / "none.csv"), "--out", str(out), "--workers", "0",
                       "--device", "cpu"])
    assert rc == 0
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 20 and list(rows[0]) == predict.FIELDS
    by = {r["pdf_id"]: r for r in rows}
    assert by["pdf00000000xx"]["label"] == "0" and by["pdf00000002xx"]["label"] == "skip"
    assert by["pdf00000005xx"]["label"] == ""
    assert all(int(r["rotation"]) in orient.CLASSES and 0 <= float(r["confidence"]) <= 1 for r in rows)
