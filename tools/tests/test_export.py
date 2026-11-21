import json

import numpy as np
import onnxruntime as ort
import torch

import export
import orient


def make_ckpt(path, size=64):
    torch.manual_seed(0)
    m = orient.build_model("mobilenetv3_small_050", pretrained=False).eval()
    torch.save({"state_dict": m.state_dict(), "backbone": "mobilenetv3_small_050", "size": size,
                "classes": orient.CLASSES, "epoch": 3}, path)


def test_export_matches_pytorch(tmp_path):
    make_ckpt(tmp_path / "m.pt")
    out = tmp_path / "m.onnx"
    meta = export.export(tmp_path / "m.pt", out, [np.ones((1, 64, 64), np.float32)])
    assert meta["max_abs_diff"] < 1e-4 and meta["classes"] == [0, 90, 180, 270]
    assert json.loads(out.with_suffix(".json").read_text())["input"]["shape"] == ["batch", 1, 64, 64]
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    y = sess.run(None, {"image": np.zeros((5, 1, 64, 64), np.float32)})[0]
    assert y.shape == (5, 4)  # lote dinámico


def test_main_missing_ckpt(tmp_path, capsys):
    assert export.main(["--ckpt", str(tmp_path / "no.pt")]) == 1
    assert "train.py" in capsys.readouterr().err
