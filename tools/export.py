#!/usr/bin/env python3
"""Exporta el modelo entrenado a ONNX para la CLI en Rust (etapa 4).

Entrada  data/models/orientation.pt
Salida   data/models/orientation.onnx   (opset 17, lote dinámico)
         data/models/orientation.json   (contrato de entrada/salida para la CLI)

Verifica que onnxruntime y PyTorch den la misma salida (diferencia máxima < --tol).

    uv run python export.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import orient

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "data" / "models"


def export(ckpt_path: Path, onnx_path: Path, samples: list[np.ndarray] | None = None, tol: float = 1e-4) -> dict:
    import onnx
    import onnxruntime as ort

    model, ckpt = orient.load_checkpoint(ckpt_path, "cpu")
    size = int(ckpt["size"])
    dummy = torch.rand(2, 1, size, size)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (dummy,), str(onnx_path), opset_version=17, dynamo=False,
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    onnx.checker.check_model(onnx.load(str(onnx_path)))

    x = np.concatenate([torch.rand(3, 1, size, size).numpy()] + [s[None] for s in (samples or [])])
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"image": x})[0]
    diff = float(np.abs(ref - got).max())
    if diff > tol:
        raise RuntimeError(f"ONNX difiere de PyTorch: diferencia máxima {diff:.2e} > {tol:.0e}")
    meta = {
        "input": {"name": "image", "shape": ["batch", 1, size, size], "dtype": "float32",
                  "range": "0-1, 1 = blanco",
                  "preprocess": f"gris; lado largo a {size} px (bilineal); centrado en lienzo blanco "
                                f"{size}x{size} con offset (size - lado) // 2"},
        "output": {"name": "logits", "shape": ["batch", 4]},
        "classes": orient.CLASSES,
        "classes_meaning": "grados en sentido horario a aplicar a la página tal como se ve para dejarla derecha",
        "backbone": ckpt["backbone"],
        "epoch": ckpt.get("epoch"),
        "max_abs_diff": diff,
    }
    onnx_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=MODELS / "orientation.pt")
    ap.add_argument("--out", type=Path, default=MODELS / "orientation.onnx")
    ap.add_argument("--pages", type=Path, default=REPO / "data" / "pages",
                    help="se usan algunas páginas reales para la verificación")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args(argv)
    if not args.ckpt.exists():
        print(f"No existe {args.ckpt}: entrena primero con train.py", file=sys.stderr)
        return 1
    samples = []
    if (args.pages / "manifest.csv").exists():
        from PIL import Image
        size = int(torch.load(args.ckpt, map_location="cpu", weights_only=False)["size"])
        for r in orient.read_manifest(args.pages)[:4]:
            with Image.open(args.pages / r["path"]) as img:
                samples.append(orient.to_input(img, size))
    try:
        meta = export(args.ckpt, args.out, samples, args.tol)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"{args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)  diferencia máxima {meta['max_abs_diff']:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
