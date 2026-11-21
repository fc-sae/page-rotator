"""Genera edge.onnx: modelo falso para las pruebas de la CLI.

'Ve' la barra oscura de la página sintética: logits = oscuridad de las mitades
[arriba, izquierda, abajo, derecha] → clases [0, 90, 180, 270] (misma convención que el modelo real).
Una página en blanco da logits iguales (confianza 0.25, nunca se toca).

    cd tools && uv run python ../rotator/tests/fixtures/make_edge_model.py
"""
from pathlib import Path

import torch


class EdgeModel(torch.nn.Module):
    def forward(self, x):
        d = 1 - x[:, 0]
        h, w = d.shape[1] // 2, d.shape[2] // 2
        return 50 * torch.stack([d[:, :h].mean((1, 2)), d[:, :, :w].mean((1, 2)),
                                 d[:, h:].mean((1, 2)), d[:, :, w:].mean((1, 2))], 1)


out = Path(__file__).with_name("edge.onnx")
torch.onnx.export(EdgeModel(), (torch.rand(2, 1, 384, 384),), str(out), opset_version=17, dynamo=False,
                  input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
print(out, out.stat().st_size)
