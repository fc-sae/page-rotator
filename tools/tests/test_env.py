"""Verifica que el entorno de entrenamiento está completo y ve la GPU."""

import pytest


def test_imports():
    import numpy  # noqa: F401
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401
    import PIL  # noqa: F401
    import pypdfium2  # noqa: F401
    import sklearn  # noqa: F401
    import timm  # noqa: F401
    import torchvision  # noqa: F401


def test_cuda_available():
    import torch

    if not torch.cuda.is_available():
        pytest.fail(f"CUDA no disponible (torch {torch.__version__}, build cuda={torch.version.cuda})")
    x = torch.ones(2, 2, device="cuda")
    assert (x @ x).sum().item() == 8.0
