"""Resolve optional resources in source and portable installations."""

import sys
from pathlib import Path


def vocal_model_path() -> Path:
    filename = "UVR_MDXNET_1_9703.onnx"
    if getattr(sys, "frozen", False):
        external = Path(sys.executable).resolve().parent / "models" / filename
        if external.is_file():
            return external
    return Path(__file__).resolve().parent / "models" / filename
