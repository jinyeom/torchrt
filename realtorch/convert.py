from torch import nn

from .onnx import export_onnx
from .tensorrt import TensorRTModule


def convert(model: nn.Module) -> TensorRTModule:
    pass