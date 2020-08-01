from pathlib import Path
from typing import Union

from torch import nn

from .onnx import export_onnx
from .tensorrt import TensorRTModule


def convert(
    model: nn.Module,
    onnx_path: Union[str, Path],
    engine_path: Union[str, Path],
    input_shape: Tuple[int, ...],
    opset_version: int = 11,
    verbose: bool = False,
    simplify: bool = True,
    max_batch_size: int = 1,
    max_workspace_size: int = 1 << 25,
    fp16_mode: bool = False,
    force_rebuild: bool = False,
) -> TensorRTModule:
    input_shape = (max_batch_size, *input_shape)
    onnx_path = export_onnx(
        model,
        onnx_path,
        input_shape,
        opset_version=opset_version,
        verbose=verbose,
        simplify=simplify,
    )
    rt_module = TensorRTModule(
        onnx_path,
        engine_path,
        max_batch_size=max_batch_size,
        max_workspace_size=max_workspace_size,
        fp16_mode=fp16_mode,
        force_rebuild=force_rebuild,
    )
    return rt_module
