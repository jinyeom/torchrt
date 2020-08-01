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
    simplify: bool = True,
    max_batch_size: int = 1,
    max_workspace_size: int = 1 << 25,
    fp16_mode: bool = False,
    force_rebuild: bool = False,
    verbose: bool = False,
) -> TensorRTModule:
    """
    Convert a PyTorch module to a TensorRTModule.

    Parameters
    ----------
    model
        Source PyTorch nn.Module to convert.
    onnx_path
        Destination path to which the ONNX file is exported.
    engine_path
        Destination path to which the TensorRT engine is exported. When `force_rebuild`
        is set to False, the target TensorRT module loads the cached engine, rather than
        building it from a saved ONNX file.
    input_shape
        Shape of the model input, without batch size, e.g., for a typical convolutional
        network, `input_shape` expects (C, H, W).
    opset_version
        Version of ONNX opset.
    simplify
        Flag for simplifying the exported ONNX file.
    max_batch_size
        Maximum batch size for inference.
    max_workspace_size
        Maximum amount of memory allowed for the TensorRT engine.
    fp16_mode
        Flag for FP16 precision.
    force_rebuild
        Flag for rebuilding the TensorRT engine.
    verbose
        Flag for verbose mode.

    Returns
    -------
    A converted TensorRTModule.
    """
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
