<div align="center">

# TorchRT

**A lightweight PyTorch >> ONNX >> TensorRT converter with a focus on simplicity**

</div>

TorchRT provides a simple way to convert a PyTorch `nn.Module` to a TensorRT-equipped module. While the goal of this package is similar to that of [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) from NVIDIA, TorchRT does this by first converting the module to ONNX, allowing options to optimize the model graph without the need to maintain conversion code for each operator.

## References
- [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)
- [onnx-simplifier](https://github.com/daquexian/onnx-simplifier)
