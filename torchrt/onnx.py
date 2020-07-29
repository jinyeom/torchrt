# adopted and modified from https://github.com/daquexian/onnx-simplifier

from copy import deepcopy
from collections import OrderedDict
from typing import List, Dict, Union, Tuple
from pathlib import Path
import onnx
import onnx.helper
import onnx.optimizer
import onnx.shape_inference
import onnx.numpy_helper
import onnxruntime as onnxrt
import numpy as np
import torch
from torch import nn


def _numpy_dtype(elem_type: int) -> int:
    return [
        None,
        np.float32,
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.int32,
        np.int64,
        str,
        np.bool,
        np.float16,
        np.double,
        np.uint32,
        np.uint64,
        np.complex64,
        np.complex128,
        np.float16,
    ][elem_type]


def _forward(model: onnx.ModelProto, nodes: List[onnx.NodeProto]):
    # add outputs of the argument nodes as model outputs.
    model = deepcopy(model)
    for node in nodes:
        for output in node.output:
            value_info = onnx.ValueInfoProto(name=output)
            model.graph.output.append(value_info)

    # create ONNX runtime session
    sess_options = onnxrt.SessionOptions()
    sess_options.graph_optimization_level = onnxrt.GraphOptimizationLevel(0)
    sess_options.log_severity_level = 3
    sess = onnxrt.InferenceSession(
        model.SerializeToString(),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    # get names of input nodes that are not initializers
    input_names = set([v.name for v in model.graph.input])
    init_names = set([v.name for v in model.graph.initializer])
    input_names = input_names - init_names

    # generate random inputs
    inputs = {}
    for v in model.graph.input:
        name = v.name
        shape = (d.dim_value for d in v.type.tensor_type.shape.dim)
        dtype = _numpy_dtype(v.type.tensor_type.elem_type)
        if name in input_names:
            inputs[name] = np.random.rand(*shape).astype(dtype)

    output_names = [x.name for x in sess.get_outputs()]
    run_options = onnxrt.RunOptions()
    run_options.log_severity_level = 3
    outputs = sess.run(output_names, inputs, run_options=run_options)

    return OrderedDict(zip(output_names, outputs))


def _get_constant_nodes(m: onnx.ModelProto) -> List[onnx.NodeProto]:
    const_nodes = []
    const_tensors = [x.name for x in m.graph.initializer]
    for node in m.graph.node:
        if node.op_type == "Constant":
            const_tensors.append(node.output[0])

    tensors_nms = []

    for node in m.graph.node:
        if any(x in tensors_nms for x in node.input):
            tensors_nms.extend(node.output)
        elif node.op_type == "Shape":
            const_nodes.append(node)
            const_tensors.extend(node.output)
        elif node.op_type == "NonMaxSuppression":
            tensors_nms.extend(node.output)
        elif all([x in const_tensors for x in node.input]):
            const_nodes.append(node)
            const_tensors.extend(node.output)
    return deepcopy(const_nodes)


def _insert_node(nodes: List[onnx.NodeProto], index: int, node: onnx.NodeProto):
    nodes.append(nodes[-1])
    for i in reversed(range(index + 1, len(nodes) - 1)):
        nodes[i].CopyFrom(nodes[i - 1])
    nodes[index].CopyFrom(node)


def _del_const_nodes(
    model: onnx.ModelProto,
    const_nodes: List[onnx.NodeProto],
    outputs: Dict[str, np.ndarray],
) -> onnx.ModelProto:
    for i, node in enumerate(model.graph.node):
        if node in const_nodes:
            for output_name in node.output:
                new_node = deepcopy(node)
                new_node.name = "node_" + output_name
                new_node.op_type = "Constant"
                value = outputs[output_name]
                value = onnx.numpy_helper.from_array(value, name=output_name)
                new_attr = onnx.helper.make_attribute("value", value)
                del new_node.input[:]
                del new_node.attribute[:]
                del new_node.output[:]
                new_node.output.append(output_name)
                new_node.attribute.append(new_attr)
                _insert_node(model.graph.node, i + 1, new_node)
            del model.graph.node[i]
    return model


def _optimize(model: onnx.ModelProto) -> onnx.ModelProto:
    input_num = len(model.graph.input)
    onnx.helper.strip_doc_string(model)
    model = onnx.optimizer.optimize(
        model,
        [
            "eliminate_deadend",
            "eliminate_identity",
            "eliminate_nop_dropout",
            "eliminate_nop_monotone_argmax",
            "eliminate_nop_pad",
            "extract_constant_to_initializer",
            "eliminate_unused_initializer",
            "eliminate_nop_transpose",
            "fuse_add_bias_into_conv",
            "fuse_consecutive_log_softmax",
            "fuse_consecutive_reduce_unsqueeze",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
            "fuse_matmul_add_bias_into_gemm",
            "fuse_pad_into_conv",
            "fuse_transpose_into_gemm",
        ],
        fixed_point=True,
    )
    return model


def _simplify(model: onnx.ModelProto, optimize: bool = True):
    model_copy = deepcopy(model)
    model = onnx.shape_inference.infer_shapes(model)

    const_nodes = _get_constant_nodes(model)
    outputs = _forward(model, const_nodes)
    model = _del_const_nodes(model, const_nodes, outputs)
    onnx.checker.check_model(model)

    if optimize:
        model = _optimize(model)
        onnx.checker.check_model(model)

    return model


def export_onnx(
    model: nn.Module,
    path: Union[str, Path],
    input_shape: Tuple[int, ...],
    opset_version: int = 11,
    verbose: bool = False,
    simplify: bool = True,
) -> Path:
    model = model.eval().cpu()
    dummy_input = torch.rand(input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=opset_version,
        verbose=verbose,
        do_constant_folding=True,
        keep_initializers_as_inputs=True,
    )
    if simplify:
        onnx_model = onnx.load(path)
        onnx_model = _simplify(onnx_model)
        onnx.save(onnx_model, path)

    return path
