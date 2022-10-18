# Copyright (C) 2022 listenai Co.Ltd
# All rights reserved. 
# Created by leifang on 2022.09.31

import onnx
import numpy as np
from typing import Dict, List
from onnx import mapping, NodeProto, TensorProto, AttributeProto

from .graph import (
    ConstantEntry,
    EmptyEntry,
    GraphEntry,
    InputEntry,
    OutputEntry,
    Tensor,
    GraphNode,
    Graph,
)


def _parse_attr(ap: AttributeProto) -> Dict:
    """Convert AttributeProto to dict, with names as keys"""
    attrs = {}
    for a in ap:
        for f in ["f", "i", "s"]:
            if a.HasField(f):
                attrs[a.name] = getattr(a, f)
                if isinstance(attrs[a.name], bytes):
                    attrs[a.name] = attrs[a.name].decode(encoding="utf-8")
        for f in ["floats", "ints", "strings"]:
            if list(getattr(a, f)):
                assert a.name not in attrs, "Only one type of attr is allowed"
                attrs[a.name] = tuple(getattr(a, f))
        for f in ["t", "g"]:
            if a.HasField(f):
                attrs[a.name] = getattr(a, f)
        for f in ["tensors", "graphs"]:
            if list(getattr(a, f)):
                raise NotImplementedError("Filed {} is not supported.".format(f))
        if a.name not in attrs:
            raise ValueError("Cannot parse attribute: \n{}\n.".format(a))
    return attrs


def _parse_array(tensor_proto: TensorProto):
    """Grab data in TensorProto and convert to numpy array."""
    try:
        from onnx.numpy_helper import to_array
    except ImportError:
        raise ImportError(
            "Onnx and protobuf need to be installed. "
            + "Instructions to install - https://github.com/onnx/onnx"
        )
    if len(tuple(tensor_proto.dims)) > 0:
        np_array = to_array(tensor_proto).reshape(tuple(tensor_proto.dims))
    else:
        # If onnx's params are scalar values without dims mentioned.
        np_array = np.array(to_array(tensor_proto))
    return np.array(np_array)


def _attribute_to_key_value(attr_proto):
    a = attr_proto
    key = a.name
    value = None
    for f in ["f", "i", "s"]:
        if a.HasField(f):
            value = getattr(a, f)
            # Needed for supporting python version  > 3.5
            if isinstance(value, bytes):
                value = value.decode(encoding="utf-8")
    for f in ["floats", "ints", "strings"]:
        if list(getattr(a, f)):
            value = tuple(getattr(a, f))

    if a.HasField("t"):  # tensor
        value = _parse_array(getattr(a, "t"))

    if a.HasField("g"):  # tensor
        value = getattr(a, "g")

    for f in ["tensors", "graphs"]:
        if list(getattr(a, f)):
            raise NotImplementedError("Filed {} is not supported.".format(f))

    return key, value


def _constant_to_array(node: NodeProto):
    attrs = {}
    for a in node.attribute:
        k, v = _attribute_to_key_value(a)
        attrs[k] = v

    if "value" in attrs:
        return attrs["value"]

    for k in ["value_int", "value_ints", "value_float", "value_floats"]:
        if k in attrs:
            return np.array(attrs[k])

    return None


def _convert_dtype(onnx_type):
    # pylint: disable=no-member
    dtype_map = {
        TensorProto.FLOAT16: np.float16,
        TensorProto.FLOAT: np.float32,
        TensorProto.DOUBLE: np.float64,
        TensorProto.INT8: np.int8,
        TensorProto.INT16: np.int16,
        TensorProto.INT32: np.int32,
        TensorProto.INT64: np.int64,
        TensorProto.UINT8: np.uint8,
        TensorProto.UINT16: np.uint16,
        TensorProto.UINT32: np.uint32,
        TensorProto.UINT64: np.uint64,
        TensorProto.BOOL: np.bool,
    }
    # pylint: enable=no-member
    return dtype_map.get(onnx_type)


def _convert_shape(tensor_shape) -> List:
    import sympy

    shape = []
    for x in tensor_shape.dim:
        if x.HasField("dim_value"):
            shape.append(x.dim_value)
        elif x.HasField("dim_param"):
            shape.append(sympy.Symbol(x.dim_param))
        else:
            raise TypeError
    return shape


def _convert_from_onnx_model(model: onnx.ModelProto) -> Graph:
    onnx_graph = model.graph
    thinker_graph = Graph()
    thinker_graph.onnx_model_desc = {
        "opset_imports": model.opset_import,
        "ir_version": model.ir_version,
    }
    params = {}
    for tensor in onnx_graph.initializer:
        if not tensor.name.strip():
            raise ValueError("Tensor's name is required!")
        params[tensor.name] = Tensor.from_numpy(_parse_array(tensor))

    graph_outputs = [output.name for output in onnx_graph.output]

    for input in onnx_graph.input:
        if input.name not in params:
            dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[input.type.tensor_type.elem_type]
            shape = _convert_shape(input.type.tensor_type.shape)
            tensor = Tensor(shape, dtype)
            node_entry = InputEntry(input.name, tensor=tensor)
            thinker_graph.inputs.append(node_entry.name)
            thinker_graph.add_entry(node_entry)

    for onnx_node in onnx_graph.node:
        for input in onnx_node.input:
            if input in params and input not in thinker_graph.entries:
                node_entry = ConstantEntry(input, tensor=params[input])
                thinker_graph.add_entry(node_entry)

    for onnx_node in onnx_graph.node:
        if onnx_node.op_type == "Constant":
            constant = _constant_to_array(onnx_node)
            node_entry = ConstantEntry(
                onnx_node.output[0], tensor=Tensor.from_numpy(constant)
            )
            thinker_graph.add_entry(node_entry)

    for onnx_node in onnx_graph.node:
        if onnx_node.op_type == "Constant":
            continue

        if (
            onnx_node.HasField("name")
            and onnx_node.name != ""
            and onnx_node.name not in thinker_graph.nodes
        ):
            node_name = onnx_node.name
        else:
            node_name = "%s_%s" % (onnx_node.op_type, onnx_node.output[0])

        thinker_attrs = _parse_attr(onnx_node.attribute)
        if onnx_node.op_type == "Cast":
            thinker_attrs["to"] = _convert_dtype(thinker_attrs["to"])

        thinker_node = GraphNode(onnx_node.op_type, node_name, thinker_attrs)
        for i, input_tensor_name in enumerate(onnx_node.input):
            if input_tensor_name == "":
                entry = EmptyEntry()
                thinker_graph.add_entry(entry)
                input_tensor_name = entry.name

            if input_tensor_name.startswith("Scale"):
                input_tensor_name = input_tensor_name.replace("Scale", "scale")
            thinker_node.inputs.append(input_tensor_name)

        for output_tensor_name in onnx_node.output:
            node_entry = GraphEntry(output_tensor_name)
            if output_tensor_name in graph_outputs:
                node_entry = OutputEntry(output_tensor_name, None)
            thinker_node.outputs.append(output_tensor_name)
            thinker_graph.add_entry(node_entry)

        thinker_graph.add_node(thinker_node)

    thinker_graph.outputs = [x.name for x in onnx_graph.output]
    thinker_graph.update()
    return thinker_graph


def load_onnx_model(model: str) -> Graph:
    onnx_model = onnx.load(model)
    return _convert_from_onnx_model(onnx_model)


__all__ = ["load_onnx_model"]
