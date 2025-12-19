# Copyright (C) 2025 listenai Co.Ltd
# All rights reserved.
# Created by leifang on 2022.09.31

import onnx
import struct
import numpy as np
from typing import Dict, List, Tuple
from onnx import mapping, NodeProto, TensorProto, AttributeProto

from .enum_defines import Colors, ModelConfig
from .save_model import save_to_onnx_model
from .graph import (ConstantEntry, EmptyEntry, GraphEntry, InputEntry, OutputEntry, Tensor, GraphNode, Graph)

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
                raise NotImplementedError(f"Filed {f} is not supported.")
        if a.name not in attrs:
            raise ValueError(f"Cannot parse attribute: \n{a}\n.")
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
        np_array = np.array(to_array(tensor_proto))
    return np.array(np_array)

def _attribute_to_key_value(attr_proto):
    a = attr_proto
    key = a.name
    value = None
    for f in ["f", "i", "s"]:
        if a.HasField(f):
            value = getattr(a, f)
            if isinstance(value, bytes):
                value = value.decode(encoding="utf-8")
    for f in ["floats", "ints", "strings"]:
        if list(getattr(a, f)):
            value = tuple(getattr(a, f))
    if a.HasField("t"):
        value = _parse_array(getattr(a, "t"))
    if a.HasField("g"):
        value = getattr(a, "g")
    for f in ["tensors", "graphs"]:
        if list(getattr(a, f)):
            raise NotImplementedError(f"Filed {f} is not supported.")
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

def _convert_attr(attrs: Dict) -> Dict:
    new_attrs = {}
    for k, v in attrs.items():
        if k == "x_bits":
            new_attrs["data_bits"] = v
        elif k == "w_bits":
            new_attrs["parameter_bits"] = v
        elif k == "stride":
            new_attrs["strides"] = v
        elif k== 'scale_i':
            new_attrs['scale_x'] = v
        else:
            new_attrs[k] = v
    return new_attrs

def _convert_op_type(op_type: str) -> str:
    map_dict = {
        "QBatchNorm2d": "BatchNorm2dInt",
        "QLayerNorm2d": "LayerNormInt",
        "QLinear": "LinearInt",
        "QConv1d": "Conv1dInt",
        "QConv2d": "Conv2dInt",
        "QConvBN2d": "Conv2dInt",
        "QConvTranspose2d": "ConvTranspose2dInt",
        "QAvgPool2d": "AvgPool2dInt",
        "QMaxPool2d": "MaxPool2dInt",
        "QGRU": "GRUInt",
        "QLSTM": "LstmInt",
        "QBmm": "BmmInt",
        "QSigmoid": "iqSigmoid",
        "QTanh": "iqTanh",
        "QCat": "iqCat",
        "QAdd": "iqAdd",
        "QMul": "iqMul",
        "QSoftmax": "SoftmaxInt",
        "QLSTM":"LSTMInt",
        "QGRU":"GRUInt"
    }
    return map_dict.get(op_type, op_type)

def _convert_from_onnx_model(graph_path: str, model_config: ModelConfig, is_dump: bool=False) -> Graph:
    model = onnx.load(graph_path)
    print(f"{Colors.GREEN}2.1 load onnx graph:{graph_path} passed{Colors.RESET}")

    onnx_graph = model.graph
    thinker_graph = Graph()
    thinker_graph.onnx_model_desc = {
        "opset_imports": model.opset_import,
        "ir_version": model.ir_version,
    }
    thinker_graph.name = graph_path.split("/")[-1].split(".")[0]

    params = {}
    for tensor in onnx_graph.initializer:
        if not tensor.name.strip():
            raise ValueError("Tensor's name is required!")
        params[tensor.name] = Tensor.from_numpy(_parse_array(tensor))

    graph_outputs = [output.name for output in onnx_graph.output]
    for graph_input in onnx_graph.input:
        if graph_input.name not in params:
            dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[graph_input.type.tensor_type.elem_type]
            shape = _convert_shape(graph_input.type.tensor_type.shape)
            tensor = Tensor(shape, dtype)
            node_entry = InputEntry(graph_input.name, tensor=tensor)
            thinker_graph.inputs.append(node_entry)
            thinker_graph.add_entry(node_entry)

    for onnx_node in onnx_graph.node:
        for node_input in onnx_node.input:
            if node_input in params and node_input not in thinker_graph.entries:
                node_entry = ConstantEntry(node_input, tensor=params[node_input])
                thinker_graph.add_entry(node_entry)

    for onnx_node in onnx_graph.node:
        if onnx_node.op_type == "Constant":
            constant = _constant_to_array(onnx_node)
            node_entry = ConstantEntry(onnx_node.output[0], tensor=Tensor.from_numpy(constant))
            thinker_graph.add_entry(node_entry)
    for onnx_node in onnx_graph.node:
        if onnx_node.op_type == "Constant":
            continue

        if (onnx_node.HasField("name") and onnx_node.name != "" \
            and onnx_node.name not in thinker_graph.nodes):
            node_name = onnx_node.name
        else:
            node_name = f"{onnx_node.op_type}_{onnx_node.output[0]}"

        onnx_attrs = _parse_attr(onnx_node.attribute)
        thinker_attrs = _convert_attr(onnx_attrs)

        if 'platform' in thinker_attrs:
            if thinker_graph.platform != None:
                assert thinker_graph.platform == thinker_attrs['platform']
            else:
                thinker_graph.platform = thinker_attrs['platform']

        if onnx_node.op_type == "Cast":
            thinker_attrs["to"] = _convert_dtype(thinker_attrs["to"])

        thinker_op_type = _convert_op_type(onnx_node.op_type)

        thinker_node = GraphNode(thinker_op_type, node_name, thinker_attrs)
        for input_tensor_name in onnx_node.input:
            if input_tensor_name == "":
                entry = EmptyEntry()
                thinker_graph.add_entry(entry)
                input_tensor_name = entry.name
                thinker_node.inputs.append(entry)
            elif input_tensor_name.startswith("Scale"):
                input_tensor_name = input_tensor_name.replace("Scale", "scale")
                thinker_node.inputs.append(thinker_graph.entries[input_tensor_name])
            else:
                thinker_node.inputs.append(thinker_graph.entries[input_tensor_name])

        for output_tensor_name in onnx_node.output:
            if output_tensor_name in graph_outputs:
                node_entry = OutputEntry(output_tensor_name, None)
            else:
                node_entry = GraphEntry(output_tensor_name)
            thinker_node.outputs.append(node_entry)
            thinker_graph.add_entry(node_entry)

        thinker_graph.add_node(thinker_node)

    if thinker_graph.platform == None:
        thinker_graph.platform  = "venus"

    thinker_graph.outputs = [thinker_graph.entries[x.name] for x in onnx_graph.output]
    thinker_graph.update()
    thinker_graph._apply_dynamic_axes(model_config.dynamic_shape)
    thinker_graph.init_tensor()

    for node in thinker_graph.nodes.values():
        if "platform" in node.attrs:
            assert node.attrs["platform"] == thinker_graph.platform
        else:
            node.attrs["platform"] = thinker_graph.platform
    print(f"{Colors.GREEN}2.2 convert to custom IR passed{Colors.RESET}")

    if model_config.inputs:
        thinker_graph.inputs = []
        for _, name in enumerate(model_config.inputs):
            assert (name in thinker_graph.entries), f"entry:{name} do not in original graph"
            thinker_graph.entries[name].set_graph_input()
            thinker_graph.inputs.append(thinker_graph.entries[name])

    if model_config.outputs:
        thinker_graph.outputs = []
        for _, name in enumerate(model_config.outputs):
            assert (name in thinker_graph.entries), f"entry:{name} do not in original graph"
            thinker_graph.entries[name].set_graph_output()
            thinker_graph.outputs.append(thinker_graph.entries[name])

    if model_config.inputs or model_config.outputs:
        thinker_graph.update()
        thinker_graph._apply_dynamic_axes(model_config.dynamic_shape)
        thinker_graph.init_tensor()
        print(f"{Colors.GREEN}2.3 subgraph partition passed{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}2.3 subgraph partition skipped{Colors.RESET}")

    if is_dump:
        save_to_onnx_model(thinker_graph, f"./workspace/{thinker_graph.name}/model.ignore/1_graph_constant_fold.onnx")
        print(f"{Colors.GREEN}2.4 Intermediate computation graph exported successfully{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}2.4 Intermediate computation graph exported skipped{Colors.RESET}")
    return thinker_graph

def load_and_convert_onnx_model(model: str, model_config: ModelConfig, is_dump: bool=False) -> Graph:
    return _convert_from_onnx_model(model, model_config, is_dump)

__all__ = ["load_and_convert_onnx_model"]