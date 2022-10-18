# Copyright (C) 2022 listenai Co.Ltd
# All rights reserved. 
# Created by leifang on 2022.09.31

import os
import sympy
import onnx
import numpy as np
from typing import AnyStr, Optional

from onnx.numpy_helper import from_array
from onnx.helper import make_tensor_value_info
from onnx.helper import make_node, make_graph, make_model
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from .graph import Graph


def _normalize_attrs(node):
    from enum import Enum

    new_attrs = {}
    for name, attr in node.attrs.items():
        if isinstance(attr, Enum):
            new_attrs[name] = attr.name
        elif name == "to" and node.op_type == "Cast":
            new_attrs[name] = _convert_dtype(node.attrs["to"])
        else:
            if name != "graph":
                new_attrs[name] = attr

    return new_attrs


def _convert_dtype(dtype):
    """Convert thinker tensor dtype to onnx tensor dtype"""
    np_dtype = np.dtype(dtype)
    onnx_dtype = NP_TYPE_TO_TENSOR_TYPE.get(np_dtype)
    if onnx_dtype is None:
        raise TypeError(f"Unsupported dtype: {dtype}!")
    return onnx_dtype


def _convert_shape(shape):
    """Convert thinker tensor shape to onnx tensor shape"""
    ret = list()
    for i in shape:
        if isinstance(i, sympy.Symbol):
            ret.append(i.name)
        elif isinstance(i, int) or isinstance(i, str):
            ret.append(i)
        elif isinstance(i, np.int64):
            ret.append(int(i))
        else:
            raise TypeError(f"Unsupported dim type:{type(i)}!")
    return ret


def _convet_to_onnx_model(
    thinkerGraph: Graph, graph_name: Optional[str] = "", **model_attrs
) -> onnx.ModelProto:
    """Convert thinker graph to onnx model
    Due to all Constant Node has been conveted to GraphEntry.THINKER_TENSOR_CONSTANT,
    Constant Node couldn't been rebuilt in the target onnx model.
    """
    onnx_inputs = list()
    onnx_outputs = list()
    onnx_initializer = list()
    onnx_nodes = list()

    # sort
    thinkerGraph.update()

    # graph input
    for inp in thinkerGraph.inputs:
        entry = inp.tensor
        dtype = _convert_dtype(entry.dtype)
        shape = _convert_shape(entry.shape)
        tensor_vi = make_tensor_value_info(inp.name, dtype, shape)
        onnx_inputs.append(tensor_vi)

    # graph output
    for out in thinkerGraph.outputs:
        onnx_outputs.append(onnx.ValueInfoProto(name=out.name))

    # graph node
    for node in thinkerGraph.nodes.values():
        new_attrs = _normalize_attrs(node)
        for key in new_attrs.keys():
            if isinstance(new_attrs[key], np.dtype):
                new_attrs[key] = _convert_dtype(new_attrs[key])
        onnx_node = make_node(
            op_type=node.op_type,
            inputs=[i.name for i in node.inputs],
            outputs=[i.name for i in node.outputs],
            name=node.name,
            **new_attrs,
        )
        onnx_nodes.append(onnx_node)

    # graph initializer
    for entry in thinkerGraph.entries.values():
        if entry.is_constant():
            tensor = entry.tensor
            if tensor.data is None:
                raise ValueError(f"tensor.data is None!")
            onnx_constant = from_array(tensor.data, name=entry.name)
            onnx_initializer.append(onnx_constant)

    # make graph
    onnx_graph = make_graph(
        onnx_nodes, graph_name, onnx_inputs, onnx_outputs, onnx_initializer
    )
    save_model_attrs = {}
    save_model_attrs.update(model_attrs)
    onnx_model = make_model(
        onnx_graph, producer_name="thinker", producer_version="v1", **save_model_attrs
    )
    return onnx_model


def save_to_onnx_model(
    thinkerGraph: Graph, path: AnyStr, name: Optional[AnyStr] = "", **model_attrs
):
    """save thinker graph to onnx model"""
    graph = thinkerGraph.clone(is_update=False)
    onnx_model = _convet_to_onnx_model(graph, name, **model_attrs)

    # create folder
    str_lists = path.split("/")
    temp = ""
    for i in range(len(str_lists) - 1):
        temp += str_lists[i] + "/"
    if os.path.exists(temp) == False and temp != "":
        os.makedirs(temp)
    try:
        onnx.save(onnx_model, path)
    except ValueError:
        pass


__all__ = ["save_to_onnx_model"]
