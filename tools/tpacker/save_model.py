import os
import sympy
import onnx
import numpy as np
from enum import Enum
from typing import AnyStr, Optional, Dict

from .graph_analysis.ops.utils import calc_expr
from .xsympy import is_sympy
from onnx.numpy_helper import from_array
from onnx.helper import (
    make_tensor_value_info,
    make_node,
    make_graph,
    make_model,
    make_tensor_value_info,
)
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from .graph import Graph

def _convert_dtype(dtype) -> int:
    """
    将Thinker的张量数据类型转换为ONNX的数据类型。
    
    Args:
        dtype: Thinker的张量数据类型
        
    Returns:
        ONNX对应的数据类型
        
    Raises:
        TypeError: 如果不支持该数据类型
    """
    np_dtype = np.dtype(dtype)
    onnx_dtype = NP_TYPE_TO_TENSOR_TYPE.get(np_dtype)
    if onnx_dtype is None:
        raise TypeError(f"Unsupported dtype: {dtype}!")
    return onnx_dtype

def _convert_shape(shape) -> list:
    """
    将Thinker的张量形状转换为ONNX的形状表示。
    
    Args:
        shape: Thinker的张量形状
        
    Returns:
        ONNX对应的形状列表
        
    Raises:
        TypeError: 如果不支持该维度类型
    """
    ret = []
    for dim in shape:
        if isinstance(dim, sympy.Symbol):
            ret.append(dim.name)
        elif isinstance(dim, (int, str)):
            ret.append(dim)
        elif isinstance(dim, np.int64):
            ret.append(int(dim))
        elif is_sympy(dim):
            ret.append(str(dim))
        else:
            raise TypeError(f"Unsupported dim type: {type(dim)}!")
    return ret

def _normalize_attrs(node) -> Dict:
    """
    规范化节点属性，将枚举类型转换为字符串，并处理特定的“to”属性。
    
    Args:
        node: 当前节点
        
    Returns:
        规范化后的属性字典
    """
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

def _convert_to_onnx_model(
    thinkerGraph: Graph,
    graph_name: Optional[str] = "",
    **model_attrs
) -> onnx.ModelProto:
    """
    将Thinker图转换为ONNX模型。
    
    Args:
        thinkerGraph: Thinker图对象
        graph_name: ONNX图的名称，默认为空字符串
        **model_attrs: 其他模型属性
        
    Returns:
        转换后的ONNX模型对象
    """
    onnx_inputs = []
    onnx_outputs = []
    onnx_initializer = []
    onnx_nodes = []

    # 更新图
    thinkerGraph.update()

    # 处理输入
    for inp in thinkerGraph.inputs:
        entry = inp.tensor
        dtype = _convert_dtype(entry.dtype)
        shape = _convert_shape(entry.shape)
        tensor_vi = make_tensor_value_info(inp.name, dtype, shape)
        onnx_inputs.append(tensor_vi)

    # 处理输出
    for out in thinkerGraph.outputs:
        onnx_outputs.append(onnx.ValueInfoProto(name=out.name))

    # 处理节点
    for node in thinkerGraph.nodes.values():
        new_attrs = _normalize_attrs(node)
        for key in new_attrs:
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

    # 处理动态参数
    dynamic_args_max = {}
    if thinkerGraph.dynamic_args_max is not None:
        for k, v in thinkerGraph.dynamic_args_max.items():
            dynamic_args_max[str(k)] = v

    # 处理初始化器
    for entry in thinkerGraph.entries.values():
        if entry.is_constant():
            tensor = entry.tensor
            if tensor.data is None:
                raise ValueError(f"tensor.data is None for entry: {entry.name}")
            if tensor.data.dtype == np.dtype('O'):
                new_data = np.zeros(tensor.shape, dtype=np.int64)
                for i in range(len(tensor.data)):
                    new_data[i] = calc_expr(str(tensor.data[i]), dynamic_args_max)
                tensor.data = new_data
            onnx_constant = from_array(tensor.data, name=entry.name)
            onnx_initializer.append(onnx_constant)

    # 创建ONNX图和模型
    onnx_graph = make_graph(
        onnx_nodes,
        graph_name,
        onnx_inputs,
        onnx_outputs,
        onnx_initializer,
    )
    save_model_attrs = {"producer_name": "thinker", "producer_version": "v1"}
    save_model_attrs.update(model_attrs)
    onnx_model = make_model(onnx_graph, **save_model_attrs)
    return onnx_model

def save_to_onnx_model(
    thinkerGraph: Graph,
    path: AnyStr,
    name: Optional[AnyStr] = "",
    **model_attrs
):
    """
    将自定义IR转换为ONNX模型文件。
    
    Args:
        thinkerGraph: Thinker图对象
        path: 输出文件路径
        name: ONNX图的名称，默认为空字符串
        **model_attrs: 其他模型属性
    """
    graph = thinkerGraph.clone(is_update=False)
    onnx_model = _convert_to_onnx_model(graph, name, **model_attrs)

    # 补充值信息
    for entry in graph.entries.values():
        if entry.is_constant():
            continue
        shape = _convert_shape(entry.tensor.shape)
        value_info = make_tensor_value_info(
            entry.name,
            NP_TYPE_TO_TENSOR_TYPE[entry.tensor.dtype],
            shape,
        )
        onnx_model.graph.value_info.append(value_info)

    # 创建输出目录
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 保存模型
    try:
        onnx.save(onnx_model, path)
    except ValueError:
        pass

__all__ = ["save_to_onnx_model"]