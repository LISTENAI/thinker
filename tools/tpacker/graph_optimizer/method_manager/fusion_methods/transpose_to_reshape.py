import numpy as np
from typing import List, Optional, Tuple

from ....xsympy import is_sympy
from ..method_register import register_method
from ....graph import Graph, GraphNode, GraphEntry, Tensor, ConstantEntry

@register_method("Transpose_to_Reshape")
def transpose_to_reshape(graph: Graph) -> Graph:
    """
    将Transpose节点转换为Reshape节点，并合并连续的Reshape节点。
    
    该函数遍历图中的Transpose节点，将其转换为Reshape节点，
    并删除原始的Transpose节点。最后，合并连续的Reshape节点以简化图结构。
    
    Args:
        graph: 当前图对象
        
    Returns:
        优化后的图对象
    """
    add_node_list = []
    del_node_list = []
    
    # 将Transpose节点转换为Reshape节点
    _convert_transpose_to_reshape(graph, add_node_list, del_node_list)
    
    # 删除Transpose节点
    _cleanup_nodes(graph, del_node_list)
    
    # 添加Reshape节点
    _add_new_nodes(graph, add_node_list)
    
    # 合并连续的Reshape节点
    _merge_consecutive_reshape(graph)
    
    return graph

def _convert_transpose_to_reshape(
    graph: Graph,
    add_node_list: List[GraphNode],
    del_node_list: List[GraphNode]
):
    """
    将Transpose节点转换为Reshape节点。
    
    Args:
        graph: 当前图对象
        add_node_list: 新增的Reshape节点列表
        del_node_list: 删除的Transpose节点列表
    """
    for node in graph.nodes.values():
        if node.op_type == "Transpose":
            input_shape = node.inputs[0].tensor.shape
            shape = []
            for _, p in enumerate(input_shape):
                shape.append(p) if p == 1 else shape.append(str(_))
            perm = node.attrs['perm']
            assert len(perm) == len(shape)
            new_shape = []
            new_shape2 = []
            for p in perm:
                if is_sympy(input_shape[p]):
                    new_shape.append(-1)
                else:
                    new_shape.append(input_shape[p])
                if shape[p] != 1:
                    new_shape2.append(shape[p])
            shape = [item for item in shape if item != 1]
            if shape == new_shape2:
                new_node = GraphNode("Reshape", node.name)
                reshape_data = np.array(new_shape, dtype=np.int64)
                reshape_tensor = Tensor.from_numpy(reshape_data)
                new_entry =  ConstantEntry(node.name, reshape_tensor)
                new_node.inputs = [node.inputs[0], new_entry]
                new_node.outputs = node.outputs
                del_node_list.append(node)
                add_node_list.append(new_node)
                graph.add_entry(new_entry)

def _cleanup_nodes(graph: Graph, del_node_list: List[GraphNode]):
    """
    删除指定的节点。
    
    Args:
        graph: 当前图对象
        del_node_list: 删除的节点列表
    """
    for node in del_node_list:
        if node.name in graph.nodes:
            del graph.nodes[node.name]

def _add_new_nodes(graph: Graph, add_node_list: List[GraphNode]):
    """
    添加新的节点到图中。
    
    Args:
        graph: 当前图对象
        add_node_list: 新增的节点列表
    """
    for node in add_node_list:
        graph.add_node(node)
    graph.update()

def _merge_consecutive_reshape(graph: Graph):
    """
    合并连续的Reshape节点。
    
    Args:
        graph: 当前图对象
    """
    del_node_list = []
    for node in list(graph.nodes.values()):
        if node.op_type == "Reshape":
            next_nodes = node.outputs[0].dst_nodes
            if len(next_nodes) == 1 and next_nodes[0].op_type == "Reshape":
                del_node_list.append(node)
                index = next((i for i, inp in enumerate(next_nodes[0].inputs) if inp.name == node.outputs[0].name), None)
                if index is not None:
                    next_nodes[0].inputs[index] = node.inputs[0]
    
    # 删除重复的Reshape节点
    for node in del_node_list:
        if node.name in graph.nodes:
            del graph.nodes[node.name]
    
    graph.update()