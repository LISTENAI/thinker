import numpy as np
from typing import List, Optional, Tuple

from ....xsympy import is_sympy
from ..method_register import register_method
from ....graph import Graph, GraphNode, GraphEntry, Tensor


@register_method("Remove_Slice")
def remove_slice(graph: Graph) -> Graph:
    """
    移除Slice节点。
    
    该函数遍历图中的每个节点，如果节点是Slice类型，
    并且满足特定条件（如起始位置为0，步长为1，且切片范围不超过数据维度），
    则替换Slice节点的输出为原始输入，并删除Slice节点。
    
    Args:
        graph: 当前图对象
        
    Returns:
        优化后的图对象
    """
    del_node_list = []
    
    for node in list(graph.nodes.values()):
        if node.op_type == "Slice":
            data = node.inputs[0]
            starts = node.inputs[1].tensor.data[0]
            ends = node.inputs[2].tensor.data[0]
            ends = -1 if ends == 0x7fffffffffffffff else ends
            axes = node.inputs[3].tensor.data[0]
            steps = node.inputs[4].tensor.data[0] if len(node.inputs) == 5 else 1
            dist = ends - starts
            
            if not (is_sympy(dist) or is_sympy(data.tensor.shape[axes])):
                if data.tensor.shape[axes] <= dist and starts == 0 and steps == 1:
                    output_entry = node.outputs[0]
                    
                    if output_entry in graph.outputs:
                        # 替换输出
                        index = graph.outputs.index(output_entry)
                        graph.outputs[index] = node.inputs[0]
                    else:
                        # 替换后续节点输入
                        for next_node in output_entry.dst_nodes:
                            if next_node.inputs:
                                next_node.inputs[0] = node.inputs[0]
                        
                        # 删除条目
                        if output_entry.name in graph.entries:
                            del graph.entries[output_entry.name]
                        
                    del_node_list.append(node)
    
    # 删除节点
    for node in del_node_list:
        if node.name in graph.nodes:
            del graph.nodes[node.name]
    
    graph.update()
    return graph