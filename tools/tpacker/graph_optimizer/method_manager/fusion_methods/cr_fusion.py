import numpy as np
from typing import List, Optional, Tuple

from ....xsympy import is_sympy
from ..method_register import register_method
from ....graph import Graph, GraphNode, GraphEntry, Tensor


@register_method("CR_fusion")
def fuse_CR(graph: Graph) -> Graph:
    """
    融合卷积（Conv）和激活函数（Activation）节点。
    
    该函数遍历图中的每个节点，如果节点是卷积类型且其输出只有一个节点，
    并且该节点是激活函数类型，则将激活函数的类型信息添加到卷积节点的属性中，
    并删除中间的激活函数节点。
    
    Args:
        graph: 当前图对象
        
    Returns:
        优化后的图对象
    """
    del_node_list = []

    activation_map = {
        "Relu": 1,
        "Prelu": 2,
        "Clip": 3
    }

    for node in list(graph.nodes.values()):
        if node.op_type in {"conv1dInt", "Conv2dInt", "ConvTranspose2dInt"}:
            if not node.outputs[0].is_graph_output() and len(node.outputs[0].dst_nodes) == 1:
                next_node = node.outputs[0].dst_nodes[0]
                
                if next_node.op_type in activation_map:
                    act_type = activation_map[next_node.op_type]
                    node.attrs["act_type"] = act_type
                    
                    # 删除中间条目
                    if node.outputs[0].name in graph.entries:
                        del graph.entries[node.outputs[0].name]
                    
                    # 更新输出
                    node.outputs = next_node.outputs
                    del_node_list.append(next_node)
    
    # 删除节点
    for node in del_node_list:
        if node.name in graph.nodes:
            del graph.nodes[node.name]
    
    graph.update()
    return graph