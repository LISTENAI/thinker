import numpy as np
from typing import List

from ...graph import Graph, GraphNode, GraphEntry
from ...enum_defines import MemType
from .sim_fusion import register_method

_op_list = ["Shape", "Transpose", "Reshape", "Concat", "Gather", "Unsqueeze", "Slice"]


def _sort_nodes(
    graph: Graph, name_list: List[GraphEntry], quant_list: List[GraphNode]
) -> bool:
    for x in name_list:
        for i in range(len(x.dst_nodes)):
            next_node = x.dst_nodes[i]
            if next_node.op_type in _op_list:
                return _sort_nodes(graph, next_node.outputs, quant_list)
            elif next_node.op_type == "Quant":
                quant_list.append(next_node)
                continue
            else:
                return False
    return True


@register_method("CR")
def fuse_CR(graph: Graph) -> Graph:
    for node in graph.nodes.values():
        if node.op_type == "Conv2dInt" or node.op_type == "ConvTranspose2dInt":
            if (
                len(node.outputs) == 1
                and len(node.outputs[0].dst_nodes) == 1
                and not node.outputs[0].is_graph_output()
            ):
                next_node = node.outputs[0].dst_nodes[0]
                if next_node.op_type == "Relu":
                    node.attrs["act_type"] = 1
                    del graph.nodes[next_node.name]
                    del graph.entries[node.outputs[0].name]
                    node.outputs = next_node.outputs
                elif next_node.op_type == "Prelu":
                    node.attrs["act_type"] = 2
                    del graph.nodes[next_node.name]
                    del graph.entries[node.outputs[0].name]
                    node.outputs = next_node.outputs
                elif next_node.op_type == "Clip":
                    node.attrs["act_type"] = 3
                    del graph.nodes[next_node.name]
                    del graph.entries[node.outputs[0].name]
                    node.outputs = next_node.outputs
                else:
                    node.attrs["act_type"] = 0
    return graph


@register_method("Remove_QuantDequant")
def remove_dquant(graph: Graph) -> Graph:
    for i in range(len(graph.inputs)):
        _quant_list = list()
        flag = _sort_nodes(graph, [graph.inputs[i]], _quant_list)
        if flag:
            graph.inputs[i].tensor.dtype = np.dtype("int8")
            for j in range(len(_quant_list)):
                node = _quant_list[j]
                assert node.op_type == "Quant"
                scale_o = node.attrs["scale_x"]
                import math
                temp = math.log(scale_o, 2)
                assert abs(temp - int(temp)) < 0.000001
                graph.inputs[i].tensor.scale = int(temp)
                dst_nodes = node.outputs[0].dst_nodes
                for new_node in dst_nodes:
                    new_node.inputs[0] = node.inputs[0]
                del graph.nodes[node.name]
                del graph.entries[node.outputs[0].name]

    for node in graph.nodes.values():
        if node.op_type == "Dequant" and node.outputs[0] in graph.outputs:
            graph.outputs.remove(node.outputs[0])
            node.inputs[0].tensor.dtype = np.int8
            node.inputs[0].set2_graph_output()
            graph.outputs.append(node.inputs[0])
            del graph.nodes[node.name]
            del graph.entries[node.outputs[0].name]

    return graph
