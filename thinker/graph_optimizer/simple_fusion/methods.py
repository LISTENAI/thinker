import numpy as np
from typing import List

from ...graph import Graph, GraphNode, GraphEntry, Tensor, ConstantEntry
from ...enum_defines import MemType
from .sim_fusion import register_method

_op_list = ["Transpose", "Reshape", "Concat", "Gather", "Unsqueeze", "Slice"]
_op_list2 = ["Conv2dInt", "Conv1dInt", "LinearInt", "LSTMInt", "LogSoftmax", "Concat"]


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
            elif next_node.op_type == "Shape":
                continue
            else:
                return False
    return True

def _sort_nodes2(
    graph: Graph, name_list: List[GraphEntry]) -> bool:
    for x in name_list:
        for i in range(len(x.dst_nodes)):
            next_node = x.dst_nodes[i]
            assert len(next_node.outputs) == 1
            if next_node.op_type in _op_list2:
                return False
            elif next_node.outputs[0] in graph.outputs:
                return True
            else:
                return _sort_nodes2(graph, next_node.outputs)

@register_method("CR_fusion")
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
                # else:
                #     node.attrs["act_type"] = 0
    return graph

@register_method("Remove_QuantDequant")
def remove_dquant(graph: Graph) -> Graph:
    for i in range(len(graph.inputs)):
        _quant_list = list()
        flag = _sort_nodes(graph, [graph.inputs[i]], _quant_list)
        if flag:
            next_node = graph.inputs[i].dst_nodes[0]
            while(next_node != None and hasattr(next_node,'op_type') and next_node.op_type != "Quant"):
                next_node = next_node.outputs[0].dst_nodes[0]
            next_node_data_bits = next_node.attrs['data_bits']
            if(next_node_data_bits == 8):
                graph.inputs[i].tensor.dtype = np.dtype("int8")
            elif(next_node_data_bits == 16):
                graph.inputs[i].tensor.dtype = np.dtype("int16")
            elif(next_node_data_bits == 32):
                graph.inputs[i].tensor.dtype = np.dtype("int32")
            else:
                pass
            for j in range(len(_quant_list)):
                node = _quant_list[j]
                assert node.op_type == "Quant"
                scale_o = node.attrs["scale_x"]
                import math
                temp = math.log(scale_o, 2)
                assert abs(temp - int(temp)) < 0.000001
                graph.inputs[i].tensor.scale = int(temp)
                dst_nodes = node.outputs[0].dst_nodes

                node_index=list()
                for next_node in dst_nodes:
                    num_input = len(next_node.inputs)
                    for num in range(num_input):
                        if next_node.inputs[num] == node.outputs[0]:
                            node_index.append(num)

                assert len(node_index) == len(dst_nodes)
                for index, new_node in enumerate(dst_nodes):
                    new_node.inputs[node_index[index]] = node.inputs[0]
                del graph.nodes[node.name]
                del graph.entries[node.outputs[0].name]                            


    del_node_name = list()
    del_entries_name  = list()
    for node in graph.nodes.values():
        if node.op_type == "Dequant":
            if node.outputs[0] in graph.outputs:
                graph.outputs.remove(node.outputs[0])
                node.inputs[0].tensor.dtype = np.int8
                node.inputs[0].set2_graph_output()
                graph.outputs.append(node.inputs[0])
                del_node_name.append(node.name)
                del_entries_name.append(node.outputs[0].name)
            elif _sort_nodes2(graph, [node.outputs[0]]):
                next_node = node.outputs[0].dst_nodes[0]
                next_node.inputs[0] = node.inputs[0]
                del_node_name.append(node.name)
                del_entries_name.append(node.outputs[0].name)

    for i in range(len(del_node_name)):
        del graph.nodes[del_node_name[i]]        

    for i in range(len(del_entries_name)):
        del graph.entries[del_entries_name[i]]

    return graph


@register_method("Transpose2Reshape")
def remove_transpose(graph: Graph) -> Graph:
    add_node_list = list()
    del_node_list = list()
    for node in graph.nodes.values():
        if node.op_type == "Transpose":
            shape = list(node.inputs[0].tensor.shape)
            perm = node.attrs['perm']
            assert len(perm) == len(shape)
            new_shape = list()
            new_shape2 = list()
            for p in perm:
                new_shape.append(shape[p])
                if shape[p] != 1:
                    new_shape2.append(shape[p])
            for i in range(len(shape)):
                if 1 in shape:
                    shape.remove(1)
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

    for node in del_node_list:
        del graph.nodes[node.name]

    for node in add_node_list:
        graph.add_node(node)

    graph.update()

    del_node_list = []
    for node in graph.nodes.values():
        if node.op_type == "Reshape":
            next_nodes = node.outputs[0].dst_nodes
            if len(next_nodes) == 1 and next_nodes[0].op_type == "Reshape":
                del_node_list.append(node)
                index = None
                for i in range(len(next_nodes[0].inputs)):
                    if next_nodes[0].inputs[i].name == node.outputs[0]:
                        index = i
                next_nodes[0].inputs[index] = node.inputs[0]

    for node in del_node_list:
        del graph.nodes[node.name]
    graph.update()
    
    return graph
