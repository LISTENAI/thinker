import numpy as np
from typing import List, Optional, Tuple

from ....xsympy import is_sympy
from ..method_register import register_method
from ....graph import Graph, GraphNode, GraphEntry, Tensor

_op_list = ["Transpose", "Reshape", "Gather", "Unsqueeze", "Slice"]
_op_list2 = ["Conv2dInt", "Conv1dInt", "LinearInt", "LSTMInt", "LogSoftmax", "Concat"]

def _sort_nodes(graph: Graph, name_list: List[GraphEntry], quant_list: List[GraphNode]) -> int:
    for x in name_list:
        if x in graph.outputs:
            return False
        for i in range(len(x.dst_nodes)):
            next_node = x.dst_nodes[i]
            if next_node.op_type in _op_list:
                return _sort_nodes(graph, next_node.outputs, quant_list)
            elif next_node.op_type == "Concat":
                return 2
            elif next_node.op_type == "Quant":
                quant_list.append(next_node)
                continue
            elif next_node.op_type == "Shape":
                continue
            else:
                return -1
    return 1

def _sort_nodes_for_concat(graph: Graph, name_list: List[GraphEntry]) -> int:
    for x in name_list:
        if x in graph.outputs:
            return False
        for i in range(len(x.dst_nodes)):
            next_node = x.dst_nodes[i]
            if next_node.op_type in _op_list:
                return _sort_nodes_for_concat(graph, next_node.outputs)
            elif next_node.op_type == "Concat":
                return next_node

def _sort_nodes_up(graph: Graph, name_list: List[GraphEntry]) -> bool:
    flag = list()
    for x in name_list:
        if x in graph.inputs:
            flag.append(True)
            continue
        elif x.is_constant():
            if x.tensor.dtype in (np.int8, np.int16, np.int32, np.int64):
                flag.append(True)
            else:
                flag.append(False)
            continue
        pre_node = x.src_node
        if pre_node.op_type in _op_list:
            flag.append(_sort_nodes_up(graph, pre_node.inputs))
        elif pre_node.op_type == "Dequant":
            flag.append(True)
        else:
            flag.append(False)
    if flag:
        return True
    else:
        return False

def _replace_up_node(graph: Graph, name_list: List[GraphEntry], del_node_list: List[GraphNode], del_entries_list: List[GraphEntry]):
    for x in name_list:
        if x in graph.inputs:
            if x.tensor.dtype not in (np.int8, np.int16, np.int32, np.int64):
                x.tensor.dtype = np.int8
                continue
        elif x.is_constant():
            assert x.tensor.dtype in (np.int8, np.int16, np.int32, np.int64)
            continue
        pre_node = x.src_node
        if pre_node.op_type in _op_list:
            _replace_up_node(graph, pre_node.inputs, del_node_list, del_entries_list)
        elif pre_node.op_type == "Dequant":
            node = x.dst_nodes[0]
            index=None
            for idx in range(len(node.inputs)):
                if node.inputs[idx] == pre_node.outputs[0]:
                    index = idx
                    break
            node.inputs[index] = pre_node.inputs[0]
            for idx in range(len(node.inputs)):
                node.attrs['scale_x_{}'.format(idx)] = pre_node.attrs['scale_o']
            del_node_list.append(pre_node)
            del_entries_list.append(pre_node.outputs[0])
        else:
            AssertionError

def _sort_nodes_down(graph: Graph, name_list: List[GraphEntry]) -> int:
    flag = list()
    for x in name_list:
        if x in graph.outputs:
            flag.append( True)
            continue
        elif x.is_constant():
            if x.tensor.dtype in (np.int8, np.int16, np.int32, np.int64):
                flag.append(True)
            else:
                flag.append(False)
            continue
        for i in range(len(x.dst_nodes)):
            next_node = x.dst_nodes[i]
            if next_node.op_type in _op_list:
                flag.append(_sort_nodes_down(graph, next_node.outputs))
            elif next_node.op_type == "Quant":
                flag.append(True)
            else:
                flag.append(False)
    if flag:
        return True
    else:
        return False

def _replace_down_node(graph: Graph, name_list: List[GraphEntry], del_node_list: List[GraphNode], del_entries_list: List[GraphEntry]):
    for x in name_list:
        if x in graph.outputs:
            continue
        elif x.is_constant():
            assert x.tensor.dtype in (np.int8, np.int16, np.int32, np.int64)

        for i in range(len(x.dst_nodes)):
            next_node = x.dst_nodes[i]
            if next_node.op_type in _op_list:
                _replace_down_node(graph, next_node.outputs, del_node_list, del_entries_list)
            elif next_node.op_type == "Quant":
                next_next_node = next_node.outputs[0].dst_nodes[0]
                index = None
                for idx in range(len(next_next_node.inputs)):
                    if next_next_node.inputs[idx] == next_node.outputs[0]:
                        index = idx
                        break
                next_next_node.inputs[index] = x
                x.dst_nodes[0] = next_next_node
                del_node_list.append(next_node)
                del_entries_list.append(next_node.outputs[0])
            else:
                AssertionError

def _search_for_scale(graph: Graph, name_list: List[GraphEntry])->int:
    for x in name_list:
        if x in graph.outputs:
            continue
        elif x.is_constant():
            assert x.tensor.dtype in (np.int8, np.int16, np.int32, np.int64)

        for i in range(len(x.dst_nodes)):
            next_node = x.dst_nodes[i]
            # if next_node.op_type in _op_list:
            #     return _search_for_scale(graph, next_node.outputs)
            if next_node.op_type == "Quant":
                return next_node.attrs['scale_x']
            elif next_node.op_type == "Shape":
                continue
            else:
                assert next_node.op_type in _op_list
                return _search_for_scale(graph, next_node.outputs)

def _sort_nodes2(graph: Graph, name_list: List[GraphEntry]) -> bool:
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

@register_method("Remove_QuantDequant")
def remove_dquant(graph: Graph) -> Graph:
    add_node_list = list()
    del_node_list = list()
    del_entries_list = list()
    for i, entry in enumerate(graph.inputs):
        _quant_list = list()
        flag = _sort_nodes(graph, [entry], _quant_list)
        if flag==1:
            next_node = entry.dst_nodes[0]
            while(next_node != None and hasattr(next_node,'op_type') and next_node.op_type != "Quant"):
                next_node = next_node.outputs[0].dst_nodes[0]
            next_node_data_bits = next_node.attrs['data_bits']
            if(next_node_data_bits == 8):
                entry.tensor.dtype = np.dtype("int8")
                entry.tensor.bits  = 1
            elif(next_node_data_bits == 16):
                entry.tensor.dtype = np.dtype("int16")
                entry.tensor.bits  = 2
            elif(next_node_data_bits == 32):
                entry.tensor.dtype = np.dtype("int32")
                entry.tensor.bits  = 4
            else:
                pass
            for j in range(len(_quant_list)):
                node = _quant_list[j]
                assert node.op_type == "Quant"
                scale_o = node.attrs["scale_x"]
                import math
                temp = math.log(scale_o, 2)
                assert abs(temp - int(temp)) < 0.000001
                entry.tensor.scale = int(temp)
                dst_nodes = node.outputs[0].dst_nodes

                node_index=list()
                for next_node in dst_nodes:
                    num_input = len(next_node.inputs)
                    for index in range(num_input):
                        if next_node.inputs[index] == node.outputs[0]:
                            node_index.append(index)

                assert len(node_index) == len(dst_nodes)
                for index, new_node in enumerate(dst_nodes):
                    new_node.inputs[node_index[index]] = node.inputs[0]  
                del_node_list.append(node)
                del_entries_list.append(node.outputs[0])
        elif flag == 2:
                node = _sort_nodes_for_concat(graph, entry.dst_nodes[0].outputs)
                concat_flag = _sort_nodes_up(graph, node.inputs)
                concat_flag = concat_flag and _sort_nodes_down(graph, node.outputs)
                if concat_flag:
                    new_node = GraphNode("iqCat", node.name+'_new')
                    new_node.inputs = node.inputs
                    new_node.outputs = node.outputs
                    del_node_list.append(node)
                    _replace_up_node(graph, node.inputs, del_node_list, del_entries_list)
                    new_node.attrs = node.attrs
                    new_node.attrs['scale_o'] = _search_for_scale(graph, node.outputs)
                    _replace_down_node(graph, node.outputs, del_node_list, del_entries_list)
                    add_node_list.append(new_node)

    for i in range(len(del_node_list)):
        del graph.nodes[del_node_list[i]]  
    for i in range(len(del_entries_list)):
        del graph.entries[del_entries_list[i]]

    for i in range(len(add_node_list)):
        graph.add_node(add_node_list[i])

    graph.update()

    del_node_name = list()
    del_entries_name  = list()
    for node in graph.nodes.values():
        if node.op_type == "Dequant":
            if node.outputs[0] in graph.outputs:
                graph.outputs.remove(node.outputs[0])
                node.inputs[0].tensor.dtype = np.int8
                node.inputs[0].tensor.bits = 1
                node.inputs[0].set_graph_output()
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

    graph.update()

    return graph
