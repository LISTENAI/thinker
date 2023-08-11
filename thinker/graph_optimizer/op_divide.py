from ..graph import Graph,  GraphNode
from ..enum_defines import ALIGN2
from ..save_model import save_to_onnx_model

node_type_list = ["Conv1dInt", "Conv2dInt", "ConvTranspose2dInt"]

def op_divide(
    graph: Graph,
    is_dump: bool = False,
    dump_file_path: str = "./model.ignore/graph_op_divide.onnx",
):
    add_node_list = list()
    del_node_list = list()
    # group conv
    for node in graph.nodes.values():
        if node.op_type in node_type_list:
            weight = node.inputs[1].tensor
            kernel_num = weight.shape[0]
            group = node.attrs["group"]
            channel_split = kernel_num // group
            " group conv "
            if 1 != group and group != kernel_num:
                "insert split node"
                new_node = GraphNode("Split", node.name + "_split")
                new_node.attrs["axis"] = 1
                new_node.attrs["dims"] = group
                new_node.inputs = [node.inputs[0]]
                new_node.outputs = []
                " insert concat node "
                new_node2 = GraphNode("iqCat", node.name + "_concat")
                new_node2.inputs = []
                new_node2.attrs["dim"] = 1
                new_node2.outputs = node.outputs

                " split conv to group conv "
                for g in range(group):
                    new_entry = node.inputs[0].clone()
                    new_entry.name += "_{}".format(g)
                    new_entry.set2_graph_normal()
                    new_node.outputs.append(new_entry)
                    shape = new_entry.tensor.shape
                    assert shape[1] % group == 0, "group conv/deconv must can to be divided by group with no remainder"
                    new_shape = (shape[0], shape[1] // group, shape[2]) if len(shape) <= 3 else (shape[0], shape[1] // group, shape[2], shape[3])
                    new_entry.tensor.shape = new_shape
                    graph.add_entry(new_entry)

                    conv_split_node = node.clone()
                    conv_split_node.name = node.name + "_{}".format(g)
                    conv_split_node.inputs[0] = new_entry
                    conv_split_node.attrs["group"] = 1
                    conv_split_node.op = None
                    
                    for i in range(1, len(node.inputs)):
                        temp_entry = node.inputs[i].clone()
                        temp_entry.name += "_{}".format(g)
                        if 1 == i:
                            weight_data = node.inputs[i].data
                            temp_entry.data = weight_data[g * channel_split : (g + 1) * channel_split]
                            temp_entry.tensor.shape = tuple(temp_entry.data.shape)
                        elif 2 == i:
                            bias_data = node.inputs[i].data
                            temp_entry.data = bias_data[g * channel_split : (g + 1) * channel_split]
                            temp_entry.tensor.shape = tuple(temp_entry.data.shape)

                        conv_split_node.inputs[i] = temp_entry
                        graph.add_entry(temp_entry)

                    conv_split_out = node.outputs[0].clone()
                    conv_split_out.name += "_{}_{}".format(node.name, g)

                    conv_split_node.outputs[0] = conv_split_out
                    graph.add_entry(conv_split_out)
                    add_node_list.append(conv_split_node)

                    new_node2.inputs.append(conv_split_out)
                    new_node2.attrs["scale_x_{}".format(g)] = node.attrs["scale_o"]

                " remove old nodes and entries from graph "
                new_node2.attrs["scale_o"] = node.attrs["scale_o"]
                for i in range(1, len(node.inputs)):
                    del graph.entries[node.inputs[i].name]
                del_node_list.append(node)


                " add new nodes to graph "
                add_node_list.append(new_node)
                add_node_list.append(new_node2)

            elif 1 != group and group == weight.shape[0]:
                "depthwise conv"
                channel_out_align = ALIGN2(weight.shape[0] // group)
                weight_size = channel_out_align * weight.shape[2]
                assert weight_size <= 32768, "kernel size of depthwise conv must <= 32KB"
        
    for node in del_node_list:
        del graph.nodes[node.name]

    for node in add_node_list:
        graph.add_node(node)

    graph.update()
    
    if is_dump:
        save_to_onnx_model(graph, dump_file_path)
    return graph


__all__ = ["op_divide"]
