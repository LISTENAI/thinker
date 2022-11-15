from typing import List

from ..graph import Graph, GraphEntry, GraphNode
from ..enum_defines import MemType, DevType, ALIGN8
from ..save_model import save_to_onnx_model


def _sort_nodes(graph: Graph, name_list: List[GraphEntry]) -> bool:
    for x in name_list:
        for i in range(len(x.dst_nodes)):
            next_node = x.dst_nodes[i]
            if (
                next_node.op_type == "Conv2dInt"
                or next_node.op_type == "ConvTranspose2dInt"
                or next_node.op_type == "LinearInt"
            ):
                return False
            else:
                return _sort_nodes(graph, next_node.outputs)
    return True


def _label_nodes(graph: Graph, name_list: List[GraphEntry]):
    for x in name_list:
        x.tensor.mem_type = MemType.PSRAM
        for i in range(len(x.dst_nodes)):
            next_node = x.dst_nodes[i]
            next_node.dev_type = DevType.HIFI
            _label_nodes(graph, next_node.outputs)


def op_split(
    ori_graph: Graph,
    is_dump: bool = False,
    threshold: int = 65536,
    dump_file_path: str = "./model.ignore/graph_op_split.onnx",
):
    linearint_flag = False
    graph = Graph.clone(ori_graph, is_update=True)
    " search big Conv or group Conv for split "
    for node in ori_graph.nodes.values():
        if node.op_type == "Conv1dInt":
            inputdata = node.inputs[0].tensor
            h = inputdata.shape[2]
            w = 1
            weight = node.inputs[1].tensor
            kernel_num = weight.shape[0]
            kernel_c = weight.shape[1]
            kernel_h = weight.shape[2]
            kernel_w = 1
            output = node.outputs[0].tensor
            attr = node.attrs
            group = attr["group"]
            strides = attr["strides"]
            stride_h = strides[0]
            stride_w = 1
            " group conv "
            if 1 != group and group != weight.shape[0]:
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

                flag = _sort_nodes(graph, node.outputs)

                " split conv to group conv "
                for g in range(group):
                    new_entry = node.inputs[0].clone()
                    new_entry.name += "_in_{}".format(g)
                    new_entry.set2_graph_normal()
                    new_node.outputs.append(new_entry)
                    shape = new_entry.tensor.shape
                    new_shape = (shape[0], shape[1] // group, shape[2])
                    new_entry.tensor.shape = new_shape
                    graph.add_entry(new_entry)

                    conv_split_node = node.clone()
                    conv_split_node.name = node.name + "_{}".format(g)
                    conv_split_node.inputs[0] = new_entry
                    conv_split_node.attrs["group"] = 1
                    conv_split_node.op = None

                    num_input_align = ALIGN8(kernel_c)
                    num_output_align = ((kernel_num // group + 1) // 2) * 2
                    kernel_size = (
                        num_input_align * num_output_align * kernel_h * kernel_w
                    )
                    data_size = (
                        ((w + 8 * stride_w - 1) // (8 * stride_w))
                        * (8 * stride_w)
                        * num_input_align
                        * h
                    )
                    assert (
                        kernel_size * weight.dtype.itemsize <= 32768
                        or data_size * inputdata.dtype.itemsize <= 65536
                    ), "input data size:{} exceed 64KB and weight data size:{} exceed 32KB after group".format(
                        data_size * inputdata.dtype.itemsize,
                        kernel_size * weight.dtype.itemsize,
                    )

                    for i in range(1, len(node.inputs)):
                        new_entry2 = node.inputs[i].clone()
                        new_entry2.name += "_{}".format(g)
                        if 1 == i:
                            weight_data = node.inputs[i].data
                            weight_shape_split = weight.shape[0] // group
                            if g == group - 1:
                                new_entry2.data = weight_data[
                                    g * weight_shape_split : weight.shape[0]
                                ]
                            else:
                                new_entry2.data = weight_data[
                                    g
                                    * weight_shape_split : (g + 1)
                                    * weight_shape_split
                                ]
                            new_entry2.tensor.shape = tuple(new_entry2.data.shape)
                        elif 2 == i:
                            bias_data = node.inputs[i].data
                            bias_shape_split = bias_data.shape[0] // group
                            if g == group - 1:
                                new_entry2.data = bias_data[
                                    g * bias_shape_split : bias_data.shape[0]
                                ]
                            else:
                                new_entry2.data = bias_data[
                                    g * bias_shape_split : (g + 1) * bias_shape_split
                                ]
                            new_entry2.tensor.shape = tuple(new_entry2.data.shape)

                        conv_split_node.inputs[i] = new_entry2
                        graph.add_entry(new_entry2)

                    conv_split_out = node.outputs[0].clone()
                    conv_split_out.name += "_{}".format(g)
                    if flag:
                        conv_split_out.tensor.mem_type = MemType.PSRAM
                    conv_split_node.outputs[0] = conv_split_out
                    graph.add_entry(conv_split_out)
                    graph.add_node(conv_split_node)

                    new_node2.inputs.append(conv_split_out)

                " remove old nodes and entries from graph "

                for i in range(1, len(node.inputs)):
                    del graph.entries[node.inputs[i].name]
                for i in range(1, len(node.outputs)):
                    del graph.entries[node.outputs[i].name]
                del graph.nodes[node.name]

                " add new nodes to graph "
                graph.add_node(new_node)
                if flag:
                    new_node2.dev_type = DevType.HIFI
                    _label_nodes(graph, new_node2.outputs)
                graph.add_node(new_node2)

            elif 1 != group and group == weight.shape[0]:
                "depthwise conv"
                channel_out_align = ((weight.shape[0] // group + 1) // 2) * 2
                weight_size = channel_out_align * weight.shape[2]
                assert weight_size <= 32768
            else:
                "common conv"
                channel_input_align = ((weight.shape[1] + 7) // 8) * 8
                channel_out_align = ((weight.shape[0] + 1) // 2) * 2
                weight_size = channel_input_align * channel_out_align * weight.shape[2]
                data_size = (
                    ((w + 8 * stride_w - 1) // (8 * stride_w))
                    * (8 * stride_w)
                    * channel_input_align
                    * h
                )

                # assert(weight_size * weight.dtype.itemsize <= 32768 or data_size  * inputdata.dtype.itemsize <= 65536), \
                # "input data size:{} exceed 64KB and weight data size:{} exceed 32KB in {}"\
                # .format(data_size*inputdata.dtype.itemsize, weight_size*weight.dtype.itemsize, node.name)

                if weight_size <= 32768:
                    continue

                flag = _sort_nodes(
                    graph, node.outputs
                )  # last conv/convtranspose layer in graph

                channel_out_max = (
                    32768 // (channel_input_align * weight.shape[2])
                ) & 0xFFFFFFFE
                split_num = (weight.shape[0] + channel_out_max - 1) // channel_out_max

                " insert concat node "
                new_node2 = GraphNode("iqCat", node.name + "_concat")
                new_node2.inputs = []
                new_node2.attrs["dim"] = 1
                new_node2.outputs = node.outputs

                " split conv to split_num conv "
                for g in range(split_num):
                    conv_split_node = node.clone()
                    conv_split_node.name = node.name + "_{}".format(g)
                    conv_split_node.op = None

                    for i in range(1, len(node.inputs)):
                        new_entry = node.inputs[i].clone()
                        new_entry.name += "_out_{}".format(g)
                        if 1 == i:
                            weight_data = node.inputs[i].data
                            weight_shape_split = channel_out_max
                            if g == split_num - 1:
                                new_entry.data = weight_data[
                                    g * weight_shape_split : weight.shape[0]
                                ]
                            else:
                                new_entry.data = weight_data[
                                    g
                                    * weight_shape_split : (g + 1)
                                    * weight_shape_split
                                ]
                            new_entry.tensor.shape = tuple(new_entry.data.shape)
                        elif 2 == i:
                            bias_data = node.inputs[i].data
                            bias_shape_split = channel_out_max
                            if g == split_num - 1:
                                new_entry.data = bias_data[
                                    g * bias_shape_split : bias_data.shape[0]
                                ]
                            else:
                                new_entry.data = bias_data[
                                    g * bias_shape_split : (g + 1) * bias_shape_split
                                ]
                            new_entry.tensor.shape = tuple(new_entry.data.shape)
                        conv_split_node.inputs[i] = new_entry
                        graph.add_entry(new_entry)

                    conv_split_out = node.outputs[0].clone()
                    conv_split_out.name += "_{}".format(g)
                    if flag:
                        conv_split_out.tensor.mem_type = MemType.PSRAM
                    conv_split_node.outputs[0] = conv_split_out
                    graph.add_entry(conv_split_out)
                    graph.add_node(conv_split_node)

                    new_node2.inputs.append(conv_split_out)

                " remove old nodes and entries from graph "
                for i in range(1, len(node.inputs)):
                    del graph.entries[node.inputs[i].name]
                for i in range(1, len(node.outputs)):
                    del graph.entries[node.outputs[i].name]
                del graph.nodes[node.name]

                " add new nodes to graph "
                if flag:
                    new_node2.dev_type = DevType.HIFI
                    _label_nodes(graph, new_node2.outputs)
                graph.add_node(new_node2)
        # if node.op_type == "Conv2dInt":
        elif node.op_type == "Conv2dInt":
            inputdata = node.inputs[0].tensor
            h = inputdata.shape[2]
            w = inputdata.shape[3]
            weight = node.inputs[1].tensor
            kernel_num = weight.shape[0]
            kernel_c = weight.shape[1]
            kernel_h = weight.shape[2]
            kernel_w = weight.shape[3]
            output = node.outputs[0].tensor
            attr = node.attrs
            group = attr["group"]
            strides = attr["strides"]
            stride_w = strides[0]
            stride_h = strides[1]
            " group conv "
            if 1 != group and group != weight.shape[0]:
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

                flag = _sort_nodes(graph, node.outputs)

                " split conv to group conv "
                for g in range(group):
                    new_entry = node.inputs[0].clone()
                    new_entry.name += "_in_{}".format(g)
                    new_entry.set2_graph_normal()
                    new_node.outputs.append(new_entry)
                    shape = new_entry.tensor.shape
                    new_shape = (shape[0], shape[1] // group, shape[2], shape[3])
                    new_entry.tensor.shape = new_shape
                    graph.add_entry(new_entry)

                    conv_split_node = node.clone()
                    conv_split_node.name = node.name + "_{}".format(g)
                    conv_split_node.inputs[0] = new_entry
                    conv_split_node.attrs["group"] = 1
                    conv_split_node.op = None

                    num_input_align = ALIGN8(kernel_c)
                    num_output_align = ((kernel_num // group + 1) // 2) * 2
                    kernel_size = (
                        num_input_align * num_output_align * kernel_h * kernel_w
                    )
                    data_size = (
                        ((w + 8 * stride_w - 1) // (8 * stride_w))
                        * (8 * stride_w)
                        * num_input_align
                        * h
                    )
                    assert (
                        kernel_size * weight.dtype.itemsize <= 32768
                        or data_size * inputdata.dtype.itemsize <= 65536
                    ), "input data size:{} exceed 64KB and weight data size:{} exceed 32KB after group".format(
                        data_size * inputdata.dtype.itemsize,
                        kernel_size * weight.dtype.itemsize,
                    )

                    for i in range(1, len(node.inputs)):
                        new_entry2 = node.inputs[i].clone()
                        new_entry2.name += "_{}".format(g)
                        if 1 == i:
                            weight_data = node.inputs[i].data
                            weight_shape_split = weight.shape[0] // group
                            if g == group - 1:
                                new_entry2.data = weight_data[
                                    g * weight_shape_split : weight.shape[0]
                                ]
                            else:
                                new_entry2.data = weight_data[
                                    g
                                    * weight_shape_split : (g + 1)
                                    * weight_shape_split
                                ]
                            new_entry2.tensor.shape = tuple(new_entry2.data.shape)
                        elif 2 == i:
                            bias_data = node.inputs[i].data
                            bias_shape_split = bias_data.shape[0] // group
                            if g == group - 1:
                                new_entry2.data = bias_data[
                                    g * bias_shape_split : bias_data.shape[0]
                                ]
                            else:
                                new_entry2.data = bias_data[
                                    g * bias_shape_split : (g + 1) * bias_shape_split
                                ]
                            new_entry2.tensor.shape = tuple(new_entry2.data.shape)

                        conv_split_node.inputs[i] = new_entry2
                        graph.add_entry(new_entry2)

                    conv_split_out = node.outputs[0].clone()
                    conv_split_out.name += "_{}".format(g)
                    if flag:
                        conv_split_out.tensor.mem_type = MemType.PSRAM
                    conv_split_node.outputs[0] = conv_split_out
                    graph.add_entry(conv_split_out)
                    graph.add_node(conv_split_node)

                    new_node2.inputs.append(conv_split_out)
                    new_node2.attrs["scale_x_{}".format(g)] = node.attrs["scale_o"]

                " remove old nodes and entries from graph "
                new_node2.attrs["scale_o"] = node.attrs["scale_o"]
                for i in range(1, len(node.inputs)):
                    del graph.entries[node.inputs[i].name]
                for i in range(1, len(node.outputs)):
                    del graph.entries[node.outputs[i].name]
                del graph.nodes[node.name]

                " add new nodes to graph "
                graph.add_node(new_node)
                if flag:
                    new_node2.dev_type = DevType.HIFI
                    _label_nodes(graph, new_node2.outputs)
                graph.add_node(new_node2)

            elif 1 != group and group == weight.shape[0]:
                "depthwise conv"
                channel_out_align = ((weight.shape[0] // group + 1) // 2) * 2
                weight_size = channel_out_align * weight.shape[2] * weight.shape[3]
                assert weight_size <= 32768
            else:
                "common conv"
                channel_input_align = ((weight.shape[1] + 7) // 8) * 8
                channel_out_align = ((weight.shape[0] + 1) // 2) * 2
                weight_size = (
                    channel_input_align
                    * channel_out_align
                    * weight.shape[2]
                    * weight.shape[3]
                )
                data_size = (
                    ((w + 8 * stride_w - 1) // (8 * stride_w))
                    * (8 * stride_w)
                    * channel_input_align
                    * h
                )

                # assert(weight_size * weight.dtype.itemsize <= 32768 or data_size  * inputdata.dtype.itemsize <= 65536), \
                # "input data size:{} exceed 64KB and weight data size:{} exceed 32KB in {}"\
                # .format(data_size*inputdata.dtype.itemsize, weight_size*weight.dtype.itemsize, node.name)

                if weight_size <= 32768:
                    continue

                flag = _sort_nodes(
                    graph, node.outputs
                )  # last conv/convtranspose layer in graph

                channel_out_max = (
                    32768 // (channel_input_align * weight.shape[2] * weight.shape[3])
                ) & 0xFFFFFFFE
                split_num = (weight.shape[0] + channel_out_max - 1) // channel_out_max

                " insert concat node "
                new_node2 = GraphNode("iqCat", node.name + "_concat")
                new_node2.inputs = []
                new_node2.attrs["dim"] = 1
                new_node2.outputs = node.outputs

                " split conv to split_num conv "
                for g in range(split_num):
                    conv_split_node = node.clone()
                    conv_split_node.name = node.name + "_{}".format(g)
                    conv_split_node.op = None

                    for i in range(1, len(node.inputs)):
                        new_entry = node.inputs[i].clone()
                        new_entry.name += "_out_{}".format(g)
                        if 1 == i:
                            weight_data = node.inputs[i].data
                            weight_shape_split = channel_out_max
                            if g == split_num - 1:
                                new_entry.data = weight_data[
                                    g * weight_shape_split : weight.shape[0]
                                ]
                            else:
                                new_entry.data = weight_data[
                                    g
                                    * weight_shape_split : (g + 1)
                                    * weight_shape_split
                                ]
                            new_entry.tensor.shape = tuple(new_entry.data.shape)
                        elif 2 == i:
                            bias_data = node.inputs[i].data
                            bias_shape_split = channel_out_max
                            if g == split_num - 1:
                                new_entry.data = bias_data[
                                    g * bias_shape_split : bias_data.shape[0]
                                ]
                            else:
                                new_entry.data = bias_data[
                                    g * bias_shape_split : (g + 1) * bias_shape_split
                                ]
                            new_entry.tensor.shape = tuple(new_entry.data.shape)
                        conv_split_node.inputs[i] = new_entry
                        graph.add_entry(new_entry)

                    conv_split_out = node.outputs[0].clone()
                    conv_split_out.name += "_{}".format(g)
                    if flag:
                        conv_split_out.tensor.mem_type = MemType.PSRAM
                    conv_split_node.outputs[0] = conv_split_out
                    graph.add_entry(conv_split_out)
                    graph.add_node(conv_split_node)

                    new_node2.inputs.append(conv_split_out)
                    new_node2.attrs["scale_x_{}".format(g)] = node.attrs["scale_o"]

                new_node2.attrs["scale_o"] = node.attrs["scale_o"]
                " remove old nodes and entries from graph "
                for i in range(1, len(node.inputs)):
                    del graph.entries[node.inputs[i].name]
                for i in range(1, len(node.outputs)):
                    del graph.entries[node.outputs[i].name]
                del graph.nodes[node.name]

                " add new nodes to graph "
                if flag:
                    new_node2.dev_type = DevType.HIFI
                    _label_nodes(graph, new_node2.outputs)
                graph.add_node(new_node2)

        elif node.op_type == "ConvTranspose2dInt":
            weight = node.inputs[1].tensor
            attr = node.attrs
            group = attr["group"]
            if 1 != group:
                flag = _sort_nodes(graph, node.outputs)
                " insert split node "
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

                " split deconv to group deconv "
                for g in range(group):
                    new_entry = node.inputs[0].clone()
                    new_entry.name += "_{}".format(g)
                    new_entry.set2_graph_normal()
                    new_node.outputs.append(new_entry)
                    shape = new_entry.tensor.shape
                    new_shape = (shape[0], shape[1] // group, shape[2], shape[3])
                    new_entry.tensor.shape = new_shape
                    graph.add_entry(new_entry)

                    conv_split_node = node.clone()
                    conv_split_node.name = node.name + "_{}".format(g)
                    conv_split_node.inputs[0] = new_entry
                    conv_split_node.attrs["group"] = 1
                    conv_split_node.op = None

                    for i in range(1, len(node.inputs)):
                        new_entry2 = node.inputs[i].clone()
                        new_entry2.name += "_{}".format(g)
                        if 1 == i:
                            weight_data = node.inputs[i].data
                            weight_shape_split = weight.shape[0] // group
                            if g == group - 1:
                                new_entry2.data = weight_data[
                                    g * weight_shape_split : weight.shape[0]
                                ]
                            else:
                                new_entry2.data = weight_data[
                                    g
                                    * weight_shape_split : (g + 1)
                                    * weight_shape_split
                                ]
                            new_entry2.tensor.shape = tuple(new_entry2.data.shape)
                        elif 2 == i:
                            bias_data = node.inputs[i].data
                            bias_shape_split = bias_data.shape[0] // group
                            if g == group - 1:
                                new_entry2.data = bias_data[
                                    g * bias_shape_split : bias_data.shape[0]
                                ]
                            else:
                                new_entry2.data = bias_data[
                                    g * bias_shape_split : (g + 1) * bias_shape_split
                                ]
                            new_entry2.tensor.shape = tuple(new_entry2.data.shape)
                        conv_split_node.inputs[i] = new_entry2
                        graph.add_entry(new_entry2)

                    conv_split_out = node.outputs[0].clone()
                    conv_split_out.name += "_{}".format(g)
                    if flag:
                        conv_split_out.mem_type = MemType.PSRAM
                    conv_split_node.outputs[0] = conv_split_out
                    graph.add_entry(conv_split_out)
                    graph.add_node(conv_split_node)

                    new_node2.inputs.append(conv_split_out)
                    new_node2.attrs["scale_x_{}".format(g)] = node.attrs["scale_o"]

                new_node2.attrs["scale_o"] = node.attrs["scale_o"]
                " remove old nodes and entries from graph "
                for i in range(1, len(node.inputs)):
                    del graph.entries[node.inputs[i].name]
                for i in range(1, len(node.outputs)):
                    del graph.entries[node.outputs[i].name]
                del graph.nodes[node.name]

                " add new nodes to graph "
                graph.add_node(new_node)
                if flag:
                    new_node2.dev_type = DevType.HIFI
                    _label_nodes(graph, new_node2.outputs)
                graph.add_node(new_node2)

            else:
                "common deconv"
                channel_input_align = ((weight.shape[0] + 7) // 8) * 8
                channel_out_align = ((weight.shape[1] + 1) // 2) * 2
                weight_size = (
                    channel_input_align
                    * channel_out_align
                    * weight.shape[2]
                    * weight.shape[3]
                )
                assert weight_size <= 32768, "size of weight：{} in ConvTranpose exceed limit 32KB\n".format(weight_size)
                continue

        elif node.op_type == "LinearInt" or node.op_type == "BmmInt":
            weight = node.inputs[1].tensor
            transB = node.attrs["transB"]
            weight_size = weight.nbytes
            right_limit = threshold

            if weight_size <= right_limit:
                if weight_size > 32 * 1024:
                    linearint_flag = True
                continue
            linearint_flag = True
            flag = _sort_nodes(graph, node.outputs)

            split_num = 1
            if transB:
                L = weight.shape[0]
                N = weight.shape[1]
            else:
                N = weight.shape[0]
                L = weight.shape[1]

            split_L = L / split_num

            int8_condition_r = (((N + 7) // 8) * 8) * (((split_L + 3) // 4) * 4)
            while int8_condition_r > right_limit or (0 != (L % split_num)):
                split_num += 1
                split_L = L / split_num
                int8_condition_r = (((N + 7) // 8) * 8) * (((split_L + 3) // 4) * 4)

            " insert concat node "
            new_node2 = GraphNode("iqCat", node.name + "_concat")
            new_node2.inputs = []
            new_node2.attrs["dim"] = -1
            new_node2.outputs = node.outputs

            " split linearint to split_num linearint "
            for g in range(split_num):
                linearint_split_node = node.clone()
                linearint_split_node.name = node.name + "_{}".format(g)
                linearint_split_node.op = None

                for i in range(1, len(node.inputs)):
                    new_entry = node.inputs[i].clone()
                    new_entry.name += "_{}".format(g)
                    new_entry.set2_graph_normal()
                    if 1 == i:
                        weight_data = node.inputs[i].data
                        if transB:
                            weight_shape_split = weight.shape[0] // split_num
                            new_entry.data = weight_data[
                                g * weight_shape_split : (g + 1) * weight_shape_split
                            ]
                        else:
                            weight_shape_split = weight.shape[1] // split_num
                            new_entry.data = weight_data[
                                :, g * weight_shape_split : (g + 1) * weight_shape_split
                            ]
                        new_entry.tensor.shape = tuple(new_entry.data.shape)
                    elif 2 == i:
                        bias_data = node.inputs[i].data
                        bias_shape_split = bias_data.shape[0] // split_num
                        new_entry.data = bias_data[
                            g * bias_shape_split : (g + 1) * bias_shape_split
                        ]
                        new_entry.tensor.shape = tuple(new_entry.data.shape)

                    linearint_split_node.inputs[i] = new_entry
                    graph.add_entry(new_entry)

                linearint_split_out = node.outputs[0].clone()
                linearint_split_out.name += "_{}".format(g)
                if flag:
                    linearint_split_out.tensor.mem_type = MemType.PSRAM
                linearint_split_node.outputs[0] = linearint_split_out
                graph.add_entry(linearint_split_out)
                graph.add_node(linearint_split_node)

                new_node2.inputs.append(linearint_split_out)
                new_node2.attrs["scale_x_{}".format(g)] = node.attrs["scale_o"]

            new_node2.attrs["scale_o"] = node.attrs["scale_o"]
            " remove old nodes and entries from graph "
            for i in range(1, len(node.inputs)):
                del graph.entries[node.inputs[i].name]
            for i in range(1, len(node.outputs)):
                del graph.entries[node.outputs[i].name]
            del graph.nodes[node.name]

            " add new nodes to graph "
            if flag:
                new_node2.dev_type = DevType.HIFI
                _label_nodes(graph, new_node2.outputs)
            graph.add_node(new_node2)
    graph.update()
    for node in graph.nodes.values():
        if node.op_type == "iqCat":
            flag = 1
            for i in range(len(node.inputs)):
                pre_node = node.inputs[i].src_node
                if pre_node.op_type != "iqCat":
                    flag = 0
                    break
            if flag:
                new_input = list()
                for i in range(len(node.inputs)):
                    pre_node = node.inputs[i].src_node
                    new_input += pre_node.inputs
                    del graph.nodes[pre_node.name]
                    del graph.entries[node.inputs[i].name]

                node.inputs = new_input
    graph.update()

    if is_dump:
        save_to_onnx_model(graph, dump_file_path)
    return (graph, linearint_flag)


__all__ = ["op_split"]
