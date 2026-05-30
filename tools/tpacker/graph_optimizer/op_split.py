import math
import numpy as np
from typing import List, Dict

from ..xsympy import is_sympy
from ..save_model import save_to_onnx_model
from ..graph_analysis.ops.utils import calc_expr
from ..graph import Graph, GraphEntry, GraphNode, Tensor
from ..enum_defines import MemType, DevType, ALIGN2, ALIGN4, ALIGN8, ALIGN16


def _remove_parameter_reuse(graph: Graph, remove_entry: List):
    """Remove parameter reuse by cloning entries to avoid memory issues."""
    entry_count = dict()
    valid_entry = dict()
    for node in graph.nodes.values():
        if node.op_type in ['LinearInt', 'Conv2dInt']:
            for entry in node.inputs:
                if entry.is_constant and 'weight' in entry.name:
                    entry_count[entry.name] = entry_count.get(entry.name, 0) + 1

    for name, count in entry_count.items():
        if count > 1:
            shape = graph.entries[name].tensor.shape
            size = np.prod(shape)
            if size > 32768:
                valid_entry[name] = count

    for node in graph.nodes.values():
        for i, entry in enumerate(node.inputs):
            if entry.name in valid_entry:
                new_entry = entry.clone()
                new_entry.name = f'@{valid_entry[entry.name]}_{new_entry.name}'
                valid_entry[entry.name] -= 1
                graph.add_entry(new_entry)
                node.inputs[i] = new_entry
                if entry.name not in remove_entry:
                    remove_entry.append(entry.name)
                if len(node.inputs) > 2:
                    bias = node.inputs[2]
                    new_bias = bias.clone()
                    new_bias.name = f'@{valid_entry[entry.name]}_{bias.name}'
                    graph.add_entry(new_bias)
                    node.inputs[2] = new_bias
                    if bias.name not in remove_entry:
                        remove_entry.append(bias.name)

    for name in remove_entry:
        if name in graph.entries:
            del graph.entries[name]

    graph.init_tensor()

def _parameter_reuse_recovery(graph: Graph, remove_entry: List):
    """Recover parameter reuse by restoring original entry names."""
    for node in graph.nodes.values():
        for i, entry in enumerate(node.inputs):
            if entry.name.startswith('@'):
                # print(data.name)
                if entry.name[3:] not in graph.entries:
                    new_entry = entry.clone()
                    new_entry.name = new_entry.name[3:]
                    node.inputs[i] = new_entry
                    graph.add_entry(new_entry)
                else:
                    node.inputs[i] = graph.entries[entry.name[3:]]
                del graph.entries[entry.name]

    graph.init_tensor()

def _sort_nodes(graph: Graph, name_list: List[GraphEntry]) -> bool:
    """Sort nodes to ensure correct execution order."""
    for entry in name_list:
        for next_node in entry.dst_nodes:
            if next_node.op_type in ("Conv2dInt", "ConvTranspose2dInt", "LinearInt", "LayerNormInt", "topN", "topN2", "ArgMax"):
                return False
            else:
                return _sort_nodes(graph, next_node.outputs)
    return True

def _label_nodes(graph: Graph, name_list: List[GraphEntry]):
    """Label nodes to specify memory type."""
    for entry in name_list:
        graph.entries[entry.name].tensor.mem_type = MemType.PSRAM
        for next_node in entry.dst_nodes:
            graph.nodes[next_node.name].dev_type = DevType.HIFI
            _label_nodes(graph, next_node.outputs)

def op_split(ori_graph: Graph, set_out_dev: bool = False, is_dump: bool = False,
            threshold1: int = 65536, threshold2: int = 65536, threshold3: int = 65536) -> Graph:

    linearint_flag = False
    new_graph = Graph.clone(ori_graph, is_update=True)

    reuse_entry = []
    _remove_parameter_reuse(new_graph, reuse_entry)
    graph = Graph.clone(new_graph, is_update=True)

    add_node_list = []
    del_node_list = []
    " search big Conv or group Conv for split "
    platform = graph.platform
    for node in new_graph.nodes.values():
        if node.op_type == "Conv1dInt":
            group = node.attrs["group"]
            stride_w = node.attrs["strides"][0]
            data = node.inputs[0].tensor
            weight = node.inputs[1].tensor
            out = node.outputs[0].tensor

            ch_in = data.shape[1]
            h_in = 1
            w_in = calc_expr(str(data.shape[2]), graph.dynamic_args_max) if is_sympy(data.shape[2]) else data.shape[2]

            kernel_n = weight.shape[0]
            kernel_c = weight.shape[1]
            kernel_h = 1
            kernel_w = weight.shape[2]
            ou_c = out.shape[1]
            ou_h = 1
            ou_w = calc_expr(str(out.shape[2]), graph.dynamic_args_max) if is_sympy(out.shape[2]) else out.shape[2]
            # group conv1d
            if (1 != group) and (group != kernel_n):
                raise AssertionError("Group Conv1dInt not supported!")
            # depthwise conv1d
            elif (1 != group) and (group == kernel_n):
                if platform == "venus":
                    aligned_kernel = ALIGN16(kernel_n) * kernel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * kernel_h * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_limit = 65536
                    kernel_limit = min(32768, threshold1)
                    out_limit = max(65536, threshold2)
                elif platform == "arcs":
                    aligned_kernel = ALIGN8(kernel_n) * kernel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * kernel_h * ((w_in + 4 * stride_w - 1) // (4 * stride_w)) * (4 * stride_w)
                    data_limit = 16384
                    kernel_limit = max(8192, threshold1)
                    out_limit = max(32768, threshold2)
                else:
                    aligned_kernel = ALIGN8(kernel_n) * ALIGN2(kernel_h) * kernel_w
                    split_data_size_align = ALIGN4(ch_in) * kernel_h * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_limit = 32768
                    kernel_limit = max(32768, threshold1)
                    out_limit = max(65536, threshold2)
                split_out_size = ou_c * ou_w * kernel_h

                assert split_data_size_align <= data_limit and aligned_kernel <= kernel_limit, \
                "input size of depthwiseConv1d cannot exceed limit"
            # common conv1d
            else:
                aligned_kernel_without_cout = ALIGN8(ch_in) * kernel_h * kernel_w
                if platform == "venus":
                    aligned_kernel = ALIGN2(kernel_n) * aligned_kernel_without_cout
                    align_input_without_h = ALIGN8(ch_in) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_limit = 65536
                    kernel_limit = min(32768, threshold1)
                    out_limit = max(65536, threshold2)
                elif platform == "arcs":
                    aligned_kernel = ALIGN2(kernel_n) * aligned_kernel_without_cout
                    align_input_without_h = ALIGN8(ch_in) * ((w_in + 4 * stride_w - 1) // (4 * stride_w)) * (4 * stride_w)
                    data_limit = 16384
                    kernel_limit = max(8192, threshold1)
                    out_limit = max(32768, threshold2)
                else:
                    aligned_kernel = ALIGN4(kernel_n) * aligned_kernel_without_cout
                    align_input_without_h = ALIGN8(ch_in) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_limit = 65536
                    kernel_limit = max(32768, threshold1)
                    out_limit = max(65536, threshold2)
                out_without_h = ou_c * ou_w
                assert align_input_without_h <= data_limit, "input size of conv1dInt exceeds the size limit."

                if aligned_kernel <= kernel_limit and out_without_h * ou_h <= out_limit:
                    continue

                out_size_without_cout   = ou_h * ou_w
                kernel_size_without_cout  = ALIGN8(ch_in) * kernel_h * kernel_w
                if platform in {"venus", "arcs"}:
                    channel_out_max = (kernel_limit // kernel_size_without_cout) & 0xFFFFFFFE
                    channel_out_max = min(channel_out_max, (out_limit // out_size_without_cout) & 0xFFFFFFFE)
                else:
                    channel_out_max = (out_limit // kernel_size_without_cout) & 0xFFFFFFFC
                    channel_out_max = min(channel_out_max, (out_limit // out_size_without_cout) & 0xFFFFFFFC)

                split_num   = math.ceil(kernel_n / channel_out_max)

                if platform in {"venus", "arcs"}:
                    channel_out_mean = ALIGN2(ou_c // split_num)
                    while((channel_out_mean + 2) * kernel_size_without_cout <= kernel_limit
                    and ((channel_out_mean + 2) * ou_w <= out_limit)):
                        channel_out_mean += 2
                else:
                    channel_out_mean = ALIGN4(ou_c // split_num)
                    while((channel_out_mean + 4) * kernel_size_without_cout <= kernel_limit
                    and ((channel_out_mean + 4) * ou_w <= out_limit)):
                        channel_out_mean += 4
                channel_out_real = channel_out_mean


                if split_num == 1:
                    continue
                # insert concat node
                new_node = GraphNode("iqCat", node.name + "_concat")
                new_node.inputs = []
                new_node.attrs["axis"] = 1
                new_node.attrs['platform'] = node.attrs['platform']
                new_node.outputs = node.outputs

                # split conv to split_num conv
                for g in range(split_num):
                    conv_split_node = node.clone()
                    conv_split_node.name = node.name + "_{}".format(g)
                    conv_split_node.op  = None

                    for i in range(1, len(node.inputs)):
                        new_entry = node.inputs[i].clone()
                        new_entry.name = "{}::{}_{}".format(node.name, node.inputs[i].name, g)
                        if 1 == i:
                            weight_data = node.inputs[i].data
                            if g == split_num - 1:
                                new_entry.data = weight_data[g * channel_out_real : weight.shape[0]]
                            else:
                                new_entry.data = weight_data[g * channel_out_real : (g + 1) * channel_out_real]
                            new_entry.tensor.shape = tuple(new_entry.data.shape)
                        elif 2 == i:
                            bias_data = node.inputs[i].data
                            if g == split_num - 1:
                                new_entry.data = bias_data[g * channel_out_real : bias_data.shape[0]]
                            else:
                                new_entry.data = bias_data[g * channel_out_real : (g + 1) * channel_out_real]
                            new_entry.tensor.shape = tuple(new_entry.data.shape)
                        conv_split_node.inputs[i] = new_entry
                        graph.add_entry(new_entry)

                    conv_split_out = node.outputs[0].clone()
                    conv_split_out.name = "{}_{}".format(node.outputs[0].name, g)
                    conv_split_node.outputs[0] = conv_split_out
                    graph.add_entry(conv_split_out)
                    add_node_list.append(conv_split_node)
                    new_node.inputs.append(conv_split_out)
                    new_node.attrs["scale_x_{}".format(g)] = node.attrs["scale_o"]

                # remove old nodes and entries from graph
                new_node.attrs["scale_o"] = node.attrs["scale_o"]
                for i in range(1, len(node.inputs)):
                    del graph.entries[node.inputs[i].name]
                for i in range(1, len(node.outputs)):
                    del graph.entries[node.outputs[i].name]
                del_node_list.append(node)
                add_node_list.append(new_node)

        elif node.op_type == "Conv2dInt":      #
            data = node.inputs[0].tensor
            weight = node.inputs[1].tensor
            out = node.outputs[0].tensor

            ch_in = data.shape[1]
            h_in = calc_expr(str(data.shape[2]), graph.dynamic_args_max) if is_sympy(data.shape[2]) else data.shape[2]
            w_in = data.shape[3]

            kernel_n, kernel_c, kernel_h, kernel_w = weight.shape

            ou_c = out.shape[1]
            ou_h = calc_expr(str(out.shape[2]), graph.dynamic_args_max) if is_sympy(out.shape[2]) else out.shape[2]
            ou_w = out.shape[3]

            stride_h, stride_w = node.attrs["strides"]
            group = node.attrs["group"]
            pads = node.attrs["pads"]
            # group convolution
            if 1 != group and group != kernel_n:
                raise AssertionError("Group Conv2dInt not supported!")

            # depthwise convolution
            elif 1 != group and group == kernel_n:
                if platform == "venus":
                    aligned_kernel = ALIGN16(kernel_n) * kernel_h * kernel_w
                    align_input_without_h = ALIGN8(ch_in) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_limit = 65536
                    kernel_limit = min(32768, threshold1)
                    out_limit = max(65536, threshold2)
                elif platform == "arcs":
                    aligned_kernel = ALIGN8(kernel_n) * kernel_h * kernel_w
                    align_input_without_h = ALIGN8(ch_in) * ((w_in + 4 * stride_w - 1) // (4 * stride_w)) * (4 * stride_w)
                    data_limit = 16384
                    kernel_limit = max(8192, threshold1)
                    out_limit = max(32768, threshold2)
                else:
                    aligned_kernel = ALIGN8(kernel_n) * ALIGN2(kernel_h) * kernel_w
                    align_input_without_h = ALIGN4(ch_in) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_limit = 32768
                    kernel_limit = max(32768, threshold1)
                    out_limit = max(65536, threshold2)
                # data_size_align_min         = split_data_size_align * kernel_h

                align_input_without_h = calc_expr(str(align_input_without_h), graph.dynamic_args_max) if is_sympy(align_input_without_h) else align_input_without_h
                assert align_input_without_h * kernel_h <= data_limit, "Splitting into the smallest in_h of depthwiseConv2d must not exceed limit."
                assert aligned_kernel <= kernel_limit, "The aligned kernel_size should not exceed hardware constraints."

            # common convolution
            else:
                if platform == "venus":
                    aligned_kernel = ALIGN2(kernel_n) * ALIGN8(ch_in) * kernel_h * kernel_w
                    align_input_without_h = ALIGN8(ch_in) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_limit = 65536
                    kernel_limit = min(32768, threshold1) # Venus only supports external splitting
                    threshold1 = kernel_limit
                    out_limit = max(65536, threshold2)
                elif platform == "arcs":
                    aligned_kernel = ALIGN2(kernel_n) * ALIGN8(ch_in) * kernel_h * kernel_w
                    align_input_without_h = ALIGN8(ch_in) * ((w_in + 4 * stride_w - 1) // (4 * stride_w)) * (4 * stride_w)
                    data_limit = 16384
                    kernel_limit = max(8192, threshold1)
                    out_limit = max(32768, threshold2)
                else:
                    aligned_kernel = ALIGN4(kernel_n) * ALIGN8(ch_in) * kernel_h * kernel_w
                    align_input_without_h = ALIGN8(ch_in) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_limit = 65536
                    kernel_limit = max(32768, threshold1)
                    out_limit = max(65536, threshold2)
                out_size = ou_c * ou_h * ou_w
                out_size = calc_expr(str(out_size), graph.dynamic_args_max) if is_sympy(out_size) else out_size
                align_input_without_h = calc_expr(str(align_input_without_h), graph.dynamic_args_max) if is_sympy(align_input_without_h) else align_input_without_h
                assert align_input_without_h * kernel_h <= data_limit, "Splitting into the smallest in_h of Conv2d must not exceed limit."
                if aligned_kernel <= kernel_limit and out_size <= out_limit:
                    continue

                split_h_out_max = ou_h
                split_out_size = out_size
                if align_input_without_h * h_in > data_limit:
                    split_h_in_max  = data_limit // align_input_without_h
                    split_h_out_max = max((split_h_in_max - kernel_h - pads[0] + stride_h)  // stride_h, 1)
                    split_out_size = ou_c * ou_w * split_h_out_max

                channel_out_max = ou_c
                if split_out_size > out_limit:
                    if platform in {"venus", "arcs"}:
                        channel_out_max = (out_limit // (ou_w * split_h_out_max)) & 0xFFFFFFFE
                    else:
                        channel_out_max = (out_limit // (ou_w * split_h_out_max)) & 0xFFFFFFFC

                aligned_kernel_without_cout  = ALIGN8(ch_in) * kernel_h * kernel_w
                if aligned_kernel > kernel_limit:
                    if platform in {"venus", "arcs"}:
                        channel_out_max = min((kernel_limit // aligned_kernel_without_cout) & 0xFFFFFFFE, channel_out_max)
                    else:
                        channel_out_max = min((kernel_limit // aligned_kernel_without_cout) & 0xFFFFFFFC, channel_out_max)

                split_num   = math.ceil(kernel_n / channel_out_max)

                if split_num == 1:
                    continue

                if platform in {"venus", "arcs"}:
                    channel_out_mean = ALIGN2(ou_c // split_num)
                    while(channel_out_mean * aligned_kernel_without_cout * 2 <= kernel_limit
                    and (channel_out_mean * ou_w * split_h_out_max * 2 <= out_limit)):
                        channel_out_mean *= 2
                        split_num = ALIGN2(split_num)
                else:
                    channel_out_mean = ALIGN4(ou_c // split_num)
                    while(channel_out_mean * aligned_kernel_without_cout * 2 <= kernel_limit
                    and (channel_out_mean * ou_w * split_h_out_max * 2 <= out_limit)):
                        channel_out_mean *= 2
                        split_num = ALIGN2(split_num)
                channel_out_real = channel_out_mean

                if split_num == 1:
                    continue

                # insert concat node
                new_node = GraphNode("iqCat", node.name + "_concat")
                new_node.inputs = []
                new_node.attrs["axis"] = 1
                new_node.attrs['platform'] = node.attrs['platform']
                new_node.outputs = node.outputs

                # split conv to split_num conv
                for g in range(split_num):
                    conv_split_node = node.clone()
                    conv_split_node.name = node.name + "_{}".format(g)
                    conv_split_node.op = None

                    for i in range(1, len(node.inputs)):
                        new_entry = node.inputs[i].clone()
                        new_entry.name = "{}::{}_{}".format(node.name, node.inputs[i].name, g)
                        if 1 == i:
                            weight_data = node.inputs[i].data
                            if g == split_num - 1:
                                new_entry.data = weight_data[g * channel_out_real : weight.shape[0]]
                            else:
                                new_entry.data = weight_data[g * channel_out_real : (g + 1) * channel_out_real]
                            new_entry.tensor.shape = tuple(new_entry.data.shape)
                        elif 2 == i:
                            bias_data = node.inputs[i].data
                            if g == split_num - 1:
                                new_entry.data = bias_data[g * channel_out_real : bias_data.shape[0]]
                            else:
                                new_entry.data = bias_data[g * channel_out_real : (g + 1) * channel_out_real]
                            new_entry.tensor.shape = tuple(new_entry.data.shape)
                        conv_split_node.inputs[i] = new_entry
                        graph.add_entry(new_entry)

                    conv_split_out = node.outputs[0].clone()
                    conv_split_out.name += "{}_{}".format(node.outputs[0].name, g)

                    conv_split_node.outputs[0] = conv_split_out
                    graph.add_entry(conv_split_out)
                    add_node_list.append(conv_split_node)

                    new_node.inputs.append(conv_split_out)
                    new_node.attrs["scale_x_{}".format(g)] = node.attrs["scale_o"]

                new_node.attrs["scale_o"] = node.attrs["scale_o"]
                # remove old nodes and entries from graph
                for i in range(1, len(node.inputs)):
                    del graph.entries[node.inputs[i].name]
                for i in range(1, len(node.outputs)):
                    del graph.entries[node.outputs[i].name]

                del_node_list.append(node)
                add_node_list.append(new_node)

        elif node.op_type == "ConvTranspose2dInt":
            data = node.inputs[0].tensor
            weight = node.inputs[1].tensor
            out = node.outputs[0].tensor

            ch_in = data.shape[1]
            h_in = calc_expr(str(data.shape[2]), graph.dynamic_args_max) if is_sympy(data.shape[2]) else data.shape[2]
            w_in = data.shape[3]

            kernel_c, kernel_n, kernel_h, kernel_w = weight.shape

            ou_c = out.shape[1]
            ou_h = calc_expr(str(out.shape[2]), graph.dynamic_args_max) if is_sympy(out.shape[2]) else out.shape[2]
            ou_w = out.shape[3]

            stride_h, stride_w = node.attrs["strides"]
            group = node.attrs["group"]
            pads = node.attrs["pads"]

            # group ConvTranspose2dInt
            if 1 != group and group != kernel_n:
                raise AssertionError("Group ConvTranspose2dInt not supported!")

            # depthwise ConvTranspose2dInt
            elif 1 != group and group == kernel_n:
                if platform == "venus":
                    aligned_kernel = ALIGN16(kernel_n) * kernel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * kernel_h * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_limit = 65536
                    kernel_limit = min(32768, threshold1)
                    out_limit = max(65536, threshold1)
                elif platform == "arcs":
                    aligned_kernel = ALIGN8(kernel_n) * kernel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * kernel_h * ((w_in + 4 * stride_w - 1) // (4 * stride_w)) * (4 * stride_w)
                    data_limit = 16384
                    kernel_limit = max(8192, threshold1)
                    out_limit = max(32768, threshold2)
                else:
                    aligned_kernel = ALIGN8(kernel_n) * ALIGN2(kernel_h) * kernel_w
                    split_data_size_align = ALIGN4(ch_in) * kernel_h * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_limit = 32768
                    kernel_limit = max(32768, threshold1)
                    out_limit = max(65536, threshold2)

                assert split_data_size_align <= data_limit and aligned_kernel <= kernel_limit, "min h_in of depthwiseConv2d must not exceed limit"
            # common ConvTranspose2dInt
            else:
                if platform == "venus":
                    aligned_kernel = ALIGN2(kernel_n) * ALIGN8(ch_in) * kernel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_limit = 65536
                    kernel_limit = min(32768, threshold1)
                    out_limit = max(65536, threshold2)
                elif platform == "arcs":
                    aligned_kernel = ALIGN2(kernel_n) * ALIGN8(ch_in) * kernel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * ((w_in + 4 * stride_w - 1) // (4 * stride_w)) * (4 * stride_w)
                    data_limit = 16384
                    kernel_limit = max(8192, threshold1)
                    out_limit = max(32768, threshold2)
                else:
                    aligned_kernel = ALIGN4(kernel_n) * ALIGN8(ch_in) * kernel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_limit = 65536
                    kernel_limit = min(32768, threshold1)
                    out_limit = max(65536, threshold2)
                out_size = ou_c * ou_h * ou_w
                out_size = calc_expr(str(out_size), graph.dynamic_args_max) if is_sympy(out_size) else out_size
                split_data_size_align = calc_expr(str(split_data_size_align), graph.dynamic_args_max) if is_sympy(split_data_size_align) else split_data_size_align
                assert split_data_size_align * kernel_h <= data_limit, "min h_in of ConvTranspose2dInt must not exceed limit!"
                if aligned_kernel <= kernel_limit and out_size <= out_limit:
                    continue

                kernel_size_without_cout  = ALIGN8(ch_in) * kernel_h * kernel_w
                channel_out_max = ou_c
                if aligned_kernel > kernel_limit:
                    if platform in {"venus", "arcs"}:
                        channel_out_max = (kernel_limit // kernel_size_without_cout) & 0xFFFFFFFE
                        # channel_out_max = min(channel_out_max, (out_limit // out_size_without_cout) & 0xFFFFFFFE)
                    else:
                        channel_out_max = (kernel_limit // kernel_size_without_cout) & 0xFFFFFFFC
                        # channel_out_max = min(channel_out_max, (out_limit // out_size_without_cout) & 0xFFFFFFFC)

                split_h_out_max = ou_h
                split_out_size = channel_out_max * ou_w * ou_h
                if split_data_size_align * h_in > data_limit:
                    split_h_in_max  = data_limit // split_data_size_align
                    split_h_out_max = max((split_h_in_max - kernel_h - pads[0] + stride_h)  // stride_h, 1)
                    split_out_size = channel_out_max * ou_w * split_h_out_max

                if split_out_size > out_limit:
                    if platform in {"venus", "arcs"}:
                        channel_out_max = min(channel_out_max, (out_limit // (ou_w *split_h_out_max)) & 0xFFFFFFFE)
                    else:
                        channel_out_max = min(channel_out_max, (out_limit // (ou_w *split_h_out_max)) & 0xFFFFFFFC)

                split_num   = math.ceil(kernel_n / channel_out_max)

                if split_num == 1:
                    continue

                # insert concat node
                new_node = GraphNode("iqCat", node.name + "_concat")
                new_node.inputs = []
                new_node.attrs["axis"] = 1
                new_node.attrs['platform'] = node.attrs['platform']
                new_node.outputs = node.outputs

                # split conv to split_num conv
                for g in range(split_num):
                    conv_split_node = node.clone()
                    conv_split_node.name = node.name + "_{}".format(g)
                    conv_split_node.op = None

                    for i in range(1, len(node.inputs)):
                        new_entry = node.inputs[i].clone()
                        new_entry.name = "{}::{}_{}".format(node.name, node.inputs[i].name, g)
                        if 1 == i:
                            weight_data = node.inputs[i].data
                            if g == split_num - 1:
                                new_entry.data = weight_data[:,g * channel_out_max : weight.shape[1]]
                            else:
                                new_entry.data = weight_data[:,g * channel_out_max : (g + 1) * channel_out_max]
                            new_entry.tensor.shape = tuple(new_entry.data.shape)
                        elif 2 == i:
                            bias_data = node.inputs[i].data
                            if g == split_num - 1:
                                new_entry.data = bias_data[g * channel_out_max : bias_data.shape[0]]
                            else:
                                new_entry.data = bias_data[g * channel_out_max : (g + 1) * channel_out_max]
                            new_entry.tensor.shape = tuple(new_entry.data.shape)
                        conv_split_node.inputs[i] = new_entry
                        graph.add_entry(new_entry)

                    conv_split_out = node.outputs[0].clone()
                    conv_split_out.name = "{}_{}".format(node.outputs[0].name, g)

                    conv_split_node.outputs[0] = conv_split_out
                    graph.add_entry(conv_split_out)
                    add_node_list.append(conv_split_node)

                    new_node.inputs.append(conv_split_out)
                    new_node.attrs["scale_x_{}".format(g)] = node.attrs["scale_o"]

                new_node.attrs["scale_o"] = node.attrs["scale_o"]
                # remove old nodes and entries from graph
                for i in range(1, len(node.inputs)):
                    del graph.entries[node.inputs[i].name]
                for i in range(1, len(node.outputs)):
                    del graph.entries[node.outputs[i].name]
                del_node_list.append(node)
                add_node_list.append(new_node)

        elif node.op_type == "LinearInt":
            data            = node.inputs[0].tensor
            weight          = node.inputs[1].tensor
            M   = 1
            for i in range(len(data.shape)-1):
                M *= data.shape[i]
            L   = weight.shape[0]
            N   = weight.shape[1]
            assert N == data.shape[-1]

            split_num = 1
            split_num_right = 1
            if platform == "venus":
                assert weight.dtype in (np.int8, np.int16, np.int32)
                if weight.dtype == np.int8:
                    left_size_align     = ALIGN4(M) * ALIGN8(N)
                    right_size_align    = ALIGN8(N) * ALIGN4(L)
                    left_size_limit     = 65536
                    right_size_limit    = threshold3
                    int8_condition_r    = right_size_align
                    while int8_condition_r > right_size_limit:
                        split_num       += 1
                        split_L         = ALIGN4(math.ceil(L / split_num))
                        int8_condition_r = ALIGN8(N) * split_L
                elif weight.dtype == np.int16:
                    left_size_align     = ALIGN4(M) * ALIGN2(N)
                    right_size_align    = ALIGN2(N) * ALIGN4(L)
                    left_size_limit     = 32768
                    right_size_limit    = threshold3
                    int8_condition_r    = right_size_align
                    while int8_condition_r > right_size_limit:
                        split_num       += 1
                        split_L         = ALIGN4(math.ceil(L / split_num))
                        int8_condition_r = ALIGN2(N) * split_L
                else:
                    left_size_align     = ALIGN2(M) * ALIGN2(N)
                    right_size_align    = ALIGN2(N) * ALIGN2(L)
                    left_size_limit     = 16384
                    right_size_limit    = 8192
                    int8_condition_r    = right_size_align
                    while int8_condition_r > right_size_limit:
                        split_num       += 1
                        split_L        = ALIGN2(math.ceil(L / split_num))
                        int8_condition_r = ALIGN2(N) * split_L
            elif platform == "arcs":
                assert weight.dtype in (np.int8, np.int32)
                if weight.dtype == np.int8:
                    left_size_align     = ALIGN2(M) * ALIGN8(N)
                    right_size_align    = ALIGN8(N) * ALIGN4(L)
                    left_size_limit     = 8192
                    right_size_limit    = max(16384, threshold3)
                    int8_condition_r    = right_size_align
                    while int8_condition_r > right_size_limit:
                        split_num       += 1
                        split_L         = ALIGN4(math.ceil(L / split_num))
                        int8_condition_r = ALIGN8(N) * split_L
                else:
                    left_size_align    = ALIGN2(M) * N
                    right_size_align   = N * ALIGN2(L)
                    left_size_limit    = 2048
                    right_limit        = max(4096, threshold3)
                    int8_condition_r    = right_size_align
                    while int8_condition_r > right_size_limit:
                        split_num       += 1
                        split_L         = ALIGN2(math.ceil(L / split_num))
                        int8_condition_r = N * split_L
            else:
                assert weight.dtype in (np.int8, np.int16, np.int32)
                if weight.dtype == np.int8:
                    left_size_align     = ALIGN4(M) * ALIGN8(N)
                    right_size_align    = ALIGN8(N) * ALIGN8(L)
                    left_size_limit     = max(65536, threshold3)
                    right_size_limit    = 65536
                elif weight.dtype == np.int16:
                    left_size_align     = ALIGN4(M) * ALIGN8(N)
                    right_size_align    = ALIGN8(N) * ALIGN8(L)
                    left_size_limit     = max(32768, threshold3)
                    right_size_limit    = 32768
                else:
                    left_size_align     = ALIGN4(M) * ALIGN8(N)
                    right_size_align    = ALIGN8(N) * ALIGN8(L)
                    left_size_limit     = max(16384, threshold3)
                    right_size_limit    = 16384

                split_right_output_dim = ("transB" in node.attrs and node.attrs["transB"] == 1)

                int8_condition_left    = left_size_align
                while int8_condition_left > left_size_limit:
                    split_num       += 1
                    split_M         = ALIGN4(math.ceil(M / split_num))
                    int8_condition_left = split_M * N

                split_num_right = 1
                int8_condition_right    = right_size_align
                while int8_condition_right > right_size_limit:
                    split_num_right += 1
                    if split_right_output_dim:
                        split_L = ALIGN8(math.ceil(L / split_num_right))
                        int8_condition_right = ALIGN8(N) * split_L
                    else:
                        split_N = ALIGN8(math.ceil(N / split_num_right))
                        int8_condition_right = split_N * ALIGN8(L)

            if split_num == 1 and split_num_right == 1:
                continue

            if platform in ("venus", "arcs"):
                flag = _sort_nodes(graph, node.outputs)
                topn_flag = 0
                N_in_topN = 0
                if node.outputs[0].dst_nodes != []:
                    next_node = node.outputs[0].dst_nodes[0]
                    if next_node.op_type == "TopN":
                        topn_flag = 1
                        N_in_topN = next_node.attrs['max_num']
                    elif next_node.op_type == "ArgMax":
                        topn_flag = 1
                        N_in_topN = 1
                        platform = next_node.attrs["platform"]

                " insert concat node "
                new_node2 = GraphNode("iqCat", node.name + "_concat")
                new_node2.inputs    = []
                new_node2.attrs["axis"] = -1
                new_node2.attrs['platform'] = node.attrs['platform']
                new_node2.outputs   = []

                " split linearint to split_num linearint "
                for g in range(split_num):
                    linearint_split_node    = node.clone()
                    linearint_split_node.name = node.name + "_{}".format(g)
                    linearint_split_node.op = None

                    weight_shape_split = 0
                    for i in range(1, len(node.inputs)):
                        new_entry = node.inputs[i].clone()
                        new_entry.name += "_{}".format(g)
                        new_entry.set_graph_normal()
                        if 1 == i:
                            weight_data = node.inputs[i].data
                            if g == split_num - 1:
                                new_entry.data = weight_data[g * split_L : L]
                            else:
                                new_entry.data = weight_data[g * split_L : (g + 1) * split_L]
                            new_entry.tensor.shape = tuple(new_entry.data.shape)
                        elif 2 == i:
                            bias_data = node.inputs[i].data
                            if g == split_num - 1:
                                new_entry.data = bias_data[g * split_L : L]
                            else:
                                new_entry.data = bias_data[g * split_L : (g + 1) * split_L]
                            new_entry.tensor.shape = tuple(new_entry.data.shape)

                        linearint_split_node.inputs[i] = new_entry
                        graph.add_entry(new_entry)

                    linearint_split_out         = node.outputs[0].clone()
                    linearint_split_out.name    += "_{}".format(g)

                    if flag & set_out_dev:
                        linearint_split_out.tensor.mem_type = MemType.PSRAM
                    linearint_split_node.outputs[0] = linearint_split_out
                    graph.add_entry(linearint_split_out)
                    add_node_list.append(linearint_split_node)
                    if topn_flag:
                        index_entry         = GraphEntry(node.name + "_{}_offset".format(g))
                        index_entry_data    = np.zeros((1), dtype=np.int64)
                        index_entry_data[0] = g * split_L
                        t = Tensor.from_numpy(index_entry_data)
                        index_entry.tensor  = t
                        index_entry.set_constant()

                        topN_data_entry     = GraphEntry(node.name + "_{}_topdata".format(g))
                        topN_data_entry.set_graph_normal()

                        topN_split_node     = GraphNode("topN", node.name + "_{}_topN".format(g))
                        topN_split_node.inputs.append(linearint_split_out)
                        topN_split_node.inputs.append(index_entry)
                        topN_split_node.attrs["dim"] = -1
                        topN_split_node.attrs["max_num"] = N_in_topN
                        topN_split_node.attrs["platform"] = platform
                        topN_split_node.outputs.append(topN_data_entry)

                        graph.add_entry(index_entry)
                        graph.add_entry(topN_data_entry)
                        add_node_list.append(topN_split_node)

                        data_concat_entry   = GraphEntry(node.name + "_{}_concatdata".format(g))
                        data_concat_entry.set_graph_normal()

                        new_node2.inputs.append(topN_data_entry)
                        new_node2.attrs["scale_x_{}".format(g)] = node.attrs["scale_o"]

                    else:
                        new_node2.inputs.append(linearint_split_out)
                        new_node2.attrs["scale_x_{}".format(g)] = node.attrs["scale_o"]

                new_node2.attrs["scale_o"] = node.attrs["scale_o"]

                if topn_flag:
                    new_node2.outputs.append(data_concat_entry)
                    new_node1 = GraphNode("topN2", node.name + "_{}_topN2".format(g))
                    new_node1.attrs["dim"] = -1
                    new_node1.attrs["max_num"] = N_in_topN
                    new_node1.inputs.append(data_concat_entry)
                    new_node1.attrs["scale_x"] = node.attrs["scale_o"]
                    new_node1.attrs["scale_x"] = node.attrs["scale_o"]
                    new_node1.attrs["platform"] = platform
                    new_node1.outputs = next_node.outputs
                    graph.add_entry(data_concat_entry)
                    new_node2.outputs[0].dst_nodes.append(new_node1)
                    add_node_list.append(new_node1)
                    add_node_list.append(new_node2)

                    for i in range(len(node.outputs)):
                        del graph.entries[node.outputs[i].name]
                    del_node_list.append(node)
                    del_node_list.append(next_node)

                    # add new nodes to graph
                    if flag & set_out_dev:
                        _label_nodes(graph, new_node2.outputs)
                else:
                    new_node2.outputs = node.outputs

                    # remove old nodes and entries from graph
                    for i in range(1, len(node.inputs)):
                        del graph.entries[node.inputs[i].name]
                    for i in range(1, len(node.outputs)):
                        del graph.entries[node.outputs[i].name]

                    # add new nodes to graph
                    if flag & set_out_dev:
                        _label_nodes(graph, new_node2.outputs)
                    # add new nodes to graph
                    add_node_list.append(new_node2)
                    del_node_list.append(node)
            else:
                def _linear_split_sizes(total, parts):
                    split_base = math.ceil(total / parts)
                    sizes = []
                    for idx in range(parts):
                        start = idx * split_base
                        end = min((idx + 1) * split_base, total)
                        if end > start:
                            sizes.append(end - start)
                    return sizes

                # Split input data along channel axis when the left matrix is too large.
                channel_axis = 0 if len(data.shape) == 2 else 1
                channel_dim = data.shape[channel_axis]
                channel_dim = calc_expr(str(channel_dim), graph.dynamic_args_max) if is_sympy(channel_dim) else channel_dim
                channel_dim = int(channel_dim)
                assert split_num <= 8, "LinearInt input channel split_num must not exceed Split attrs capacity"
                assert split_num <= channel_dim, "LinearInt input channel dimension must be no less than split_num"

                channel_split_sizes = _linear_split_sizes(channel_dim, split_num)
                split_num = len(channel_split_sizes)
                channel_entries = []
                if split_num > 1:
                    split_node = GraphNode("Split", node.name + "_split")
                    split_node.inputs = [node.inputs[0]]
                    split_node.attrs["axis"] = channel_axis
                    split_node.attrs["dims"] = split_num
                    split_node.attrs["split"] = channel_split_sizes
                    split_node.outputs = []

                    for g, split_size in enumerate(channel_split_sizes):
                        split_entry = node.inputs[0].clone()
                        split_entry.name += "_{}_{}".format(node.name, g)
                        split_shape = list(split_entry.tensor.shape)
                        split_shape[channel_axis] = split_size
                        split_entry.tensor.shape = tuple(split_shape)
                        split_entry.set_graph_normal()
                        graph.add_entry(split_entry)
                        split_node.outputs.append(split_entry)
                        channel_entries.append((g, split_entry, split_size))
                    add_node_list.append(split_node)
                else:
                    channel_entries.append((0, node.inputs[0], channel_dim))

                # Split weight matrix by columns; transB=1 means columns after weight transpose.
                right_dim = L if split_right_output_dim else N
                # assert split_num_right <= 8, "LinearInt weight column split_num_right must not exceed Split attrs capacity"
                assert split_num_right <= right_dim, "LinearInt weight column dimension must be no less than split_num_right"
                right_split_sizes = _linear_split_sizes(right_dim, split_num_right)
                split_num_right = len(right_split_sizes)

                topn_flag = 0
                next_node = None
                N_in_topN = 0
                topn_platform = node.attrs.get("platform", platform)
                if split_right_output_dim and split_num_right > 1 and len(node.outputs[0].dst_nodes) == 1:
                    candidate = node.outputs[0].dst_nodes[0]
                    output_rank = len(node.outputs[0].tensor.shape)
                    if candidate.op_type in ("TopN", "topN"):
                        topn_axis = candidate.attrs.get("dim", -1)
                        if topn_axis < 0:
                            topn_axis += output_rank
                        if topn_axis == output_rank - 1:
                            topn_flag = 1
                            next_node = candidate
                            N_in_topN = candidate.attrs["max_num"]
                            topn_platform = candidate.attrs.get("platform", topn_platform)
                    elif candidate.op_type == "ArgMax":
                        topn_axis = candidate.attrs.get("axis", -1)
                        if topn_axis < 0:
                            topn_axis += output_rank
                        if topn_axis == output_rank - 1:
                            topn_flag = 1
                            next_node = candidate
                            N_in_topN = 1
                            topn_platform = candidate.attrs.get("platform", topn_platform)

                concat_node = None
                if split_num > 1:
                    concat_node = GraphNode("iqCat", node.name + "_concat")
                    concat_node.inputs = []
                    concat_node.attrs["axis"] = channel_axis
                    concat_node.attrs['platform'] = node.attrs['platform']
                    concat_node.outputs = next_node.outputs if topn_flag else node.outputs

                for g, channel_entry, channel_size in channel_entries:
                    right_entries = []
                    if split_num_right > 1 and not split_right_output_dim:
                        right_split_node = GraphNode("Split", node.name + "_{}_right_split".format(g))
                        right_split_node.inputs = [channel_entry]
                        right_split_node.attrs["axis"] = len(channel_entry.tensor.shape) - 1
                        right_split_node.attrs["dims"] = split_num_right
                        right_split_node.attrs["split"] = right_split_sizes
                        right_split_node.outputs = []

                        for r, right_size in enumerate(right_split_sizes):
                            right_entry = channel_entry.clone()
                            right_entry.name += "_right_{}".format(r)
                            right_shape = list(right_entry.tensor.shape)
                            right_shape[-1] = right_size
                            right_entry.tensor.shape = tuple(right_shape)
                            right_entry.set_graph_normal()
                            graph.add_entry(right_entry)
                            right_split_node.outputs.append(right_entry)
                            right_entries.append(right_entry)
                        add_node_list.append(right_split_node)
                    else:
                        right_entries = [channel_entry for _ in range(split_num_right)]

                    partial_outputs = []
                    col_start = 0
                    for r, right_entry in enumerate(right_entries):
                        col_end = col_start + right_split_sizes[r]
                        weight_entry = node.inputs[1].clone()
                        weight_entry.name += "_{}_{}".format(g, r)
                        if node.inputs[1].data is not None:
                            if split_right_output_dim:
                                weight_data_t = node.inputs[1].data.transpose(1, 0)
                                weight_entry.data = weight_data_t[:, col_start:col_end].transpose(1, 0)
                            else:
                                weight_entry.data = node.inputs[1].data[:, col_start:col_end]
                            weight_entry.tensor.shape = tuple(weight_entry.data.shape)
                        else:
                            weight_shape = list(weight_entry.tensor.shape)
                            if split_right_output_dim:
                                weight_shape[0] = right_split_sizes[r]
                            else:
                                weight_shape[1] = right_split_sizes[r]
                            weight_entry.tensor.shape = tuple(weight_shape)
                        graph.add_entry(weight_entry)

                        linearint_split_node = node.clone()
                        linearint_split_node.name = node.name + "_{}_{}".format(g, r)
                        linearint_split_node.op = None
                        linearint_split_node.inputs = [right_entry, weight_entry]

                        if len(node.inputs) > 2 and (split_right_output_dim or r == 0):
                            bias_entry = node.inputs[2].clone()
                            bias_entry.name += "_{}_{}".format(g, r)
                            if split_right_output_dim:
                                if node.inputs[2].data is not None:
                                    bias_entry.data = node.inputs[2].data[col_start:col_end]
                                    bias_entry.tensor.shape = tuple(bias_entry.data.shape)
                                else:
                                    bias_shape = list(bias_entry.tensor.shape)
                                    bias_shape[0] = right_split_sizes[r]
                                    bias_entry.tensor.shape = tuple(bias_shape)
                            graph.add_entry(bias_entry)
                            linearint_split_node.inputs.append(bias_entry)

                        linearint_split_out = node.outputs[0].clone()
                        linearint_split_out.name += "_{}_{}".format(node.name, "{}_{}".format(g, r))
                        if split_num > 1:
                            split_out_shape = list(linearint_split_out.tensor.shape)
                            split_out_shape[channel_axis] = channel_size
                            linearint_split_out.tensor.shape = tuple(split_out_shape)
                        if split_right_output_dim:
                            split_out_shape = list(linearint_split_out.tensor.shape)
                            split_out_shape[-1] = right_split_sizes[r]
                            linearint_split_out.tensor.shape = tuple(split_out_shape)
                        graph.add_entry(linearint_split_out)

                        linearint_split_node.outputs[0] = linearint_split_out
                        add_node_list.append(linearint_split_node)

                        if topn_flag:
                            index_entry = GraphEntry(node.name + "_{}_{}_offset".format(g, r))
                            index_entry_data = np.zeros((1), dtype=np.int64)
                            index_entry_data[0] = col_start
                            index_entry.tensor = Tensor.from_numpy(index_entry_data)
                            index_entry.set_constant()

                            topN_data_entry = GraphEntry(node.name + "_{}_{}_topdata".format(g, r))
                            topN_data_entry.set_graph_normal()

                            topN_split_node = GraphNode("topN", node.name + "_{}_{}_topN".format(g, r))
                            topN_split_node.inputs.append(linearint_split_out)
                            topN_split_node.inputs.append(index_entry)
                            topN_split_node.attrs["dim"] = -1
                            topN_split_node.attrs["max_num"] = N_in_topN
                            topN_split_node.attrs["platform"] = topn_platform
                            topN_split_node.outputs.append(topN_data_entry)

                            graph.add_entry(index_entry)
                            graph.add_entry(topN_data_entry)
                            add_node_list.append(topN_split_node)
                            partial_outputs.append(topN_data_entry)
                        else:
                            partial_outputs.append(linearint_split_out)
                        col_start = col_end

                    if topn_flag:
                        data_concat_entry = GraphEntry(node.name + "_{}_concatdata".format(g))
                        data_concat_entry.set_graph_normal()

                        topn_concat_node = GraphNode("iqCat", node.name + "_{}_topn_concat".format(g))
                        topn_concat_node.inputs = partial_outputs
                        topn_concat_node.attrs["axis"] = -1
                        topn_concat_node.attrs["platform"] = node.attrs["platform"]
                        for r in range(len(partial_outputs)):
                            topn_concat_node.attrs["scale_x_{}".format(r)] = node.attrs["scale_o"]
                        topn_concat_node.attrs["scale_o"] = node.attrs["scale_o"]
                        topn_concat_node.outputs = [data_concat_entry]
                        graph.add_entry(data_concat_entry)
                        add_node_list.append(topn_concat_node)

                        topn2_node = GraphNode("topN2", node.name + "_{}_topN2".format(g))
                        topn2_node.attrs["dim"] = -1
                        topn2_node.attrs["max_num"] = N_in_topN
                        topn2_node.inputs.append(data_concat_entry)
                        topn2_node.attrs["scale_x"] = node.attrs["scale_o"]
                        topn2_node.attrs["platform"] = topn_platform
                        if concat_node is None:
                            topn2_node.outputs = next_node.outputs
                        else:
                            topn2_output = next_node.outputs[0].clone()
                            topn2_output.name += "_{}_{}".format(node.name, g)
                            topn2_shape = list(topn2_output.tensor.shape)
                            if channel_axis < len(topn2_shape):
                                topn2_shape[channel_axis] = channel_size
                            topn2_output.tensor.shape = tuple(topn2_shape)
                            topn2_output.set_graph_normal()
                            graph.add_entry(topn2_output)
                            topn2_node.outputs = [topn2_output]
                        add_node_list.append(topn2_node)
                        channel_output = topn2_node.outputs[0]
                    elif split_num_right == 1:
                        channel_output = partial_outputs[0]
                    elif split_right_output_dim:
                        right_concat_node = GraphNode("iqCat", node.name + "_{}_right_concat".format(g))
                        right_concat_node.inputs = partial_outputs
                        right_concat_node.attrs["axis"] = -1
                        right_concat_node.attrs["platform"] = node.attrs["platform"]
                        for r in range(len(partial_outputs)):
                            right_concat_node.attrs["scale_x_{}".format(r)] = node.attrs["scale_o"]
                        right_concat_node.attrs["scale_o"] = node.attrs["scale_o"]
                        if split_num == 1:
                            right_concat_node.outputs = node.outputs
                        else:
                            channel_output = node.outputs[0].clone()
                            channel_output.name += "_{}_right_concat_out".format(g)
                            channel_out_shape = list(channel_output.tensor.shape)
                            channel_out_shape[channel_axis] = channel_size
                            channel_output.tensor.shape = tuple(channel_out_shape)
                            graph.add_entry(channel_output)
                            right_concat_node.outputs = [channel_output]
                        add_node_list.append(right_concat_node)
                        channel_output = right_concat_node.outputs[0]
                    else:
                        acc_entry = partial_outputs[0]
                        for r in range(1, len(partial_outputs)):
                            add_node = GraphNode("iqAdd", node.name + "_{}_{}_add".format(g, r))
                            add_node.inputs = [acc_entry, partial_outputs[r]]
                            add_node.attrs["scale_x"] = node.attrs["scale_o"]
                            add_node.attrs["scale_y"] = node.attrs["scale_o"]
                            add_node.attrs["scale_o"] = node.attrs["scale_o"]
                            add_node.attrs["platform"] = node.attrs["platform"]
                            if "quant_mode" in node.attrs:
                                add_node.attrs["quant_mode"] = node.attrs["quant_mode"]
                            if "platform_quant" in node.attrs:
                                add_node.attrs["platform_quant"] = node.attrs["platform_quant"]

                            is_last_add = (r == len(partial_outputs) - 1)
                            if is_last_add and split_num == 1:
                                add_out = node.outputs[0]
                            else:
                                add_out = node.outputs[0].clone()
                                add_out.name += "_{}_{}_sum".format(node.name, "{}_{}".format(g, r))
                                if split_num > 1:
                                    add_out_shape = list(add_out.tensor.shape)
                                    add_out_shape[channel_axis] = channel_size
                                    add_out.tensor.shape = tuple(add_out_shape)
                                graph.add_entry(add_out)
                            add_node.outputs = [add_out]
                            add_node_list.append(add_node)
                            acc_entry = add_out
                        channel_output = acc_entry

                    if concat_node is not None:
                        concat_node.inputs.append(channel_output)
                        concat_node.attrs["scale_x_{}".format(g)] = node.attrs["scale_o"]

                if concat_node is not None:
                    concat_node.attrs["scale_o"] = node.attrs["scale_o"]
                    add_node_list.append(concat_node)

                del_node_list.append(node)
                if topn_flag:
                    del_node_list.append(next_node)
                    if node.outputs[0].name in graph.entries:
                        del graph.entries[node.outputs[0].name]
                linearint_flag = True

    for node in del_node_list:
        del graph.nodes[node.name]

    for node in add_node_list:
        graph.add_node(node)
    # fuse iqCat
    graph.update()

    del_node_list = []
    for node in graph.nodes.values():
        if node.op_type == "iqCat":
            flag = 1
            dim1 = node.attrs["axis"]
            for i in range(len(node.inputs)):
                if None == node.inputs[i].src_node:
                    flag = 0
                    break
                else:
                    pre_node = node.inputs[i].src_node
                    if pre_node.op_type != "iqCat":
                        flag = 0
                        break
                    if len(graph.entries[node.inputs[i]].dst_nodes) > 1:
                        flag = 0
                        break
                    else:
                        dim2 = pre_node.attrs["axis"]
                        if dim1 != dim2:
                            flag = 0
                            break
            if flag:
                new_input = []
                index = 0
                for i in range(len(node.inputs)):
                    pre_node = node.inputs[i].src_node
                    new_input += pre_node.inputs
                    for j in range(len(pre_node.inputs)):
                        scale_name = "scale_x_{}".format(j)
                        new_name = "scale_x_{}".format(index)
                        node.attrs[new_name] = pre_node.attrs[scale_name]
                        node.op.attrs[new_name] = pre_node.attrs[scale_name]
                        index += 1
                    del graph.entries[node.inputs[i].name]
                    del_node_list.append(pre_node)

                node.inputs = new_input

    for node in del_node_list:
        del graph.nodes[node.name]

    graph.update()

    _parameter_reuse_recovery(graph, reuse_entry)

    if is_dump:
        save_to_onnx_model(graph, f"./workspace/{graph.name}/model.ignore/7_graph_op_split.onnx")
    return (graph, linearint_flag)


__all__ = ["op_split"]
