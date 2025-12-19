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
            h_in = calc_expr(str(data.shape[2]), graph.dynamic_args_max) if is_sympy(data.shape[2]) else data.shape[2]
            w_in = data.shape[3] if len(data.shape)== 4 else 1

            kernel_n = weight.shape[0]
            kernel_c = weight.shape[1]
            kernel_h = weight.shape[2]
            kernel_w = weight.shape[3] if len(weight.shape) == 4 else 1

            ou_c = out.shape[1]
            ou_h = calc_expr(str(out.shape[2]), graph.dynamic_args_max) if is_sympy(out.shape[2]) else out.shape[2]
            ou_w = out.shape[3] if len(out.shape) == 4 else 1
            # group conv1d
            if (1 != group) and (group != kernel_n):
                raise AssertionError("Group Conv1dInt not supported!")
            # depthwise conv1d
            elif (1 != group) and (group == kernel_n):
                if platform == "venus":
                    aligned_kernel = ALIGN16(kernel_n) * kernel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * kernel_h * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_threshold = 65536
                    kernel_limit = max(32768, threshold1)
                    out_limit = max(65536, threshold2)
                elif platform == "arcs":
                    aligned_kernel = ALIGN8(kernel_n) * kerrnel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * kernel_h * ((w_in + 4 * stride_w - 1) // (4 * stride_w)) * (4 * stride_w)
                    data_threshold = 16384
                    kernel_limit = max(8192, threshold1)
                    out_limit = max(32768, threshold2)
                else:
                    aligned_kernel = ALIGN8(kernel_n) * ALIGN2(kernel_h) * kernel_w
                    split_data_size_align = ALIGN4(ch_in) * kernel_h * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_threshold = 32768
                    kernel_limit = max(32768, threshold1)
                    out_limit = max(65536, threshold2)
                split_out_size = ou_c * ou_w * kernel_h

                assert split_data_size_align <= data_threshold and aligned_kernel <= kernel_limit, \
                "input size of depthwiseConv1d cannot exceed limit"
            # common conv1d
            else:
                if platform == "venus":
                    aligned_kernel = ALIGN2(kernel_n) * ALIGN8(ch_in) * kernel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * kernel_h * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_threshold = 65536
                    kernel_limit = max(32768, threshold1)
                    out_limit = max(65536, threshold2)
                elif platform == "arcs":
                    aligned_kernel = ALIGN2(kernel_n) * ALIGN8(ch_in) * kernel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * kernel_h * ((w_in + 4 * stride_w - 1) // (4 * stride_w)) * (4 * stride_w)
                    data_threshold = 16384
                    kernel_limit = max(8192, threshold1)
                    out_limit = max(32768, threshold2)
                else:
                    aligned_kernel = ALIGN4(kernel_n) * ALIGN8(ch_in) * kernel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * kernel_h * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_threshold = 65536
                    kernel_limit = max(32768, threshold1)
                    out_limit = max(65536, threshold2)
                split_out_size = ou_c * ou_w

                if aligned_kernel <= kernel_limit and split_out_size <= out_limit:
                    continue
                flag = _sort_nodes(graph, node.outputs)

                out_size_without_cout   = ou_h * ou_w
                kernel_size_without_cout  = ALIGN8(ch_in) * kernel_h * kernel_w
                if platform in {"venus", "arcs"}:
                    channel_out_max = (kernel_limit // kernel_size_without_cout) & 0xFFFFFFFE
                    channel_out_max = min(channel_out_max, (out_limit // out_size_without_cout) & 0xFFFFFFFE)
                else:
                    channel_out_max = (out_limit // kernel_size_without_cout) & 0xFFFFFFFC
                    channel_out_max = min(channel_out_max, (out_limit // out_size_without_cout) & 0xFFFFFFFC)

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
                    conv_split_node.op  = None

                    for i in range(1, len(node.inputs)):
                        new_entry = node.inputs[i].clone()
                        new_entry.name += "_{}_{}".format(node.name, g)
                        if 1 == i:
                            weight_data = node.inputs[i].data
                            if g == split_num - 1:
                                new_entry.data = weight_data[g * channel_out_max : weight.shape[0]]
                            else:
                                new_entry.data = weight_data[g * channel_out_max : (g + 1) * channel_out_max]
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
                    conv_split_out.name += "_{}_{}".format(node.name, g)
                    if flag & set_out_dev:
                        conv_split_out.tensor.mem_type = MemType.PSRAM
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
                    split_data_size_align = ALIGN8(ch_in) * kernel_h * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_threshold = 65536
                    kernel_limit = max(32768, threshold1)
                    out_limit = max(65536, threshold2)
                elif platform == "arcs":
                    aligned_kernel = ALIGN8(kernel_n) * kernel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * kernel_h * ((w_in + 4 * stride_w - 1) // (4 * stride_w)) * (4 * stride_w)
                    data_threshold = 16384
                    kernel_limit = max(8192, threshold1)
                    out_limit = max(32768, threshold2)
                else:
                    aligned_kernel = ALIGN8(kernel_n) * ALIGN2(kernel_h) * kernel_w
                    split_data_size_align = ALIGN4(ch_in) * kernel_h * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_threshold = 32768
                    kernel_limit = max(32768, threshold1)
                    out_limit = max(65536, threshold2)
                # data_size_align_min         = split_data_size_align * kernel_h

                assert split_data_size_align <= data_threshold and aligned_kernel <= kernel_limit, "min h_in of depthwiseConv2d must not exceed limit"

            # common convolution
            else:
                if platform == "venus":
                    aligned_kernel = ALIGN2(kernel_n) * ALIGN8(ch_in) * kernel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_threshold = 65536
                    kernel_limit = max(32768, threshold1)
                    out_limit = max(65536, threshold2)
                elif platform == "arcs":
                    aligned_kernel = ALIGN2(kernel_n) * ALIGN8(ch_in) * kernel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * ((w_in + 4 * stride_w - 1) // (4 * stride_w)) * (4 * stride_w)
                    data_threshold = 16384
                    kernel_limit = max(8192, threshold1)
                    out_limit = max(32768, threshold2)
                else:
                    aligned_kernel = ALIGN4(kernel_n) * ALIGN8(ch_in) * kernel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_threshold = 65536
                    kernel_limit = max(32768, threshold1)
                    out_limit = max(65536, threshold2)
                out_size = ou_c * ou_h * ou_w

                assert split_data_size_align * kernel_h <= data_threshold, "min h_in of conv2d must not exceed limit!"
                if aligned_kernel <= kernel_limit and out_size <= out_limit:
                    continue   
                
                kernel_size_without_cout  = ALIGN8(ch_in) * kernel_h * kernel_w
                channel_out_max = ou_c
                if aligned_kernel > kernel_limit:
                    if platform in {"venus", "arcs"}:
                        channel_out_max = (kernel_limit // kernel_size_without_cout) & 0xFFFFFFFE
                    else:
                        channel_out_max = (kernel_limit // kernel_size_without_cout) & 0xFFFFFFFC

                split_h_out_max = ou_h
                split_out_size = channel_out_max * ou_w * ou_h
                if split_data_size_align * h_in > data_threshold:
                    split_h_in_max  = data_threshold // split_data_size_align
                    split_h_out_max = max((split_h_in_max - kernel_h - pads[0] + stride_h)  // stride_h, 1)
                    split_out_size = channel_out_max * ou_w * split_h_out_max

                if split_out_size > out_limit:
                    if platform in {"venus", "arcs"}:
                        channel_out_max = min(channel_out_max, (out_limit // (ou_w * split_h_out_max)) & 0xFFFFFFFE)
                    else:
                        channel_out_max = min(channel_out_max, (out_limit // (ou_w * split_h_out_max)) & 0xFFFFFFFC)

                split_num   = math.ceil(kernel_n / channel_out_max)

                if split_num == 1:
                    continue
                
                channel_out_mean = ALIGN4(math.ceil(ou_c / split_num))
                while((math.floor((channel_out_mean * ALIGN8(ch_in) * kernel_h * kernel_w) / 32768) + 1) != 
                math.floor(((channel_out_mean + 4) * ALIGN8(ch_in) * kernel_h * kernel_w) / 32768)):
                    channel_out_mean += 4

                if (channel_out_mean * ALIGN8(ch_in) * kernel_h * kernel_w <= kernel_limit) and \
                (channel_out_mean * ou_h * ou_w <= out_limit):
                    channel_out_max = channel_out_mean

                flag = _sort_nodes(graph, node.outputs)
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
                        new_entry.name += "_{}_{}".format(node.name, g)
                        if 1 == i:
                            weight_data = node.inputs[i].data
                            if g == split_num - 1:
                                new_entry.data = weight_data[g * channel_out_max : weight.shape[0]]
                            else:
                                new_entry.data = weight_data[g * channel_out_max : (g + 1) * channel_out_max]
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
                    conv_split_out.name += "_{}_{}".format(node.name, g)
                    if flag & set_out_dev:
                        conv_split_out.tensor.mem_type = MemType.PSRAM               
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

                if flag & set_out_dev:
                    _label_nodes(graph, new_node2.outputs) 
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
                    data_threshold = 65536
                    kernel_limit = max(32768, threshold1)
                    out_limit = max(65536, threshold1)
                elif platform == "arcs":
                    aligned_kernel = ALIGN8(kernel_n) * kernel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * kernel_h * ((w_in + 4 * stride_w - 1) // (4 * stride_w)) * (4 * stride_w)
                    data_threshold = 16384
                    kernel_limit = max(8192, threshold1)
                    out_limit = max(32768, threshold2)
                else:
                    aligned_kernel = ALIGN8(kernel_n) * ALIGN2(kernel_h) * kernel_w
                    split_data_size_align = ALIGN4(ch_in) * kernel_h * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_threshold = 32768
                    kernel_limit = max(32768, threshold1)
                    out_limit = max(65536, threshold2)

                assert split_data_size_align <= data_threshold and aligned_kernel <= kernel_limit, "min h_in of depthwiseConv2d must not exceed limit"
            # common ConvTranspose2dInt
            else:
                if platform == "venus":
                    aligned_kernel = ALIGN2(kernel_n) * ALIGN8(ch_in) * kernel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_threshold = 65536
                    kernel_limit = max(32768, threshold1)
                    out_limit = max(65536, threshold2)
                elif platform == "arcs":
                    aligned_kernel = ALIGN2(kernel_n) * ALIGN8(ch_in) * kernel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * ((w_in + 4 * stride_w - 1) // (4 * stride_w)) * (4 * stride_w)
                    data_threshold = 16384
                    kernel_limit = max(8192, threshold1)
                    out_limit = max(32768, threshold2)
                else:
                    aligned_kernel = ALIGN4(kernel_n) * ALIGN8(ch_in) * kernel_h * kernel_w
                    split_data_size_align = ALIGN8(ch_in) * ((w_in + 8 * stride_w - 1) // (8 * stride_w)) * (8 * stride_w)
                    data_threshold = 65536
                    kernel_limit = max(32768, threshold1)
                    out_limit = max(65536, threshold2)
                out_size = ou_c * ou_h * ou_w

                assert split_data_size_align * kernel_h <= data_threshold, "min h_in of ConvTranspose2dInt must not exceed limit!"
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
                if split_data_size_align * h_in > data_threshold:
                    split_h_in_max  = data_threshold // split_data_size_align
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
                        new_entry.name += "_{}_{}".format(node.name, g)
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
                    conv_split_out.name += "_{}_{}".format(node.name, g)
              
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
            transB          = node.attrs["transB"]
            M   = 1
            for i in range(len(data.shape)-1):
                M *= data.shape[i]
            L   = weight.shape[0] if transB else weight.shape[1]
            N   = weight.shape[1] if transB else weight.shape[0]
            assert N == data.shape[-1]

            split_num = 1
            if platform == "venus":
                assert weight.dtype in (np.int8, np.int16, np.int32)
                if weight.dtype == np.int8:
                    left_size_align     = ALIGN4(M) * ALIGN8(N)
                    right_size_align    = ALIGN8(N) * ALIGN4(L)
                    left_size_limit     = 65536
                    right_size_limit    = max(32768, threshold3)
                    int8_condition_r    = right_size_align
                    while int8_condition_r > right_size_limit:
                        split_num       += 1
                        split_L         = ALIGN4(math.ceil(L / split_num))
                        int8_condition_r = ALIGN8(N) * split_L
                elif weight.dtype == np.int16:
                    left_size_align     = ALIGN4(M) * ALIGN2(N)
                    right_size_align    = ALIGN2(N) * ALIGN4(L)
                    left_size_limit     = 32768
                    right_size_limit    = max(16384, threshold3)
                    int8_condition_r    = right_size_align
                    while int8_condition_r > right_size_limit:
                        split_num       += 1
                        split_L         = ALIGN4(math.ceil(L / split_num))
                        int8_condition_r = ALIGN2(N) * split_L                  
                else:
                    left_size_align     = ALIGN2(M) * ALIGN2(N)
                    right_size_align    = ALIGN2(N) * ALIGN2(L)
                    left_size_limit     = 16384
                    right_size_limit    = max(8192, threshold3)
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
                    left_size_limit     = 65536
                    right_size_limit    = max(65536, threshold3)
                elif weight.dtype == np.int16:
                    left_size_align     = ALIGN4(M) * ALIGN8(N)
                    right_size_align    = ALIGN8(N) * ALIGN8(L)
                    left_size_limit     = 32768
                    right_size_limit    = max(32768, threshold3)
                else:
                    left_size_align     = ALIGN4(M) * ALIGN8(N)
                    right_size_align    = ALIGN8(N) * ALIGN8(L)
                    left_size_limit     = 16384
                    right_size_limit    = max(16384, threshold3)

                int8_condition_r    = right_size_align
                while int8_condition_r > right_size_limit:
                    split_num       += 1
                    split_L         = ALIGN8(math.ceil(L / split_num))
                    int8_condition_r = ALIGN8(N) * split_L

            if split_num == 1:
                continue

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
