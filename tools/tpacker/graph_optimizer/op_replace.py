import numpy as np

from ..xsympy import is_sympy
from ..save_model import save_to_onnx_model
from ..graph_analysis.ops.utils import calc_expr
from ..graph import Graph, GraphNode, Tensor, ConstantEntry
from ..platform_utils import normalize_platform_name


node_list = ["Conv2dInt", "ConvTranspose2dInt"]


def _supports_strided_concat(platform):
    return normalize_platform_name(platform) in {"arcs", "venusA"}

def check_same_pads(dst_nodes, is_stream):
    """
    检查所有分支的pad参数是否相同
    Args:
        dst_nodes: 目标节点列表
        is_stream: split_w 或 split_h
    Returns:
        True 如果所有分支pad相同, False 否则
    """
    conv_nodes = [n for n in dst_nodes if n.op_type == "Conv2dInt"]
    if len(conv_nodes) <= 1:
        return True

    pads_list = []
    for n in conv_nodes:
        pads = list(n.attrs.get('pads', []))
        pads_list.append(tuple(pads))

    # 检查是否所有 pad 都相同
    return len(set(pads_list)) == 1

def get_conv_branches(graph):
    """
    收集图中所有存在分叉的输入及其对应的Conv2dInt节点
    Returns:
        dict: {input_entry: [Conv2dInt节点列表]}
    """
    branch_map = {}
    for node in graph.nodes.values():
        if node.op_type == "Conv2dInt" and len(node.inputs) > 0:
            input_entry = node.inputs[0]
            if input_entry and input_entry.src_node is not None:
                src_output = input_entry.src_node.outputs[0]
                if len(src_output.dst_nodes) > 1:
                    # 存在分叉：多个节点使用同一个输入
                    if input_entry not in branch_map:
                        branch_map[input_entry] = []
                    branch_map[input_entry].append(node)
    return branch_map

def split_conv1d(graph: Graph, is_stream: str):
    add_node_list = list()
    conv_nodes_with_branch = []
    for node in graph.nodes.values():
        pads_flag = False
        if node.op_type == "Conv1dInt":
            data    = node.inputs[0].tensor
            kernels = node.attrs['kernel_shape']
            strides = node.attrs['strides']
            pads    = list(node.attrs['pads'])

            x_h     = calc_expr(str(data.shape[2]), graph.dynamic_args_max) if is_sympy(data.shape[2]) else data.shape[1]

            if kernels[0] == 1:
                continue
            if len(pads) == 2:
              if (pads[0], pads[1]) != (0, 0):
                  pads_flag = True
            elif len(pads) == 1:
              if pads[0] != 0:
                pads_flag = True

            if pads_flag:
                # 分叉检测：如果输入被多个Conv1dInt使用，检查pad是否相同
                input_entry = node.inputs[0]
                conv_branches = []
                same_pads = True
                if input_entry.src_node is not None:
                    src_output = input_entry.src_node.outputs[0]
                    dst_nodes = src_output.dst_nodes
                    conv_branches = [n for n in dst_nodes if n.op_type == "Conv1dInt"]
                    if len(conv_branches) > 1:
                        same_pads = check_same_pads(conv_branches, is_stream)
                        # pad相同时，只在第一个分支处理
                        if same_pads and conv_nodes_with_branch:
                            # 检查是否已经添加过这个分支组
                            prev_input = conv_nodes_with_branch[-1][0].inputs[0] if conv_nodes_with_branch else None
                            if prev_input is input_entry:
                                continue
                conv_nodes_with_branch.append((node, conv_branches, same_pads))

    # 统一创建iqPad节点
    for node, conv_branches, same_pads in conv_nodes_with_branch:
        pads = list(node.attrs['pads'])
        new_node    = GraphNode("iqPad", node.name + "_pads")
        new_node.attrs['mode'] = 'constant'
        new_node.attrs['platform'] = node.attrs['platform']
        new_entry   = node.inputs[0].clone()
        new_entry.name += "_prepads"
        new_entry.set_graph_normal()
        graph.add_entry(new_entry)
        new_node.inputs.append(node.inputs[0])
        new_node.outputs.append(new_entry)
        if node.inputs[0].src_node == None:
            continue
        dst_nodes = node.inputs[0].src_node.outputs[0].dst_nodes

        if same_pads and len(conv_branches) > 1:
            # pad相同：在根节点统一提取，所有分支连接同一个iqPad
            for n in dst_nodes:
                n.inputs[0] = new_entry
        else:
            # 无分支或pad不一致：只连接当前节点
            node.inputs[0] = new_entry

        pads_value = np.zeros((6,), dtype=np.int64, order = "C")
        if len(pads) == 1:
            pads_value[-1] = pads[-1]
            pads_value[-4] = pads[-1]
            pads[-1] = 0
        elif len(pads) == 2:
            pads_value[-1] = pads[-1]
            pads_value[-4] = pads[-2]
            pads[-1] = 0
            pads[-2] = 0

        # 根据连接方式更新对应节点的pads属性
        if same_pads and len(conv_branches) > 1:
            for n in dst_nodes:
                n.attrs['pads'] = tuple(pads)
        else:
            node.attrs['pads'] = tuple(pads)

        pads_value_tensor = Tensor.from_numpy(pads_value)
        pads_entry = ConstantEntry(new_node.name+'_pads', pads_value_tensor)
        graph.add_entry(pads_entry)
        new_node.inputs.append(pads_entry)

        data_value = np.zeros((1,), dtype=np.int64, order = "C")
        data_tensor = Tensor.from_numpy(data_value)
        data_entry = ConstantEntry(new_node.name+'_data', data_tensor)
        graph.add_entry(data_entry)
        new_node.inputs.append(data_entry)
        add_node_list.append(new_node)

    for node in add_node_list:
        graph.add_node(node)
    
    graph.update()

def split_conv2d(graph: Graph, is_stream: str):
    add_node_list = list()
    # 先收集所有需要处理的卷积节点及其分叉信息
    conv_nodes_with_branch = []  # [(node, conv_branches, pad相同标志)]
    for node in graph.nodes.values():
        pads_flag = False
        if node.op_type == "Conv2dInt":
            data    = node.inputs[0].tensor
            kernels = node.attrs['kernel_shape']
            strides = node.attrs['strides']
            pads    = list(node.attrs['pads'])

            x_h     = calc_expr(str(data.shape[1]), graph.dynamic_args_max) if is_sympy(data.shape[1]) else data.shape[1]
            x_w     = calc_expr(str(data.shape[2]), graph.dynamic_args_max) if is_sympy(data.shape[2]) else data.shape[2]

            if kernels[0] == 1 and (is_stream == "split_w"):
                continue
            if kernels[1] == 1 and (is_stream == "split_w"):
                continue
            if ((pads[-1], pads[-3]) != (0, 0) and (is_stream == "split_w")):
                pads_flag = True
            if ((pads[-2], pads[-4]) != (0, 0) and (is_stream == "split_h")):
                pads_flag = True

            if pads_flag:
                # 分叉检测：如果输入被多个Conv2dInt使用，检查pad是否相同
                input_entry = node.inputs[0]
                conv_branches = []
                same_pads = True
                if input_entry.src_node is not None:
                    src_output = input_entry.src_node.outputs[0]
                    dst_nodes = src_output.dst_nodes
                    conv_branches = [n for n in dst_nodes if n.op_type == "Conv2dInt"]
                    if len(conv_branches) > 1:
                        same_pads = check_same_pads(conv_branches, is_stream)
                        # pad相同时，只在第一个分支处理
                        if same_pads and conv_nodes_with_branch:
                            # 检查是否已经添加过这个分支组
                            prev_input = conv_nodes_with_branch[-1][0].inputs[0] if conv_nodes_with_branch else None
                            if prev_input is input_entry:
                                continue
                conv_nodes_with_branch.append((node, conv_branches, same_pads))

    # 统一创建iqPad节点
    for node, conv_branches, same_pads in conv_nodes_with_branch:
        pads = list(node.attrs['pads'])
        new_node    = GraphNode("iqPad", node.name + "_pads")
        new_node.attrs['mode'] = 'constant'
        new_node.attrs['platform'] = node.attrs['platform']
        new_entry   = node.inputs[0].clone()
        new_entry.name += "_prepads"
        new_entry.set_graph_normal()
        graph.add_entry(new_entry)
        new_node.inputs.append(node.inputs[0])
        new_node.outputs.append(new_entry)
        if node.inputs[0].src_node == None:
            continue
        dst_nodes = node.inputs[0].src_node.outputs[0].dst_nodes

        if same_pads and len(conv_branches) > 1:
            # pad相同：在根节点统一提取，所有分支连接同一个iqPad
            for n in dst_nodes:
                n.inputs[0] = new_entry
        else:
            # 无分支或pad不一致：只连接当前节点
            node.inputs[0] = new_entry

        pads_value = np.zeros((8,), dtype=np.int64, order = "C")
        # 从w维度拆分
        if is_stream == "split_w":
          pads_value[-1] = pads[-1]
          if len(pads) == 2:
              pads_value[-5] = pads[-1]
              pads[-1] = 0
          elif len(pads) == 4:
              pads_value[-5] = pads[-3]
              pads[-3] = 0
              pads[-1] = 0
        elif is_stream == "split_h":
        # # 从h维度拆分
          pads_value[-2] = pads[-2]
          if len(pads) == 2:
              pads_value[-6] = pads[-2]
              pads[-2] = 0
          elif len(pads) == 4:
              pads_value[-6] = pads[-4]
              pads[-2] = 0
              pads[-4] = 0

        # 根据连接方式更新对应节点的pads属性
        if same_pads and len(conv_branches) > 1:
            for n in dst_nodes:
                n.attrs['pads'] = tuple(pads)
        else:
            node.attrs['pads'] = tuple(pads)

        slice_start = Tensor.from_numpy(pads_value)
        pads_entry = ConstantEntry(new_node.name+'_pads', slice_start)
        graph.add_entry(pads_entry)
        new_node.inputs.append(pads_entry)

        data_value = np.zeros((1,), dtype=np.int64, order = "C")
        data_tensor = Tensor.from_numpy(data_value)
        data_entry = ConstantEntry(new_node.name+'_data', data_tensor)
        graph.add_entry(data_entry)
        new_node.inputs.append(data_entry)
        add_node_list.append(new_node)

    for node in add_node_list:
        graph.add_node(node)
    
    graph.update()

def stream_convert(graph: Graph, is_stream: str, is_dump: bool = False) -> Graph:
    """
    Args:
      graph: Graph
        待处理graph
      is_dump: bool
        是否保存处理后的graph，默认保存，即保存；
      dump_file_path: str
        保存路径，默认保存到./3_graph_stream_convert.onnx
    Returns:
      g: Graph
        改造后的graph
    Function Description:
      计算图流式改造，需要iqPad标识需要做拼接的维度，支持因果和非因果卷积
    """

    split_conv1d(graph, is_stream)
    split_conv2d(graph, is_stream)
    graph.init_tensor()

    add_nodes_list = list()
    del_nodes_list = list()
    graph_inputs = list(graph.inputs)
    graph_outputs = list(graph.outputs)
    num_otuput = len(graph_outputs)
    for node in graph.nodes.values():
      if node.op_type == 'iqPad':
        # assert len(node.outputs[0].dst_nodes) == 1, "The number of outputs of the iqPad operator must equal 1"
        data_shape = node.inputs[0].tensor.shape
        data_shape_list = list(data_shape)
        pads = node.inputs[1].tensor
        pads_len = pads.shape[0] // 2
        pads_value = list()
        for i in range(pads_len):
          pads_value.append(pads.data[i] + pads.data[pads_len + i])

        pad_axis = list()
        for i in range(pads_len):
          if pads_value[pads_len - i - 1]:
            data_shape_list[len(data_shape_list) - i - 1] = pads_value[pads_len - i - 1]
            pad_axis.append(len(data_shape_list) - i - 1)
        assert len(pad_axis) == 1, "invalid parameter for iqPad"
        # if len(pad_axis) != 1:
        #   breakpoint()

        platform = node.attrs.get('platform', graph.platform)
        if _supports_strided_concat(platform):
          axis = pad_axis[0]
          history_shape = list(data_shape)
          history_shape[axis] = pads_value[axis]

          history = node.inputs[0].clone()
          history.name = node.inputs[0].name + '_transpose_history'
          history.tensor.shape = tuple(history_shape)
          history.set_graph_input()
          graph.add_entry(history)
          graph_inputs.append(history)

          scale = pow(2, node.inputs[0].tensor.scale)
          concat = GraphNode("iqCat", node.name + '_concat')
          concat.attrs['dim'] = axis
          concat.attrs['scale_x_0'] = scale
          concat.attrs['scale_x_1'] = scale
          concat.attrs['scale_o'] = scale
          concat.attrs['platform'] = platform
          concat.inputs = [history, node.inputs[0]]
          concat.outputs = node.outputs
          add_nodes_list.append(concat)

          start = np.array([-pads_value[axis]], dtype=np.int64)
          state_start = ConstantEntry(node.name + '_transpose2_start', Tensor.from_numpy(start))
          graph.add_entry(state_start)
          end = np.array([65536], dtype=np.int64)
          state_end = ConstantEntry(node.name + '_transpose2_end', Tensor.from_numpy(end))
          graph.add_entry(state_end)
          axes = np.array([axis], dtype=np.int64)
          state_axes = ConstantEntry(node.name + '_transpose2_axes', Tensor.from_numpy(axes))
          graph.add_entry(state_axes)

          state = node.inputs[0].clone()
          state.name = node.name + '_transpose2_state'
          state.tensor.shape = tuple(history_shape)
          graph.add_entry(state)
          state_slice = GraphNode("Slice", node.name + '_transpose2_slice')
          state_slice.inputs = [node.inputs[0], state_start, state_end, state_axes]
          state_slice.outputs = [state]
          add_nodes_list.append(state_slice)

          if pads.data[pads_len + axis] != 0:
            src_node = node.inputs[0].src_node
            branches = src_node.outputs[0].dst_nodes if src_node is not None else []
            iqadd_branches = [branch for branch in branches if branch.op_type == "iqAdd"]
            if iqadd_branches:
              branch = iqadd_branches[-1]
              branch_start = ConstantEntry(
                  branch.name + '_start', Tensor.from_numpy(np.array([1], dtype=np.int64)))
              branch_end = ConstantEntry(
                  branch.name + '_end', Tensor.from_numpy(np.array([-1], dtype=np.int64)))
              branch_axes = ConstantEntry(
                  branch.name + '_axes', Tensor.from_numpy(np.array([2], dtype=np.int64)))
              graph.add_entry(branch_start)
              graph.add_entry(branch_end)
              graph.add_entry(branch_axes)
              branch_state = branch.inputs[0].clone()
              branch_state.name = node.name + '_state'
              graph.add_entry(branch_state)
              branch_slice = GraphNode("Slice", node.name + '_slice')
              branch_slice.inputs = [node.outputs[0], branch_start, branch_end, branch_axes]
              branch_slice.outputs = [branch_state]
              add_nodes_list.append(branch_slice)
              branch.inputs[0] = branch_state

          state.set_graph_output()
          graph_outputs.insert(-1 * num_otuput, state)

          for next_node in node.outputs[0].dst_nodes:
            if next_node.op_type == "Reshape":
              next_node = next_node.outputs[0].dst_nodes[0]
            if next_node.op_type == "Conv2dInt" and len(node.outputs[0].dst_nodes) == 1:
              if next_node.attrs['dilations'] != (1, 1):
                next_node.attrs['dilations'] = (1, 1)
                kernel_axis = 0 if axis == 2 else 1
                history_shape[axis] = next_node.attrs['kernel_shape'][kernel_axis] - 1
          history.tensor.shape = tuple(history_shape)
          del_nodes_list.append(node)
          continue

        new_entry1 = node.inputs[0].clone()
        new_entry1.name = new_entry1.name + '_transpose'
        new_entry1.set_graph_normal()

        assert len(data_shape) in {3 ,4}, "just support 3D/4D data for stream convert"
        new_shape = tuple()
        if len(data_shape) == 4:
          if pad_axis[0] == 2:
            new_shape = (0, 2, 1 ,3)
          elif pad_axis[0] == 3:
            new_shape = (0, 3, 1, 2)
        else:
          if pad_axis[0] == 1:
            new_shape = (1, 0, 2)
          elif pad_axis[0] == 2:
            new_shape = (2, 0, 1)

        if len(data_shape) == 4:
          new_entry1.tensor.shape = (data_shape[new_shape[0]], data_shape[new_shape[1]], data_shape[new_shape[2]], data_shape[new_shape[3]])
        else:
          new_entry1.tensor.shape = (data_shape[new_shape[0]], data_shape[new_shape[1]], data_shape[new_shape[2]])
        graph.add_entry(new_entry1)

        # add transpose
        new_entry2 = new_entry1.clone()
        new_entry2.name = new_entry2.name + '_status'
        if len(data_shape) == 4:
          new_entry2.tensor.shape = (new_entry1.tensor.shape[0], pads_value[pad_axis[0]], new_entry1.tensor.shape[1], new_entry1.tensor.shape[2])
        else:
          new_entry2.tensor.shape = (pads_value[pad_axis[0]], new_entry1.tensor.shape[1], new_entry1.tensor.shape[2])
        graph.add_entry(new_entry2)

        new_entry3 = new_entry1.clone()
        new_entry3.name = new_entry3.name + '_history'
        if len(data_shape) == 4:
          new_entry3_shape = [new_entry1.tensor.shape[0], pads_value[pad_axis[0]], new_entry1.tensor.shape[2], new_entry1.tensor.shape[3]]
        else:
          new_entry3_shape = [pads_value[pad_axis[0]], new_entry1.tensor.shape[1], new_entry1.tensor.shape[2]]
        graph.add_entry(new_entry3)
        new_entry3.set_graph_input()
        graph_inputs.append(new_entry3)

        new_node1 = GraphNode("Transpose", node.name + '_transpose1')
        new_node1.attrs['ndim'] = len(data_shape_list)
        new_node1.attrs['perm'] = new_shape
        add_nodes_list.append(new_node1)

        new_node2 = GraphNode("Transpose", node.name + '_transpose2')
        new_node2.attrs['ndim'] = len(data_shape_list)
        if len(data_shape) == 4:
          if new_shape == (0, 2, 1 ,3):
            new_node2.attrs['perm'] = (0, 2, 1, 3)
          else:
            new_node2.attrs['perm'] = (0, 2, 3, 1)
        else:
          if new_shape == (2, 0, 1):
            new_node2.attrs['perm'] = (1, 2, 0)
          else:
            new_node2.attrs['perm'] = (1, 0, 2)        
        add_nodes_list.append(new_node2)

        scale = pow(2, node.inputs[0].tensor.scale)
        new_node3 = GraphNode("iqCat", node.name + '_concat')
        if len(data_shape) == 4:
          new_node3.attrs['dim'] = 1
        elif len(data_shape) == 3:
          new_node3.attrs['dim'] = 0
        new_node3.attrs['scale_x_0'] = scale
        new_node3.attrs['scale_x_1'] = scale
        new_node3.attrs['scale_o'] = scale
        new_node3.attrs['platform'] = node.attrs['platform']
        add_nodes_list.append(new_node3)

        start = np.zeros((1,), dtype=np.int64, order = "C")
        start[0] = -1 * (pads_value[pad_axis[0]])
        slice_start = Tensor.from_numpy(start)
        new_entry9 = ConstantEntry(new_node2.name+'_start', slice_start)
        graph.add_entry(new_entry9)

        end = np.zeros((1,), dtype=np.int64, order = "C")
        end[0] = 65536
        slice_end = Tensor.from_numpy(end)
        new_entry10 = ConstantEntry(new_node2.name+'_end', slice_end)
        graph.add_entry(new_entry10)

        axes = np.zeros((1,), dtype=np.int64, order = "C")
        axes[0] = 1
        slice_axes = Tensor.from_numpy(axes)
        new_entry11 = ConstantEntry(new_node2.name+'_axes', slice_axes)
        graph.add_entry(new_entry11)

        new_entry12 = new_entry1.clone()
        new_entry12.name = new_node2.name + '_state'
        graph.add_entry(new_entry12)

        new_node5 = GraphNode("Slice", new_node2.name + '_slice')
        add_nodes_list.append(new_node5)


        if pads.data[pads_len + pad_axis[0]] != 0:
          src_node1 = node.inputs[0].src_node
          if src_node1 != None:
            all_branches = src_node1.outputs[0].dst_nodes
          else:
            all_branches = list()
          resnet_flag = False
          for _, branch in enumerate(all_branches):
            if branch.op_type == "iqAdd":
              start = np.zeros((1,), dtype=np.int64, order = "C")
              start[0] = 1
              slice_start = Tensor.from_numpy(start)
              new_entry5 = ConstantEntry(branch.name+'_start', slice_start)
              graph.add_entry(new_entry5)

              end = np.zeros((1,), dtype=np.int64, order = "C")
              end[0] = -1
              slice_end = Tensor.from_numpy(end)
              new_entry6 = ConstantEntry(branch.name+'_end', slice_end)
              graph.add_entry(new_entry6)

              axes = np.zeros((1,), dtype=np.int64, order = "C")
              axes[0] = 2
              slice_axes = Tensor.from_numpy(axes)
              new_entry7 = ConstantEntry(branch.name+'_axes', slice_axes)
              graph.add_entry(new_entry7)

              new_entry8 = branch.inputs[0].clone()
              new_entry8.name = node.name + '_state'
              graph.add_entry(new_entry8)

              new_node4 = GraphNode("Slice", node.name + '_slice')
              add_nodes_list.append(new_node4)
              resnet_flag = True
              index = _

          if resnet_flag:
            new_node1.inputs = [src_node1.outputs[0]]
            new_node1.outputs = [new_entry1]

            new_node2.inputs = [new_entry2]
            new_node2.outputs = node.outputs

            new_node3.inputs = [new_entry3, new_entry1] # iqCat
            new_node3.outputs = [new_entry2]

            new_node4.inputs = [node.outputs[0], new_entry5, new_entry6, new_entry7]
            new_node4.outputs = [new_entry8]

            all_branches[index].inputs = [new_entry8, all_branches[index].inputs[1]]
          else:
            new_node1.inputs = [node.inputs[0]]         # transpose
            new_node1.outputs = [new_entry1]
            new_node3.inputs = [new_entry3, new_entry1] # iqCat
            new_node3.outputs = [new_entry2]
            new_node2.inputs = [new_entry2]             # transpose
            new_node2.outputs = node.outputs
          
          new_node5.inputs = [new_entry1, new_entry9, new_entry10, new_entry11]
          new_node5.outputs = [new_entry12]
        else:
          new_node1.inputs = [node.inputs[0]]         # transpose
          new_node1.outputs = [new_entry1]
          new_node3.inputs = [new_entry3, new_entry1] # iqCat
          new_node3.outputs = [new_entry2]
          new_node2.inputs = [new_entry2]             # transpose
          new_node2.outputs = node.outputs

          new_node5.inputs = [new_entry1, new_entry9, new_entry10, new_entry11]
          new_node5.outputs = [new_entry12]

        new_entry12.set_graph_output()
        graph_outputs.insert(-1*num_otuput, new_entry12)

        # if len(node.outputs[0].dst_nodes) != 1:
        #   return
        next_nodes = node.outputs[0].dst_nodes
        for next_node in next_nodes:
          if next_node.op_type == "Reshape":
            next_node = next_node.outputs[0].dst_nodes[0]
          if next_node.op_type == "Conv2dInt" and len(node.outputs[0].dst_nodes) == 1:
            next_node_dilations = next_node.attrs['dilations']
            next_node_kernels = next_node.attrs['kernel_shape']
            if next_node_dilations != (1, 1):
              next_node.attrs['dilations'] = (1, 1)
              if len(data_shape) == 4:
                new_entry3_shape[1] = next_node_kernels[0] - 1
              else:
                new_entry3_shape[0] = next_node_kernels[0] - 1
        new_entry3.tensor.shape = tuple(new_entry3_shape)

        del_nodes_list.append(node)
      
      elif node.op_type == "ConvTranspose2dInt":
        kernels = node.attrs['kernel_shape']
        pads = node.attrs['pads']
        pads_len = len(pads)
        if pads[-2] ==0 and pads[0] ==0 :# just support stream convert in H
          continue
        
        strides = node.attrs['strides']
        output_padding = node.attrs['output_padding']
        input_shape = node.inputs[0].tensor.shape
        scale = node.attrs['scale_x']

        platform = node.attrs.get('platform', graph.platform)
        if _supports_strided_concat(platform):
          assert len(input_shape) == 4, "ConvTranspose2dInt stream convert only supports 4D data"
          axis = 2
          history_shape = list(input_shape)
          history_shape[axis] = pads[0]

          history = node.inputs[0].clone()
          history.name = node.inputs[0].name + '_transpose_history'
          history.tensor.shape = tuple(history_shape)
          history.set_graph_input()
          graph.add_entry(history)
          graph_inputs.append(history)

          concat_output = node.inputs[0].clone()
          concat_output.name = node.inputs[0].name + '_concat'
          concat_output.set_graph_normal()
          concat_output.tensor.shape = tuple(
              history_shape[i] + input_shape[i] if i == axis else input_shape[i]
              for i in range(len(input_shape)))
          graph.add_entry(concat_output)
          concat = GraphNode("iqCat", node.name + '_concat')
          concat.attrs['dim'] = axis
          concat.attrs['scale_x_0'] = scale
          concat.attrs['scale_x_1'] = scale
          concat.attrs['scale_o'] = scale
          concat.attrs['platform'] = platform
          concat.inputs = [history, node.inputs[0]]
          concat.outputs = [concat_output]
          add_nodes_list.append(concat)
          node.inputs[0] = concat_output

          state_start = ConstantEntry(
              node.name + '_start', Tensor.from_numpy(np.array([-pads[0]], dtype=np.int64)))
          state_end = ConstantEntry(
              node.name + '_end', Tensor.from_numpy(np.array([65536], dtype=np.int64)))
          state_axes = ConstantEntry(
              node.name + '_axes', Tensor.from_numpy(np.array([axis], dtype=np.int64)))
          graph.add_entry(state_start)
          graph.add_entry(state_end)
          graph.add_entry(state_axes)
          state = node.inputs[0].clone()
          state.name = node.name + '_state'
          state.tensor.shape = tuple(history_shape)
          graph.add_entry(state)
          state_slice = GraphNode("Slice", node.name + '_slice')
          state_slice.inputs = [concat_output, state_start, state_end, state_axes]
          state_slice.outputs = [state]
          add_nodes_list.append(state_slice)
          state.set_graph_output()
          graph_outputs.insert(-1 * num_otuput, state)

          new_pads = list(pads)
          new_pads[0] = kernels[0] - 1
          new_pads[2] = kernels[0] - strides[0] + output_padding[0]
          node.attrs['pads'] = tuple(new_pads)
          continue

        # add transpose
        new_entry1 = node.inputs[0].clone()
        new_entry1.name = new_entry1.name + '_transpose'
        if len(input_shape) == 4:
          new_entry1.tensor.shape = (input_shape[0], input_shape[2], input_shape[1], input_shape[3])
        elif len(input_shape) == 3:
          new_entry1.tensor.shape = (input_shape[1], input_shape[0], input_shape[2])
        graph.add_entry(new_entry1)
        new_node1 = GraphNode("Transpose", node.name + '_transpose1')
        new_node1.attrs['ndim'] = len(input_shape)
        if len(input_shape) == 4:
          new_shape = (0, 2, 1 ,3)
        else:
          new_shape = (1, 0, 2)
        new_node1.attrs['perm'] = new_shape
        new_node1.inputs = [node.inputs[0]]
        new_node1.outputs = [new_entry1]
        add_nodes_list.append(new_node1)

        # add transpose
        new_entry2 = new_entry1.clone()
        new_entry2.name = new_entry2.name + '_transpose_concat'
        graph.add_entry(new_entry2)
        new_entry3 = node.inputs[0].clone()
        new_entry3.name = new_entry2.name + '_transpose'
        graph.add_entry(new_entry3)
        new_node2 = GraphNode("Transpose", node.name + '_transpose2')
        new_node2.attrs['ndim'] = len(input_shape)
        new_node2.attrs['perm'] = new_shape
        new_node2.inputs = [new_entry2]
        new_node2.outputs = [new_entry3]
        add_nodes_list.append(new_node2)
        node.inputs[0] = new_entry3
        
        # add iqCat
        new_entry4 = new_entry1.clone()
        new_entry4.name = new_entry4.name + '_history'
        if len(input_shape) == 4:
          new_entry4.tensor.shape = (new_entry1.tensor.shape[0], pads[0], new_entry1.tensor.shape[2], new_entry1.tensor.shape[3])
        else:
          new_entry4.tensor.shape = (1, new_entry1.tensor.shape[1], new_entry1.tensor.shape[2])
        graph.add_entry(new_entry4)
        new_entry4.set_graph_input()
        graph_inputs.append(new_entry4)
        new_node3 = GraphNode("iqCat", node.name + '_concat')
        if len(input_shape) == 4:
          new_node3.attrs['dim'] = 1
        elif len(input_shape) == 3:
          new_node3.attrs['dim'] = 0
        new_node3.attrs['scale_x_0'] = scale
        new_node3.attrs['scale_x_1'] = scale
        new_node3.attrs['scale_o'] = scale
        new_node3.inputs = [new_entry4, new_entry1]
        new_node3.outputs = [new_entry2]
        add_nodes_list.append(new_node3)

        if input_shape[-2] == 1:
          start = np.zeros((1,), dtype=np.int64, order = "C")
          start[0] = -1
          slice_start = Tensor.from_numpy(start)
          new_entry5 = ConstantEntry(node.name+'_start', slice_start)
          graph.add_entry(new_entry5)

          end = np.zeros((1,), dtype=np.int64, order = "C")
          end[0] = 65537
          slice_end = Tensor.from_numpy(end)
          new_entry6 = ConstantEntry(node.name+'_end', slice_end)
          graph.add_entry(new_entry6)

          axes = np.zeros((1,), dtype=np.int64, order = "C")
          if len(input_shape) == 4:
            axes[0] = 1
          elif len(input_shape) == 3:
            axes[0] = 0
          slice_axes = Tensor.from_numpy(axes)
          new_entry7 = ConstantEntry(node.name+'_axes', slice_axes)
          graph.add_entry(new_entry7)

          new_entry8 = new_entry2.clone()
          new_entry8.tensor.shape = input_shape
          new_entry8.name = node.name + '_state'
          graph.add_entry(new_entry8)

          new_node4 = GraphNode("Slice", node.name + '_slice')
          new_node4.inputs = [new_entry2, new_entry5, new_entry6, new_entry7]
          new_node4.outputs = [new_entry8]
          add_nodes_list.append(new_node4)
          new_entry7.set_graph_output()
          graph_outputs.insert(-1*num_otuput, new_entry8)

        else:
          new_entry1.set_graph_output()
          graph_outputs.insert(-1*num_otuput, new_entry1)

        new_pads = list(pads)
        new_pads[0] = kernels[0] - 1
        new_pads[2] = kernels[0] -strides[0] + output_padding[0]
        
        node.attrs['pads'] = tuple(new_pads)
      elif node.op_type == "GRUInt":
        hidden_in = node.inputs[1]
        if hidden_in.data is not None:
          assert not np.any(hidden_in.data), "input of gru stream must have zero hidden state"
        hidden_in.data = None
        hidden_in.scale = node.inputs[0].tensor.scale
        hidden_in.set_graph_input()
        graph_inputs.append(hidden_in)
        hidden_out = node.outputs[1]
        hidden_out.set_graph_output()
        graph_outputs.insert(-num_otuput, hidden_out)
      elif node.op_type == "LSTMInt":
        hidden_in = node.inputs[1]
        if hidden_in.data is not None:
          assert not np.any(hidden_in.data), "input of lstm stream must have zero hidden state"
        hidden_in.set_graph_input()
        graph_inputs.append(hidden_in)
        cell_in = node.inputs[2]
        if cell_in.data is not None:
          assert not np.any(cell_in.data), "input of lstm stream must have zero cell state"
        cell_in.set_graph_input()
        graph_inputs.append(cell_in)
        hidden_out = node.outputs[1]
        hidden_out.set_graph_output()
        graph_outputs.insert(-num_otuput, hidden_out)
        cell_out = node.outputs[2]
        cell_out.set_graph_output()
        graph_outputs.insert(-num_otuput, cell_out)
    for node in add_nodes_list:
      graph.add_node(node)

    for node in del_nodes_list:
      del graph.nodes[node]

    graph.inputs = graph_inputs
    graph.outputs = graph_outputs
    graph.update()

    if is_dump:
      save_to_onnx_model(graph, f"./workspace/{graph.name}/model.ignore/3_graph_stream_convert.onnx")
    return graph

__all__ = ["stream_convert"]
