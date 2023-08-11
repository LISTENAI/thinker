import numpy as np
from ..graph import Graph, GraphNode, Tensor, ConstantEntry
from ..save_model import save_to_onnx_model

node_list = ["Conv2dInt", "ConvTranspose2dInt"]
def stream_convert(
    graph: Graph,
    is_dump: bool = False,
    dump_file_path: str = "./model.ignore/graph_stream_convert.onnx",
) -> Graph:

    """
    Args:
      graph: Graph
        待融合graph
      ignore_methods: List[str]
        忽略特定算子融合;
      is_dump: bool
        是否保存融合后的graph，默认保存，即保存；
      dump_file_path: str
        保存路径，默认保存到./graph_stream_convert.onnx
    Returns:
      g: Graph
        融合后的graph
    """
    add_nodes_list = list()
    del_nodes_list = list()
    graph_inputs = list(graph.inputs)
    graph_outputs = list(graph.outputs)
    for node in graph.nodes.values():
      if node.op_type == 'iqPad':
        assert len(node.outputs[0].dst_nodes) == 1, "do not support num of dst nodes > 1"
        out_pad_axis = 0
        pre_node = node.inputs[0].src_node
        next_node = node.outputs[0].dst_nodes[0]
        for i in range(len(next_node.inputs)):
          if next_node.inputs[i] == node.outputs[0]:
            out_pad_axis = i
        # get the scale info
        if ('scale_x' in next_node.attrs) & (out_pad_axis == 0):
          scale = next_node.attrs['scale_x']
        elif ('scale_y' in next_node.attrs) & (out_pad_axis == 1):
          scale = next_node.attrs['scale_y']
        elif 'scale_o' in pre_node.attrs:
          scale = pre_node.attrs['scale_o']
        else:
          raise AssertionError
        
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

        # insert node of transpose
        new_entry1 = node.inputs[0].clone()
        new_entry1.name = new_entry1.name + '_transpose'
        assert len(data_shape) in {3 ,4}, "just support 3D/4D data"

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
        new_node1 = GraphNode("Transpose", node.name + '_transpose1')
        new_node1.attrs['ndim'] = len(data_shape_list)
        new_node1.attrs['perm'] = new_shape
        new_node1.inputs = [node.inputs[0]]
        new_node1.outputs = [new_entry1]
        add_nodes_list.append(new_node1)

        # add transpose
        new_entry2 = new_entry1.clone()
        new_entry2.name = new_entry2.name + '_transpose'
        graph.add_entry(new_entry2)
        new_entry2.tensor.shape = (new_entry1.tensor.shape[0] + pads_value[pad_axis[0]], new_entry1.tensor.shape[1], new_entry1.tensor.shape[2])
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

        new_node2.inputs = [new_entry2]
        new_node2.outputs = node.outputs
        add_nodes_list.append(new_node2)

        # add iqCat
        new_entry3 = new_entry1.clone()
        new_entry3.name = new_entry3.name + '_history'
        if len(data_shape) == 4:
          new_entry3.tensor.shape = (new_entry1.tensor.shape[0], pads_value[pad_axis[0]], new_entry1.tensor.shape[2], new_entry1.tensor.shape[3])
        else:
          new_entry3.tensor.shape = (pads_value[pad_axis[0]], new_entry1.tensor.shape[1], new_entry1.tensor.shape[2])
        graph.add_entry(new_entry3)
        new_entry3.set2_graph_input()
        graph_inputs.append(new_entry3)
        new_node3 = GraphNode("iqCat", node.name + '_conat')
        if len(data_shape) == 4:
          new_node3.attrs['dim'] = 1
        elif len(data_shape) == 3:
          new_node3.attrs['dim'] = 0
        new_node3.attrs['scale_x_0'] = scale
        new_node3.attrs['scale_x_1'] = scale
        new_node3.attrs['scale_o'] = scale
        new_node3.inputs = [new_entry3, new_entry1]
        new_node3.outputs = [new_entry2]
        add_nodes_list.append(new_node3)

        new_entry1.set2_graph_output()
        graph_outputs.insert(-1, new_entry1)

        if len(node.outputs[0].dst_nodes) != 1:
          return

        del_nodes_list.append(node)
      '''
      elif node.op_type == "ConvTranspose2dInt":
        # import pdb; pdb.set_trace()
        kernels = node.attrs['kernel_shape']
        pads = node.attrs['pads']
        pads_len = len(pads)
        if pads[-2] ==0 and pads[pads_len//2 -2] ==0 :
          continue
        
        strides = node.attrs['strides']
        output_padding = node.attrs['output_padding']
        input_shape = node.inputs[0].tensor.shape
        scale = node.attrs['scale_x']

        # add transpose
        new_entry1 = node.inputs[0].clone()
        new_entry1.name = new_entry1.name + '_transpose'
        # if len(input_shape) == 4:
        new_entry1.tensor.shape = (input_shape[0], input_shape[2], input_shape[1], input_shape[3])
        # elif len(input_shape) == 3:
        #   new_entry1.tensor.shape = (input_shape[1], input_shape[0], input_shape[2])
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
        # if len(input_shape) == 4:
        new_entry4.tensor.shape = (new_entry1.tensor.shape[0], 1, new_entry1.tensor.shape[2], new_entry1.tensor.shape[3])
        # else:
        #   new_entry4.tensor.shape = (1, new_entry1.tensor.shape[1], new_entry1.tensor.shape[2])
        graph.add_entry(new_entry4)
        new_entry4.set2_graph_input()
        graph_inputs.append(new_entry4)
        new_node3 = GraphNode("iqCat", node.name + '_conat')
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

        # if pads[-2] ==0 and pads[pads_len//2 -2] ==0 :

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
          new_entry7.set2_graph_output()
          graph_outputs.insert(-1, new_entry8)

        else:
          new_entry1.set2_graph_output()
          graph_outputs.insert(-1, new_entry1)

        node.attrs['output_padding'] = (0, 3)
        node.attrs['pads'] = (2, 2, 1, 2)
      '''
    for node in add_nodes_list:
      graph.add_node(node)

    for node in del_nodes_list:
      del graph.nodes[node]

    graph.inputs = graph_inputs
    graph.outputs = graph_outputs
    graph.update()

    if is_dump:
      save_to_onnx_model(graph, dump_file_path)
    return graph

__all__ = ["stream_convert"]
