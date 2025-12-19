import numpy as np
from ..xsympy import is_sympy
from ..enum_defines import ALIGN2
from ..save_model import save_to_onnx_model
from ..graph_analysis.ops.utils import calc_expr
from ..graph import Graph, GraphNode, ConstantEntry, Tensor

# List of supported convolution node types
node_type_list = ["Conv1dInt", "Conv2dInt", "ConvTranspose2dInt"]

def op_divide(graph: Graph, is_dump: bool = False) -> Graph:
    """
    Split convolution operations into individual components
    
    Args:
        graph: Input graph to be optimized
        is_dump: Whether to save the optimized graph
        dump_file_path: Path to save the optimized graph
    
    Returns:
        Optimized graph with split operations
    """
    # List to track nodes to add/remove
    new_nodes = []
    nodes_to_remove = []

    # Process each node in the graph
    for node in graph.nodes.values():
        if node.op_type in node_type_list:
            # Get convolution parameters
            weight = node.inputs[1].tensor
            kernel_num = weight.shape[0]
            group = node.attrs["group"]

            if 1 != group and group != kernel_num:
                # Calculate channel split size
                channel_split = kernel_num // group

                # Create split and concat nodes
                split_node = GraphNode("Split", node.name + "_split")
                split_node.inputs = [node.inputs[0]]
                split_node.attrs["axis"] = 1
                split_node.attrs["dims"] = group

                concat_node = GraphNode("iqCat", node.name + "_concat")
                concat_node.inputs = []
                concat_node.attrs["dim"] = 1
                concat_node.outputs = node.outputs

                # Split convolution into groups
                for g in range(group):
                    # Create new input entries
                    new_entry = node.inputs[0].clone()
                    new_entry.name += f"_{g}"
                    new_entry.set_graph_normal()
                    split_node.outputs.append(new_entry)
                    
                    # Adjust tensor shape
                    shape = new_entry.tensor.shape
                    assert shape[1] % group == 0, "group conv/deconv must can to be divided by group with no remainder"
                    new_shape = (shape[0], shape[1] // group, shape[2]) if len(shape) <= 3 else (shape[0], shape[1] // group, shape[2], shape[3])
                    new_entry.tensor.shape = new_shape
                    graph.add_entry(new_entry)
                    
                    # Create split convolution node
                    conv_split = node.clone()
                    conv_split.name = f"{node.name}_{g}"
                    conv_split.inputs[0] = new_entry
                    conv_split.attrs["group"] = 1
                    conv_split.op = None
                    
                    # Process weights and biases
                    for i in range(1, len(node.inputs)):
                        temp_entry = node.inputs[i].clone()
                        temp_entry.name += f"_{g}"
                        if i == 1:
                            temp_entry.data = node.inputs[i].data[g*channel_split:(g+1)*channel_split]
                        elif i == 2:
                            temp_entry.data = node.inputs[i].data[g*channel_split:(g+1)*channel_split]
                        temp_entry.tensor.shape = temp_entry.data.shape
                        conv_split.inputs[i] = temp_entry
                        if temp_entry not in graph.entries:
                            graph.add_entry(temp_entry)

                    # Update outputs
                    conv_split_out = node.outputs[0].clone()
                    conv_split_out.name += f"_{node.name}_{g}"
                    conv_split.outputs[0] = conv_split_out
                    graph.add_entry(conv_split_out)
                    
                    # Add to processing lists
                    new_nodes.extend([conv_split])
                    concat_node.inputs.append(conv_split_out)
                    concat_node.attrs["scale_x_{}".format(g)] = node.attrs["scale_o"]
                concat_node.attrs["scale_o"] = node.attrs["scale_o"]
                for i in range(1, len(node.inputs)):
                    if node.inputs[i] in graph.entries:
                        del graph.entries[node.inputs[i].name]
		
                # Update graph structure
                nodes_to_remove.append(node)
                new_nodes.extend([split_node, concat_node])

    # Update graph with new nodes
    for node in nodes_to_remove:
        del graph.nodes[node.name]
        
    for node in new_nodes:
        graph.add_node(node)

    graph.update()

    # Additional Venus platform optimizations
    if graph.platform == "venus":
        new_nodes = []
        for node in graph.nodes.values():
            if node.op_type == "Conv2dInt":
                # Pad handling logic
                pads_flag = False
                data = node.inputs[0].tensor
                kernels = node.attrs['kernel_shape']
                pads = list(node.attrs['pads'])
                
                # Calculate input dimensions
                x_h = calc_expr(str(data.shape[1]), graph.dynamic_args_max) if is_sympy(data.shape[1]) else data.shape[1]
                x_w = calc_expr(str(data.shape[2]), graph.dynamic_args_max) if is_sympy(data.shape[2]) else data.shape[2]

                # Check if padding is needed
                if (x_w < kernels[-1] or x_h < kernels[-2]):
                    if len(pads) == 2:
                        if (x_w + pads[-1] >= kernels[-1]) and (x_h + pads[-2] >= kernels[-2]):
                            pads_flag = True
                    elif len(pads) == 4:
                        if (x_w + pads[-1] + pads[-3] >= kernels[-1]) and (x_h + pads[-2] + pads[-4] >= kernels[-2]):
                            pads_flag = True
                    else:
                        AssertionError, "do not support this type"
			
                if pads_flag:
                    pad_node = GraphNode("iqPad", f"{node.name}_pads")
                    pad_node.attrs['mode'] = 'constant'
                    
                    # Create and connect new entries
                    new_entry = node.inputs[0].clone()
                    new_entry.name += "_prepads"
                    new_entry.set_graph_normal()
                    graph.add_entry(new_entry)
                    
                    pad_node.inputs.append(node.inputs[0])
                    pad_node.outputs.append(new_entry)
                    if node.inputs[0].src_node == None:
                        break
		    
                    # Adjust pads values
                    pads_value = np.zeros((8,), dtype=np.int64, order="C")
                    if x_w < kernels[-1]:
                        pads_value[-1] = pads[-1]
                        if len(pads) == 2:
                            pads_value[-5] = pads[-1]
                            pads[-1] = 0
                        elif len(pads) == 4:
                            pads_value[-5] = pads[-3]
                            pads[-3] = 0
                            pads[-1] = 0

                    if x_h < kernels[-2]:
                        pads_value[-2] = pads[-2]
                        if len(pads) == 2:
                            pads_value[-6] = pads[-2]
                            pads[-2] = 0
                        elif len(pads) == 4:
                            pads_value[-6] = pads[-4]
                            pads[-2] = 0
                            pads[-4] = 0

                    # Update pads in destination nodes
                    for n in node.inputs[0].src_node.outputs[0].dst_nodes:
                        n.attrs['pads'] = tuple(pads)
                        n.inputs[0] = new_entry

                    # Add pad related entries
                    pads_entry = ConstantEntry(pad_node.name+'_pads', Tensor.from_numpy(pads_value))
                    graph.add_entry(pads_entry)
                    pad_node.inputs.append(pads_entry)
                    
                    data_entry = ConstantEntry(pad_node.name+'_data', Tensor.from_numpy(np.zeros((1,), dtype=np.int64)))
                    graph.add_entry(data_entry)
                    pad_node.inputs.append(data_entry)
                    
                    new_nodes.append(pad_node)

        # Add all new nodes to graph
        for node in new_nodes:
            graph.add_node(node)
        graph.update()

    # Save optimized graph if enabled
    if is_dump:
        save_to_onnx_model(graph, f"./workspace/{graph.name}/model.ignore/5_graph_op_divide.onnx")
        
    return graph