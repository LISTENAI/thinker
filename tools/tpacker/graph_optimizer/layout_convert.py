from typing import List
from ..enum_defines import Layout
from ..save_model import save_to_onnx_model
from ..graph import Graph, GraphNode, GraphEntry
from ..graph_analysis.ops.base import LayoutPerfData, create_operator

VARIABLE_TAG = "@"

def _make_layout_name(name: str, layout: Layout) -> str:
    """Generate layout-specific name for entries"""
    name = name.split(VARIABLE_TAG)[0]
    if layout == Layout.NCHW:
        return name
    return f"{name}{VARIABLE_TAG}{layout.name}"

def _init_graph(src_graph: Graph) -> List[Graph]:
    """Initialize a new graph with attributes and entries from source graph"""
    graph = Graph()
    graph.copy_attrs(src_graph)
    
    # Add input entries
    for entry in src_graph.inputs:
        new_entry = entry.clone()
        graph.add_entry(new_entry)
        graph.inputs.append(new_entry)
    
    # Add constant entries
    for entry in src_graph.entries.values():
        if entry.is_constant():
            new_entry = entry.clone()
            graph.add_entry(new_entry)
    
    return [graph]

def _beam_search(graph_list: List[Graph], beam_size: int) -> List[Graph]:
    """Select top 'beam_size' graphs with best performance"""
    if len(graph_list) <= beam_size:
        return graph_list
    
    import heapq
    perfs = [graph.performance for graph in graph_list]
    indices = heapq.nsmallest(beam_size, range(len(perfs)), key=perfs.__getitem__)
    return [graph_list[i] for i in indices]

def _add_pack_node(graph: Graph, src_name: str, src_layout: Layout, dst_name: str, dst_layout: Layout) -> None:
    """Add packing node to convert tensor layout"""
    src_entry = graph.entries[src_name]
    dst_entry = src_entry.clone()
    dst_entry.name = dst_name
    dst_entry.layout = dst_layout
    graph.add_entry(dst_entry)
    
    pack_node = GraphNode("Packing", f"{dst_name}_packing")
    pack_node.inputs = [src_entry]
    pack_node.outputs = [dst_entry]
    pack_node.op = create_operator("Packing", {}, [src_entry.tensor], [dst_entry.tensor])
    
    # Update performance based on layout conversion
    perfs = pack_node.op.get_layout_perf(graph.dynamic_shape)
    for perf in perfs:
        if perf.inputs_layout[0] == src_layout:
            graph.performance += perf.performance
            break
    
    graph.add_node(pack_node)

def _get_entry_by_name(graph: Graph, src_name: str) -> GraphEntry:
    """Find entry in graph by name"""
    name = src_name.split(VARIABLE_TAG)[0]
    for e in graph.entries.values():
        e_name = e.name.split(VARIABLE_TAG)[0]
        if e_name == name and e.name != src_name:
            return e
    raise ValueError(f"Entry {src_name} not found")

def _add_node_to_graph(graph: Graph, node: GraphNode, perf: LayoutPerfData) -> None:
    """Add node to graph with specified layout and performance"""
    new_node = node.clone()
    
    # Update input entries
    for i, entry in enumerate(new_node.inputs):
        if not entry.is_constant():
            entry.layout = perf.inputs_layout[i]
            entry.name = _make_layout_name(entry.name, entry.layout)
            
            if entry.name not in graph.entries:
                # Add packing node if layout not exists
                temp_entry = _get_entry_by_name(graph, entry.name)
                _add_pack_node(graph, temp_entry.name, temp_entry.layout, entry.name, entry.layout)
            new_node.inputs[i] = graph.entries[entry.name]
    
    # Update output entries
    for i, entry in enumerate(new_node.outputs):
        entry.layout = perf.outputs_layout[i]
        entry.name = _make_layout_name(entry.name, entry.layout)
        graph.add_entry(entry)
        new_node.outputs[i] = graph.entries[entry.name]
        
        # Ensure output is NCHW layout
        if entry.is_graph_output() and entry.layout != Layout.NCHW:
            output_layout = Layout.NCHW
            output_name = _make_layout_name(entry.name, output_layout)
            _add_pack_node(graph, entry.name, entry.layout, output_name, output_layout)
            graph.entries[output_name].set_graph_output()
    
    # Add node and update performance
    graph.add_node(new_node)
    graph.performance += perf.performance

def _packing2transpose(node: GraphNode) -> None:
    """Convert Packing node to Transpose node based on layout"""
    in_layout = node.inputs[0].layout
    out_layout = node.outputs[0].layout
    
    if in_layout == Layout.NCHW and out_layout == Layout.NHWC:
        node.op_type = "Transpose"
        node.attrs["perm"] = [0, 2, 3, 1]
    elif in_layout == Layout.NHWC and out_layout == Layout.NCHW:
        node.op_type = "Transpose"
        node.attrs["perm"] = [0, 3, 1, 2]
    elif (in_layout == Layout.NCHW and out_layout == Layout.NCWH) or \
         (in_layout == Layout.NCWH and out_layout == Layout.NCHW):
        node.op_type = "Transpose"
        node.attrs["perm"] = [0, 1, 3, 2]
    
    node.op = create_operator(node.op_type, node.attrs, 
                             [node.inputs[0].tensor], [node.outputs[0].tensor])

def _post_process_graph(graph: Graph) -> None:
    """Post-process graph after layout conversion"""
    for node in graph.nodes.values():
        if node.op_type == "Packing":
            _packing2transpose(node)
        node.op.layout_convert(node.op_type)
        
        if node.op_type in {"Conv2dInt", "Conv1dInt", "ConvTranspose2dInt", "MaxPool", "AvgPool2dInt"}:
            for key in node.attrs.keys():
                node.attrs[key] = node.op.attrs.get(key)

def _layout_convert(src_graph: Graph) -> Graph:
    """Main layout conversion function using beam search"""
    beam_size = 2
    beam_search_graphs = _init_graph(src_graph)
    
    for node in src_graph.nodes.values():
        if not node.op:
            raise ValueError("Node operation must not be None")
        perf_list = node.op.get_layout_perf(src_graph.dynamic_args_max)
        
        search_graphs_expand = []
        for perf in perf_list:
            for graph in beam_search_graphs:
                new_graph = graph.clone()
                _add_node_to_graph(new_graph, node, perf)
                search_graphs_expand.append(new_graph)
        
        beam_search_graphs = _beam_search(search_graphs_expand, beam_size)
    
    # Update outputs
    for graph in beam_search_graphs:
        for output in src_graph.outputs:
            graph.entries[output.name].set_graph_output()
            graph.outputs.append(graph.entries[output.name])
    
    best_graph = _beam_search(beam_search_graphs, 1)[0]
    best_graph.update()
    _post_process_graph(best_graph)
    return best_graph

def layout_optimizer(graph: Graph, is_dump: bool = True) -> Graph:
    """
    Optimize graph layout for better performance
    
    Args:
        graph: Input graph to be optimized
        is_dump: Whether to save the optimized graph
        dump_file_path: Path to save the optimized graph
    
    Returns:
        Optimized graph with best layout configuration
    """
    new_graph = _layout_convert(graph)
    if is_dump:
        save_to_onnx_model(new_graph, f"./workspace/{graph.name}/model.ignore/6_graph_layout_convert.onnx")
    return new_graph

__all__ = ["layout_optimizer"]