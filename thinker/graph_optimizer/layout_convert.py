from typing import List
from ..enum_defines import Layout
from ..save_model import save_to_onnx_model
from ..graph import Graph, GraphNode, GraphEntry
from ..resource_packer.ops.base import LayoutPerfData
from ..resource_packer.ops.base import create_operator

VARIABLE_TAG = "@"


def _make_layout_name(name: str, layout: Layout) -> str:
    name = name.split(VARIABLE_TAG)[0]
    if layout == Layout.NCHW:
        return name
    elif layout == Layout.NHWC:
        return name + VARIABLE_TAG + "NHWC"
    elif layout == Layout.NCWH:
        return name + VARIABLE_TAG + "NCWH"


def _init_graph(src_graph: Graph) -> List[Graph]:
    graph = Graph()
    graph.copy_attrs(src_graph)

    for entry in src_graph.inputs:
        new_entry = entry.clone()
        graph.add_entry(new_entry)
        graph.inputs.append(new_entry)

    for entry in src_graph.entries.values():
        if entry.is_constant():
            new_entry = entry.clone()
            graph.add_entry(new_entry)
    return [graph]


def _beam_search(graph_list: List[Graph], beam_size: int) -> List[Graph]:
    if len(graph_list) <= beam_size:
        return graph_list
    perfs = [graph.performance for graph in graph_list]
    import heapq

    indexs = heapq.nsmallest(beam_size, range(len(perfs)), perfs.__getitem__)
    dst_graph = [graph_list[index] for index in indexs]
    return dst_graph


def _add_pack_node(
    graph: Graph, src_name: str, src_layout: Layout, dst_name: str, dst_layout: Layout
) -> None:
    src_entry = graph.entries[src_name]
    dst_entry = src_entry.clone()
    dst_entry.name = dst_name
    dst_entry.set2_graph_normal()
    dst_entry.layout = dst_layout
    graph.add_entry(dst_entry)

    pack_node = GraphNode("Packing", dst_name + "_packing")
    pack_node.inputs = [src_entry]
    pack_node.outputs = [dst_entry]
    pack_node.op = create_operator(
        "Packing", {}, [src_entry.tensor], [dst_entry.tensor]
    )
    permute_perfs = pack_node.op.get_layout_perf()
    for permute_perf in permute_perfs:
        if permute_perf.inputs_layout[0] == src_layout:
            graph.performance += permute_perf.performance
            break
    graph.add_node(pack_node)


def _get_entry_by_name(graph: Graph, src_name: str) -> GraphEntry:
    entry = None
    name = src_name.split(VARIABLE_TAG)[0]
    for e in graph.entries.values():
        e_name = e.name.split(VARIABLE_TAG)[0]
        if e_name == name and e.name != src_name:
            entry = e
            break
    if entry == None:
        raise ("not find input")
    return entry


def _add_node_to_graph(graph: Graph, node: GraphNode, perf: LayoutPerfData) -> None:
    new_node = node.clone()
    for i, entry in enumerate(new_node.inputs):
        # not params
        if entry.is_constant():
            continue
        entry.layout = perf.inputs_layout[i]
        entry.name = _make_layout_name(entry.name, entry.layout)

        if entry.name not in graph.entries:
            # sample, need nchw layout,but not exist in graph, then add packing node
            temp_entry = _get_entry_by_name(graph, entry.name)
            _add_pack_node(
                graph, temp_entry.name, temp_entry.layout, entry.name, entry.layout
            )
        new_node.inputs[i] = graph.entries[entry.name]

    for i, entry in enumerate(new_node.outputs):
        entry.layout = perf.outputs_layout[i]
        entry.name = _make_layout_name(entry.name, entry.layout)
        graph.add_entry(entry)
        new_node.outputs[i] = graph.entries[entry.name]

        # output node must be nchw layout, add packing node
        if entry.is_graph_output() and entry.layout != Layout.NCHW:
            entry.set2_graph_normal()
            output_layout = Layout.NCHW
            output_name = _make_layout_name(entry.name, output_layout)
            _add_pack_node(graph, entry.name, entry.layout, output_name, output_layout)
            graph.entries[output_name].set2_graph_output()

    # add node and performance
    graph.add_node(new_node)
    graph.performance += perf.performance


def _packing2transpose(node: GraphNode) -> None:
    in_layout = node.inputs[0].layout
    out_layout = node.outputs[0].layout
    if in_layout == Layout.NCHW and out_layout == Layout.NHWC:
        node.op_type = "Transpose"
        node.attrs["perm"] = [0, 2, 3, 1]
        node.op = create_operator(
            "Transpose", node.attrs, [node.inputs[0].tensor], [node.outputs[0].tensor]
        )

    elif in_layout == Layout.NHWC and out_layout == Layout.NCHW:
        node.op_type = "Transpose"
        node.attrs["perm"] = [0, 3, 1, 2]
        node.op = create_operator(
            "Transpose", node.attrs, [node.inputs[0].tensor], [node.outputs[0].tensor]
        )
    elif (in_layout == Layout.NCHW and out_layout == Layout.NCWH) or (
        in_layout == Layout.NCWH and out_layout == Layout.NCHW
    ):
        node.op_type = "Transpose"
        node.attrs["perm"] = [0, 1, 3, 2]
        node.op = create_operator(
            "Transpose", node.attrs, [node.inputs[0].tensor], [node.outputs[0].tensor]
        )


def _post_process_graph(graph: Graph) -> None:
    for node in graph.nodes.values():
        if node.op_type == "Packing":
            _packing2transpose(node)
        # change shape
        node.op.layout_convert()


# layout optimizer by device
def _layout_convert(src_graph: Graph) -> Graph:
    beam_size = 2
    beam_search_graphs = _init_graph(src_graph)
    for node in src_graph.nodes.values():
        if node.op == None:
            raise ("node op must be not None")
        perf_list = node.op.get_layout_perf()

        search_graphs_expand = []
        for perf in perf_list:
            for graph in beam_search_graphs:
                new_graph = graph.clone()
                _add_node_to_graph(new_graph, node, perf)
                search_graphs_expand.append(new_graph)
        beam_search_graphs = _beam_search(search_graphs_expand, beam_size)

    for graph in beam_search_graphs:
        for output in src_graph.outputs:
            graph.entries[output.name].set2_graph_output()
            graph.outputs.append(graph.entries[output.name])
    best_graph = _beam_search(beam_search_graphs, 1)[0]
    best_graph.update()
    _post_process_graph(best_graph)
    return best_graph


def layout_optimizer(
    graph: Graph,
    is_dump: bool = True,
    dump_file_path: str = "./model.ignore/graph_layout_convert.onnx",
) -> Graph:
    """
    layout转换策略：在conv层的输入数据大于64KB，且W>H时，交换W和H的维度
    """
    new_graph = _layout_convert(graph)
    if is_dump:
        save_to_onnx_model(new_graph, dump_file_path)
    return new_graph


__all__ = ["layout_optimizer"]
