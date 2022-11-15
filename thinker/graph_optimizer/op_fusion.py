from ..graph import Graph
from .simple_fusion import MethodManager
from ..save_model import save_to_onnx_model


def op_fusion(
    graph: Graph,
    strategy: str,
    is_dump: bool = True,
    dump_file_path: str = "./model.ignore/graph_op_fusion.onnx",
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
        保存路径，默认保存到./graph_sim_fusion.onnx
    Returns:
      g: Graph
        融合后的graph
    """
    ignore_methods = list()
    if strategy != None:
        ignore_methods = [strategy]

    graph_after_fusion = MethodManager.apply(graph, ignore_methods)

    if is_dump:
        save_to_onnx_model(graph_after_fusion, dump_file_path)
    return graph_after_fusion


__all__ = ["op_fusion"]
