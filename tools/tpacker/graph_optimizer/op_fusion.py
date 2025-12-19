from typing import List, Tuple

from ..graph import Graph
from .method_manager import MethodManager
from ..save_model import save_to_onnx_model


def op_fusion(graph: Graph, strategy: List[str], is_dump: bool = True) -> Graph:
    """
    Args:
      graph: Graph
        待融合graph
      ignore_methods: List[str]
        忽略特定算子融合;
      is_dump: bool
        是否保存融合后的graph，默认保存，即保存；
    Returns:
      g: Graph
        融合后的graph
    Function Description:
      计算图算子消融，包括去除slice算子、CR融合、去除quant和requant算子、transpose转换为reshape和reshape合并
    """

    ignore_methods = []
    if strategy != None:
        ignore_methods = strategy
    graph_after_fusion = MethodManager.apply(graph, ignore_methods)

    if is_dump:
        save_to_onnx_model(graph_after_fusion, f"./workspace/{graph.name}/model.ignore/4_graph_op_fusion.onnx")
    return graph_after_fusion


__all__ = ["op_fusion"]
