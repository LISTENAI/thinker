from typing import List

from ..graph import Graph
from .op_fusion import op_fusion
from .layout_convert import layout_optimizer


def graph_optimizer(graph: Graph, strategy: str, is_dump: bool = False) -> Graph:
    graph = op_fusion(graph, strategy, is_dump)
    graph.init_tensor()
    print("---- op fusion success ----")

    graph = layout_optimizer(graph, is_dump)
    graph.init_tensor()
    print("---- convert layout success ----")
    return graph


__all__ = ["graph_optimizer"]
