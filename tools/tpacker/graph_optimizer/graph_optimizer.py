from ..graph import Graph
from ..enum_defines import Colors, ModelConfig
from .op_fusion import op_fusion
from .op_divide import op_divide
from .op_replace import stream_convert
from .layout_convert import layout_optimizer


def optimize_graph(graph: Graph, model_config: ModelConfig, is_dump: bool = False) -> Graph:
    if model_config.isstream:
        graph = stream_convert(graph, model_config.isstream, is_dump)
        graph.init_tensor()
        print(f"{Colors.GREEN}3.1 stream convert passed{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}3.1 stream convert skipped{Colors.RESET}")

    graph = op_fusion(graph, model_config.strategy, is_dump)
    graph.init_tensor()
    print(f"{Colors.GREEN}3.2 op fusion passed{Colors.RESET}")

    graph = op_divide(graph, is_dump)
    graph.init_tensor()
    print(f"{Colors.GREEN}3.3 op divide passed{Colors.RESET}")

    graph = layout_optimizer(graph, is_dump)
    graph.init_tensor()
    print(f"{Colors.GREEN}3.4 layout convert passed{Colors.RESET}")
    return graph


__all__ = ["optimize_graph"]
