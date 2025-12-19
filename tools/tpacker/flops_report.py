# Copyright (C) 2025 listenai Co.Ltd
# All rights reserved.
# Created by bzcai on 2022.11.24

from .graph import Graph
from .enum_defines import Colors

def statistical_calculation_amount(graph: Graph) -> None:
    """Calculate and print the total number of operations (FLOPS) in the model graph.
    
    Args:
        graph (Graph): The computational graph of the model.
    """
    dynamic_args_max = graph.dynamic_args_max or {}
    total_ops = 0
    
    for node in graph.nodes.values():
        node_ops = node.op.flops_counter(dynamic_args_max)
        total_ops += node_ops
    
    print(f"{Colors.GREEN}7.1 statistical calculation amount passed{Colors.RESET}")
    print(f"{Colors.CYAN}  total ops of model:{total_ops}{Colors.RESET}")

__all__ = ["statistical_calculation_amount"]