# Copyright (C) 2025 listenai Co.Ltd
# All rights reserved. 
# Created by leifang on 2022.09.31

import numpy as np
from typing import List, Dict, Tuple

from ..devices import Device
from .ops.utils import calc_expr
from ..xsympy import is_sympy
from ..graph import Graph, GraphEntry
from ..graph_optimizer import op_split
from ..enum_defines import MemType, DevType, Colors, MemoryConfig
from .memory import memory_plan, get_memory_size


def _parse_memory(s:str) -> Dict[str, Tuple[int]]:
    dynamic_memory = dict()
    if s is None:
        return dynamic_memory
    s1 = s.split(",")
    for i in range(len(s1)):
      s2 = s1[i].split(":")
      if len(s2)==1:
        assert s1[0].lower() in {'psram', 'share-mem', 'flash'}, f"{s1[i]}"
      assert s2[1].lower() in {'psram', 'share-mem', 'flash'}
      dynamic_memory[s2[0]] = s2[1]
    return dynamic_memory

def _get_next_id(threshold: List[int]) -> int:
    begin = len(threshold)
    end = 0
    for i in range(len(threshold) - 1):
        if (threshold[i] != 0) and (threshold[i + 1] == 0):
            begin = i
        elif (threshold[i] == 0) and (threshold[i + 1] != 0):
            end = i + 1
            break
    if begin >= end:
        return None
    else:
        return (end + begin) // 2

def _judge_end(threshold: List[int]) -> int:
    for i in range(len(threshold) - 1):
        if threshold[i] == 1 and threshold[i + 1] == -1:
            return i
    return None

def _get_memory_plan(graph: Graph, device: Device, memory_config: MemoryConfig, is_dump: bool) \
    -> Tuple[Graph, Dict[int, List[int]], bool]:
    new_graph, is_linearint = op_split(graph, False, is_dump, memory_config.threshold1, memory_config.threshold2, memory_config.threshold3)
    new_graph.init_tensor()
    new_graph.update()
    new_graph = _graph_bind_device(new_graph, device, memory_config.storage_location, memory_config.threshold4)
    new_graph.acquire_dynamic_shape()
    new_graph.pack_params()
    memory_planer = memory_plan(new_graph, memory_config.dma_prefetch)
    return new_graph, memory_planer, is_linearint

def _graph_bind_device(graph: Graph, device: Device, dy_memory: Dict[str, Tuple[int]], threshold: int) -> Graph:

    for node in graph.nodes.values():
        if node.op_type not in device.supported_operators:
            raise ImportError(f"platform:{platform} do not support {node.op_type}!")

        for i in range(len(node.inputs)):
            data_size = np.prod(node.inputs[i].tensor.shape)
            if is_sympy(data_size):
                data_size = calc_expr(str(data_size), graph.dynamic_args_max)
            if node.inputs[i].is_constant():
                if 'params' in dy_memory:
                    node.inputs[i].tensor.mem_type = MemType.from_str(MemType, dy_memory['params'])
                else:
                    node.inputs[i].tensor.mem_type = MemType.PSRAM
            elif node.inputs[i].name in dy_memory:
                node.inputs[i].tensor.mem_type = MemType.from_str(MemType, dy_memory[node.inputs[i].name])
            elif data_size > threshold:
                node.inputs[i].tensor.mem_type = MemType.PSRAM
            elif node.inputs[i].tensor.mem_type == None:
                node.inputs[i].tensor.mem_type = MemType.SHARE_MEM

        for i in range(len(node.outputs)):
            data_size = np.prod(node.outputs[i].tensor.shape)
            if is_sympy(data_size):
                data_size = calc_expr(str(data_size), graph.dynamic_args_max)
            if node.outputs[i].name in dy_memory:
                node.outputs[i].tensor.mem_type = MemType.from_str(MemType, dy_memory[node.outputs[i].name])
            elif data_size > threshold:
                node.outputs[i].tensor.mem_type = MemType.PSRAM
            elif node.outputs[i].tensor.mem_type == None:
                node.outputs[i].tensor.mem_type = MemType.SHARE_MEM

    return graph

def adapt_graph_to_hardware(graph: Graph, device: Device, memory_config: MemoryConfig, is_dump: bool) \
    -> Tuple[Graph, Dict[int, List[int]]]:

    new_graph, memory_planer, is_linearint = _get_memory_plan(graph, device, memory_config, is_dump)
    memory_tobe_allocated = get_memory_size(memory_planer, MemType.SHARE_MEM)
    print(f"{Colors.GREEN}5.1 adapt the computation graph to hardware passed{Colors.RESET}")
    print(f"{Colors.GREEN}5.2 Memory pre-allocation passed{Colors.RESET}")
    return new_graph, memory_planer


__all__ = ["adapt_graph_to_hardware"]
