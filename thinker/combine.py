# Copyright (C) 2022 listenai Co.Ltd
# All rights reserved. 
# Created by leifang on 2022.09.31
# modifyed by leifang on 2023.08.11

import numpy as np
from typing import List, Dict, Tuple

from .graph import Graph, GraphEntry
from .graph_optimizer import op_split
from .enum_defines import MemType, DevType, Platform
from .resource_packer.memory import memory_plan, get_memory_size

def parse_memory(s:str) -> Dict[str, Tuple[int]]:
    dynamic_memory = dict()
    if s is None:
        return dynamic_memory
    s1 = s.split(",")
    for i in range(len(s1)):
      s2 = s1[i].split(":")
      if len(s2)==1:
        assert s1[0].lower() in ['psram', 'share-memory', 'flash']
        return {'params':s1[0]}
      assert s2[1].lower() in ['psram', 'share-memory', 'flash']
      dynamic_memory[s2[0]] = s2[1]
    return dynamic_memory

def _get_next_id(threshold: List[int]) -> int:
    begin = len(threshold)
    end = 0
    for i in range(len(threshold)):
        if threshold[i] == 1:
            begin = i
        elif threshold[i] == -1:
            end = i
            break
    assert begin < end
    return (end + begin) // 2


def _judge_end(threshold: List[int]) -> int:
    for i in range(len(threshold) - 1):
        if threshold[i] == 1 and threshold[i + 1] == -1:
            return i
    return None


def _get_memory_plan(
    graph: Graph, platform: Platform, local_mem: str, threshold: int, is_dump: bool
) -> Tuple[Graph, Dict[int, List[int]], bool]:
    new_graph, is_linearint = op_split(graph, is_dump, threshold=threshold)
    new_graph.init_tensor()
    new_graph.update()
    new_graph = _graph_bind_device(new_graph, platform, local_mem)
    new_graph.pack_params()
    memory_planer = memory_plan(new_graph, True)
    return new_graph, memory_planer, is_linearint


def _graph_bind_device(graph: Graph, platform: Platform, memory: str) -> Graph:
    cpus = Platform.get_cpu_list(platform)
    dy_memory = parse_memory(memory)    
    if len(cpus) > 1:
        cpus = cpus[::-1]

    for node in graph.nodes.values():
        flag = 0
        if node.dev_type == None:
            for i in range(len(cpus)):
                if node.op_type in Platform.get_support_op(platform):
                    node.dev_type = cpus[i]
                    flag = 1
                    break
            if flag == 0:
                raise ImportError(f"platform:{platform} do not support {node.op_type}!")

        for i in range(len(node.inputs)):
            if node.inputs[i].is_constant():
                if 'params' in dy_memory:
                    node.inputs[i].tensor.mem_type = MemType.from_str(dy_memory['params'])
                else:
                    node.inputs[i].tensor.mem_type = MemType.PSRAM
            elif node.inputs[i] in dy_memory:
                node.inputs[i].tensor.mem_type = MemType.from_str(dy_memory[node.inputs[i].name])
            elif node.inputs[i].tensor.mem_type == None:
                node.inputs[i].tensor.mem_type = MemType.SHARE_MEM

        for i in range(len(node.outputs)):
            if node.outputs[i].name in dy_memory:
                node.outputs[i].tensor.mem_type = MemType.from_str(dy_memory[node.outputs[i].name])
            elif node.outputs[i].tensor.mem_type == None:
                node.outputs[i].tensor.mem_type = MemType.SHARE_MEM

    return graph


def graph_adapter(
    graph: Graph, platform: Platform, local_mem: str, is_dump: bool
) -> Tuple[Graph, Dict[int, List[int]]]:
    MIN_THRESHOLD = 16 * 1024
    MAX_THRESHOLD = 640 * 1024
    STEP_THRESHOLD = 16 * 1024
    threshold = np.arange(MIN_THRESHOLD, MAX_THRESHOLD + 1, STEP_THRESHOLD)
    threshold_mask = np.zeros(len(threshold))
    print("set linearint threshold and analyze memory begin")
    print("try threshold:{}".format(threshold[-1]))

    new_graph, memory_planer, is_linearint = _get_memory_plan(
        graph, platform, local_mem, threshold[-1], is_dump
    )
    memory_tobe_allocated = get_memory_size(memory_planer, MemType.SHARE_MEM)
    
    if memory_tobe_allocated > MAX_THRESHOLD:
        threshold_mask[-1] = -1
        print("try threshold:{}".format(threshold[0]))
        new_graph, memory_planer, is_linearint = _get_memory_plan(
            graph, platform, local_mem, threshold[0], is_dump
        )
        memory_tobe_allocated = get_memory_size(memory_planer, MemType.SHARE_MEM)

        if memory_tobe_allocated > MAX_THRESHOLD:
            print(
                "WARNING:SHARE-MEM to be allocated was {}, exceed {}".format(
                    memory_tobe_allocated, MAX_THRESHOLD
                )
            )
        else:
            threshold_mask[0] = 1
            while is_linearint:
                id = _get_next_id(threshold_mask)
                print("try threshold:{}".format(threshold[id]))
                new_graph, memory_planer, is_linearint = _get_memory_plan(
                    graph, platform, local_mem, threshold[id], is_dump
                )
                memory_tobe_allocated = get_memory_size(
                    memory_planer, MemType.SHARE_MEM
                )

                if memory_tobe_allocated > MAX_THRESHOLD:
                    threshold_mask[id] = -1
                else:
                    threshold_mask[id] = 1
                final_id = _judge_end(threshold_mask)
                if final_id != None:
                    print("the best threshold：{}".format(threshold[final_id]))
                    new_graph, memory_planer, is_linearint = _get_memory_plan(
                        graph, platform, local_mem, threshold[final_id], is_dump
                    )
                    break

    print("set linearint threshold and analyze memory end")

    return new_graph, memory_planer


__all__ = ["graph_adapter"]
