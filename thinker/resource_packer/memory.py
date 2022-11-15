import numpy as np
from typing import Dict, List

from ..graph import Tensor
from ..graph import Graph, GraphEntry
from ..enum_defines import DevType, MemType

WORKSPACE_NAME = "workspace"
DMA_BUFFER1_NAME = "dma_buffer1"
DMA_BUFFER2_NAME = "dma_buffer2"


def sub_list(list1, list2):
    list = []
    for e in list1:
        if e not in list2:
            list.append(e)
    return list


class EntryContext(object):
    def __init__(self, entry, life_begin=0, life_end=1000000000):
        self.entry = entry
        self.life_begin = life_begin
        self.life_end = life_end
        self.mem_id = -1
        self.share_id = -1
        self.nbytes = 0


class NodeContext(object):
    def __init__(self, node):
        self.node = node
        self.alive_tensors = list()
        self.total_size = 0


class MemoryPlaner(object):
    def __init__(self, graph, is_dump):
        self.node_ctx_list = [NodeContext(x) for x in graph.nodes.values()]
        self.entry_ctx_list = [EntryContext(e) for e in graph.entries.values()]
        self.entries = graph.entries
        # add workspace
        self.get_workspace(graph, is_dump)
        self.get_dma_buffer(graph, is_dump)

        for x in self.entry_ctx_list:
            x.nbytes = x.entry.tensor.nbytes

        self.update_share_id()
        self.get_life_period()
        self.plan_memory()
        self.update_share_mem_id()

    def get_workspace(self, graph, is_dump):
        max_workspace = {}
        new_workspace = {}
        life_begin = len(graph.nodes.values())
        life_end = 0
        for node in graph.nodes.values():
            workspace = node.op.get_workspace(node.dev_type)
            if len(workspace) == 0:
                continue
            if life_begin > node.index:
                life_begin = node.index
            if life_end < node.index:
                life_end = node.index

            tensor = workspace[0]
            if node.dev_type not in max_workspace:
                max_workspace[node.dev_type] = tensor.nbytes
            else:
                if max_workspace[node.dev_type] < tensor.nbytes:
                    max_workspace[node.dev_type] = tensor.nbytes

            if node.dev_type == DevType.HIFI:
                flag = 0
                for input in node.inputs:
                    if input.tensor.mem_type == MemType.PSRAM:
                        continue
                    else:
                        flag = 1
                        break
                if flag:
                    new_workspace[MemType.SHARE_MEM] = max_workspace[node.dev_type]
                else:
                    new_workspace[MemType.PSRAM] = max_workspace[node.dev_type]
            else:
                new_workspace[MemType.SHARE_MEM] = max_workspace[node.dev_type]

        for k, v in new_workspace.items():
            entry = GraphEntry(WORKSPACE_NAME, Tensor.from_shape([v], np.int8, k))
            entry.tensor.mem_type = k
            entry.index = len(self.entry_ctx_list)
            ctx = EntryContext(entry)
            ctx.life_begin = life_begin
            ctx.life_end = life_end
            self.entry_ctx_list.append(ctx)

    def get_dma_buffer(self, graph, is_dump):
        max_dma_buffer1 = {}
        max_dma_buffer2 = {}
        index = 0
        for node in graph.nodes.values():
            dma_size = 0
            if (
                node.op_type == "Conv2dInt"
                or node.op_type == "ConvTranspose2dInt"
                or node.op_type == "LinearInt"
                or node.op_type == "LSTMInt"
                or node.op_type == "Conv1dInt"
            ) and node.dev_type == DevType.LUNA:
                for x in node.inputs:
                    if x.tensor.mem_type != MemType.SHARE_MEM and x.is_constant():
                        dma_size += x.tensor.nbytes
                if dma_size != 0:
                    # max workspace
                    if index % 2 == 0:
                        if MemType.SHARE_MEM not in max_dma_buffer1:
                            max_dma_buffer1[MemType.SHARE_MEM] = dma_size
                        elif max_dma_buffer1[MemType.SHARE_MEM] < dma_size:
                            max_dma_buffer1[MemType.SHARE_MEM] = dma_size
                    else:
                        if MemType.SHARE_MEM not in max_dma_buffer2:
                            max_dma_buffer2[MemType.SHARE_MEM] = dma_size
                        elif max_dma_buffer2[MemType.SHARE_MEM] < dma_size:
                            max_dma_buffer2[MemType.SHARE_MEM] = dma_size
                    index += 1

        for k, v in max_dma_buffer1.items():
            entry = GraphEntry(DMA_BUFFER1_NAME, Tensor.from_shape([v], np.int8, k))
            entry.tensor.mem_type = k
            entry.index = len(self.entry_ctx_list)
            ctx = EntryContext(entry)
            self.entry_ctx_list.append(ctx)

        for k, v in max_dma_buffer2.items():
            entry = GraphEntry(DMA_BUFFER2_NAME, Tensor.from_shape([v], np.int8, k))
            entry.tensor.mem_type = k
            entry.index = len(self.entry_ctx_list)
            ctx = EntryContext(entry)
            self.entry_ctx_list.append(ctx)

    def update_share_id(self):
        for s in self.entry_ctx_list:
            s.share_id = s.entry.index
            node = s.entry.src_node
            if node and node.op.is_inplace():  # reshape, squeeze, unsqueeze
                input_name = node.inputs[0]
                src_tensor_id = self.entries.keys().index(input_name)
                s.share_id = src_tensor_id

    def get_life_period(self):
        for _, s in enumerate(self.entry_ctx_list):
            # params
            if s.entry.is_constant():
                s.life_begin = 0
                s.life_end = 100000000000
            # input
            elif s.entry.is_graph_input():
                s.life_begin = 0
                s.life_end = max([x.index for x in s.entry.dst_nodes])
            # outputs
            elif s.entry.is_graph_output():
                s.life_begin = s.entry.src_node.index
                s.life_end = len(self.node_ctx_list) - 1
            elif s.entry.name == WORKSPACE_NAME:
                pass
            elif s.entry.name == DMA_BUFFER1_NAME or s.entry.name == DMA_BUFFER2_NAME:
                s.life_begin = 0
                s.life_end = len(self.node_ctx_list) - 1
            else:
                s.life_begin = s.entry.src_node.index
                if len(s.entry.dst_nodes) == 0:  # unused tensor
                    s.life_end = s.life_begin
                else:
                    s.life_end = max([x.index for x in s.entry.dst_nodes])

        # update inplace alive status
        for i, _t in enumerate(self.entry_ctx_list):
            _t_prev = _t
            if (
                _t_prev.share_id != _t_prev.entry.index
                and _t_prev.entry.tensor.mem_type
                == self.entry_ctx_list[_t_prev.share_id].entry.tensor.mem_type
            ):  # shared from other tensor
                _t_prev = self.entry_ctx_list[_t_prev.share_id]

                # update life_period
                _t_prev.life_begin = _t.life_begin = min(
                    _t.life_begin, _t_prev.life_begin
                )
                _t_prev.life_end = _t.life_end = max(_t.life_end, _t_prev.life_end)

                _t.share_id = _t_prev.share_id

        for i, _t in enumerate(self.entry_ctx_list):
            if _t.life_end > len(self.node_ctx_list):
                continue
            if _t.share_id != _t.entry.index:
                continue
            for j in range(_t.life_begin, _t.life_end + 1):
                if _t not in self.node_ctx_list[j].alive_tensors:
                    self.node_ctx_list[j].alive_tensors.append(_t)

    def update_share_mem_id(self):
        for i, tensor in enumerate(self.entry_ctx_list):
            share_id = tensor.share_id
            if share_id != i:
                tensor.mem_id = self.entry_ctx_list[share_id].mem_id

    def plan_memory(self):
        raise NotImplementedError


class MemoryGreedyBySize(MemoryPlaner):
    def plan_memory(self):
        self.mem_sizes = {}
        node_list = self.node_ctx_list
        memory_list = []
        for entry in self.entry_ctx_list:
            if entry.entry.tensor.mem_type not in memory_list:
                memory_list.append(entry.entry.tensor.mem_type)

        for i in range(len(memory_list)):
            tensor_list = []
            for entry in self.entry_ctx_list:
                if entry.entry.tensor.mem_type == memory_list[i]:
                    tensor_list.append(entry)
            tensor_list_by_size = sorted(
                tensor_list, key=lambda x: x.nbytes, reverse=True
            )

            mem_sizes = []
            mem_list = dict()
            for _, _t in enumerate(tensor_list_by_size):
                if _t.life_end > len(node_list):
                    continue
                best_fit = 1000000000
                for k, life in mem_list.items():
                    if (
                        _t.nbytes <= mem_sizes[k]
                        and mem_sizes[k] - _t.nbytes < best_fit
                    ):
                        fit_flag = False
                        for j in range(len(life)):
                            if _t.life_begin > life[j][1] or _t.life_end < life[j][0]:
                                fit_flag = True
                                continue
                            else:
                                fit_flag = False
                                break
                        if fit_flag == True:
                            best_fit = mem_sizes[k] - _t.nbytes
                            _t.mem_id = k

                if best_fit == 1000000000:
                    _t.mem_id = len(mem_sizes)
                    mem_sizes.append(_t.nbytes)
                    mem_list[_t.mem_id] = [(_t.life_begin, _t.life_end)]
                else:
                    mem_list[_t.mem_id].append((_t.life_begin, _t.life_end))
            if len(mem_sizes) != 0:
                # print("greedy by size on {}:{},{},{}".format(memory_list[i].name, len(mem_sizes),mem_sizes,sum(mem_sizes)))
                self.mem_sizes[memory_list[i].value] = mem_sizes


class MemoryGreedyByBreadth(MemoryPlaner):
    def plan_memory(self):
        node_list = self.node_ctx_list
        self.mem_sizes = {}

        memory_list = []
        for entry in self.entry_ctx_list:
            if entry.entry.tensor.mem_type not in memory_list:
                memory_list.append(entry.entry.tensor.mem_type)

        for i in range(len(memory_list)):
            for j, node in enumerate(node_list):
                node.total_size = 0
                for _, tensor in enumerate(node.alive_tensors):
                    if tensor.entry.tensor.mem_type == memory_list[i]:
                        node.total_size += tensor.nbytes

            node_list_by_breadth = sorted(
                node_list, key=lambda x: x.total_size, reverse=True
            )

            tensor_list_by_breadth = []
            new_alive_tensor_list = []
            for _, node in enumerate(node_list_by_breadth):
                if node.total_size != 0:
                    for _t in node.alive_tensors:
                        if _t.entry.tensor.mem_type == memory_list[i]:
                            new_alive_tensor_list.append(_t)
                    alive_tensor_list = sorted(
                        new_alive_tensor_list, key=lambda x: x.nbytes, reverse=True
                    )

                    for _t in alive_tensor_list:
                        if _t not in tensor_list_by_breadth:
                            tensor_list_by_breadth.append(_t)

            mem_sizes = []
            mem_list = dict()
            for _, _t in enumerate(tensor_list_by_breadth):
                if _t.life_end > len(node_list):
                    continue
                best_fit = 1000000000
                for k, life in mem_list.items():
                    if (
                        _t.nbytes <= mem_sizes[k]
                        and mem_sizes[k] - _t.nbytes < best_fit
                    ):
                        fit_flag = False
                        for j in range(len(life)):
                            if _t.life_begin > life[j][1] or _t.life_end < life[j][0]:
                                fit_flag = True
                                continue
                            else:
                                fit_flag = False
                                break
                        if fit_flag == True:
                            best_fit = mem_sizes[k] - _t.nbytes
                            _t.mem_id = k

                if best_fit == 1000000000:
                    _t.mem_id = len(mem_sizes)
                    mem_sizes.append(_t.nbytes)
                    mem_list[_t.mem_id] = [(_t.life_begin, _t.life_end)]
                else:
                    mem_list[_t.mem_id].append((_t.life_begin, _t.life_end))
            if len(mem_sizes) != 0:
                # print("greedy by breadth on {}:{},{},{}".format(memory_list[i].name, len(mem_sizes),mem_sizes,sum(mem_sizes)))
                self.mem_sizes[memory_list[i].value] = mem_sizes


class MemoryGreedyByOrder(MemoryPlaner):
    def plan_memory(self):
        node_list = self.node_ctx_list
        self.mem_sizes = {}
        memory_list = []
        for entry in self.entry_ctx_list:
            if entry.entry.tensor.mem_type not in memory_list:
                memory_list.append(entry.entry.tensor.mem_type)

        for d in range(len(memory_list)):
            mem_sizes = []
            idle_mem = []
            prev_alive = list()

            for i in range(len(node_list)):
                cur_alive = list()
                for j in range(len(node_list[i].alive_tensors)):
                    if (
                        node_list[i].alive_tensors[j].entry.tensor.mem_type
                        == memory_list[d]
                    ):
                        cur_alive.append(node_list[i].alive_tensors[j])
                new_idle = sub_list(prev_alive, cur_alive)
                for tensor in new_idle:
                    m_id = tensor.mem_id
                    m_sz = mem_sizes[m_id]
                    idle_mem.append((m_sz, m_id))
                idle_mem = sorted(idle_mem)

                new_use = sub_list(cur_alive, prev_alive)
                for tensor in new_use:
                    m_sz = tensor.nbytes
                    for x in range(len(idle_mem)):
                        if idle_mem[x][0] >= m_sz:
                            m_id = idle_mem[x][1]
                            mem_sizes[m_id] = max(m_sz, mem_sizes[m_id])  # realloc
                            tensor.mem_id = m_id
                            idle_mem.pop(x)
                            break
                    else:
                        # malloc when no idle memory
                        tensor.mem_id = len(mem_sizes)
                        mem_sizes.append(m_sz)

                prev_alive = cur_alive

            if len(mem_sizes) != 0:
                # print("greedy by order on {}:{},{},{}".format(memory_list[d].name, len(mem_sizes),mem_sizes,sum(mem_sizes)))
                self.mem_sizes[memory_list[d].value] = mem_sizes


def memory_plan(graph: Graph, is_dump: bool = True) -> Dict[int, List[int]]:
    plan_list = [
        MemoryGreedyBySize(graph, is_dump),
        MemoryGreedyByBreadth(graph, is_dump),
        MemoryGreedyByOrder(graph, is_dump),
    ]

    memory_list = []
    for k, v in plan_list[0].mem_sizes.items():
        memory_list.append(k)

    best_plan = {}
    for i in range(len(memory_list)):
        best_id = np.array(
            [sum(x.mem_sizes[memory_list[i]]) for x in plan_list]
        ).argmin()
        best_plan = plan_list[int(best_id)]
        best_plan.mem_sizes[memory_list[i]] = list(
            map(int, best_plan.mem_sizes[memory_list[i]])
        )
        if is_dump:
            print(
                "  memory allocate on {}:{},total:{}".format(
                    MemType(memory_list[i]).name,
                    best_plan.mem_sizes[memory_list[i]],
                    sum(best_plan.mem_sizes[memory_list[i]]),
                )
            )
    return best_plan


def get_memory_size(memory_plan: Dict[int, List[int]], memory_type: MemType) -> int:
    return sum(memory_plan.mem_sizes[memory_type.value])


__all__ = [
    "WORKSPACE_NAME",
    "DMA_BUFFER1_NAME",
    "DMA_BUFFER2_NAME",
    "memory_plan",
    "get_memory_size",
]
