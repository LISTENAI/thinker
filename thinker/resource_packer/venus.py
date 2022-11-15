from typing import List, Tuple, Dict

from ..graph import Graph
from ..enum_defines import MemType, ALIGN16, TensorType
from .memory import WORKSPACE_NAME, DMA_BUFFER1_NAME, DMA_BUFFER2_NAME
from ._type import tDMA, tParameter, tMemory, tTensor, tTensorName, tOperator, tIO


def pack_memory(memory_planer: Dict[int, List[int]]) -> List[tMemory]:
    runtime_memory_list = []
    for k, v in memory_planer.mem_sizes.items():
        runtime_size = sum(v)
        runtime_memory_list.append([tMemory(k, runtime_size)])
    return runtime_memory_list


def pack_tensor(
    memory_planer: Dict[int, List[int]]
) -> Tuple[List[tTensor], List[tTensorName]]:
    mem_offset = {}
    for k, v in memory_planer.mem_sizes.items():
        mem_offset1 = [0]
        for i, x in enumerate(v):
            mem_offset1.append(mem_offset1[i] + x)

        if k not in mem_offset:
            mem_offset[k] = mem_offset1

    param_offset = {}
    for i, ctx in enumerate(memory_planer.entry_ctx_list):
        if ctx.entry.tensor.mem_type.value not in param_offset:
            param_offset[ctx.entry.tensor.mem_type.value] = 0
    tensor_name_list = []
    tensor_list = []
    for i, ctx in enumerate(memory_planer.entry_ctx_list):
        mem_id = ctx.mem_id
        tensor = ctx.entry.tensor
        run_mem_id = 0
        offset = 0
        if mem_id != -1:
            offset = mem_offset[ctx.entry.tensor.mem_type.value][mem_id]
            run_mem_id = 1
        else:
            run_mem_id = 0
            offset = param_offset[ctx.entry.tensor.mem_type.value]
            param_offset[ctx.entry.tensor.mem_type.value] += tensor.nbytes

        tensor_list.append(
            tTensor(tensor, run_mem_id, offset, ctx.entry.tensor.mem_type.value)
        )
        tensor_name_list.append(tTensorName(ctx.entry.name))
    return tensor_list, tensor_name_list


def pack_operator(
    graph: Graph, memory_planer: Dict[int, List[int]], arch=None
) -> Tuple[List[tOperator], List[tDMA]]:
    node_index = 0
    dma_list = []
    operator_list = []
    for node in graph.nodes.values():
        if not hasattr(node, "op_type"):
            continue
        op_attrs = node.op.pack_attrs()
        temp_tensor_ids = []
        dma_tensor_ids = []
        if len(node.op.get_workspace(node.dev_type)) != 0:
            for ctxt in memory_planer.entry_ctx_list:
                if "workspace" in ctxt.entry.name:
                    temp_tensor_ids.append(ctxt.entry.index)
                    break
        tensor_ids = (
            [x.index for x in node.inputs]
            + [x.index for x in node.outputs]
            + temp_tensor_ids
        )
        size = 0
        mem_type = None
        param_ids = []
        if (
            node.op_type == "Conv2dInt"
            or node.op_type == "ConvTranspose2dInt"
            or node.op_type == "LinearInt"
            or node.op_type == "LSTMInt"
            or node.op_type == "LayerNormInt"
            or node.op_type == "Conv1dInt"
        ):
            for x in node.inputs:
                if (
                    x.tensor.mem_type != MemType.SHARE_MEM
                    and x.is_constant()
                    and x.tensor.data is not None
                ):
                    param_ids.append(x.index)
                    size += x.tensor.nbytes
                    mem_type = x.tensor.mem_type
            if size != 0:
                if node_index % 2 == 0:
                    for ctxt in memory_planer.entry_ctx_list:
                        if ctxt.entry.name == DMA_BUFFER1_NAME:
                            dma_tensor_ids.append(ctxt.entry.index)
                            tensor_ids.append(ctxt.entry.index)
                            break
                else:
                    for ctxt in memory_planer.entry_ctx_list:
                        if ctxt.entry.name == DMA_BUFFER2_NAME:
                            dma_tensor_ids.append(ctxt.entry.index)
                            tensor_ids.append(ctxt.entry.index)
                            break

                if len(dma_tensor_ids) != 0:
                    dma_list.append(
                        [
                            tDMA(
                                mem_type,
                                MemType.SHARE_MEM,
                                param_ids[0],
                                dma_tensor_ids[0],
                                size,
                            )
                        ]
                    )
                node_index += 1
        num_input = len(node.inputs)
        num_output = len(node.outputs)

        operator_list.append(
            tOperator(
                op_attrs, node.op_type, node.dev_type, num_input, num_output, tensor_ids
            )
        )
    return operator_list, dma_list


def pack_io(graph: Graph) -> Tuple[List[tIO], List[tIO]]:
    input_list = []
    for input_entry in graph.inputs:
        if input_entry.tensor_type == TensorType.Input:
            tid = input_entry.index
            input_list.append(tIO(tid, input_entry.name))

    output_list = []
    for output_entry in graph.outputs:
        tid = output_entry.index
        output_list.append(tIO(tid, output_entry.name))
    return input_list, output_list


def pack_param(
    memory_planer: Dict[int, List[int]]
) -> Tuple[List[tParameter], List[tMemory]]:
    memory_list = []
    for ctx in memory_planer.entry_ctx_list:
        tensor = ctx.entry.tensor
        if tensor is not None and tensor.data is not None:
            if tensor.mem_type not in memory_list:
                memory_list.append(tensor.mem_type)

    param_list = []
    shared_memory_list = []
    for d in range(len(memory_list)):
        params_buff = b""
        for ctx in memory_planer.entry_ctx_list:
            tensor = ctx.entry.tensor
            if (
                tensor is not None
                and tensor.data is not None
                and tensor.mem_type == memory_list[d]
            ):
                t = tensor.data
                offset = ALIGN16(len(t.tobytes()))
                params_buff += t.tobytes() + b"\0" * (offset - len(t.tobytes()))
                if tensor.nbytes != offset:
                    print(
                        ctx.entry.name,
                        "the size of tensor:{} do not math offset{}!".format(
                            tensor.nbytes, offset
                        ),
                    )

        param_list.append(tParameter(memory_list[d].value, 0, params_buff))
        shared_memory_list.append(tMemory(memory_list[d].value, len(params_buff)))
        return param_list, shared_memory_list


__all__ = ["pack_memory", "pack_tensor", "pack_operator", "pack_io", "pack_param"]
