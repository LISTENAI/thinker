import math
from typing import List, Tuple, Dict
from ._type import *
from ..graph import Graph, ScalarGraph
from ..enum_defines import MemType, ALIGN16, TensorType
from ..graph_analysis.memory import WORKSPACE_NAME, DMA_BUFFER1_NAME, DMA_BUFFER2_NAME


def pack_memory(memory_planer: Dict[int, List[int]]) -> List[tMemory]:
    runtime_memory_list = []
    for k, v in memory_planer.mem_sizes.items():
        runtime_size = sum(v)
        runtime_memory_list.append([tMemory(k, runtime_size)])
    return runtime_memory_list

def pack_tensor(memory_planer: Dict[int, List[int]]) -> Tuple[List[tTensor], List[tTensorName]]:
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
            if tensor.bits == 0.5:
                param_offset[ctx.entry.tensor.mem_type.value] += math.ceil(tensor.nbytes * tensor.bits)
            else:
                param_offset[ctx.entry.tensor.mem_type.value] += tensor.nbytes

        tensor_list.append(tTensor(tensor, run_mem_id, offset, ctx.entry.tensor.mem_type.value))
        tensor_name_list.append(tTensorName(ctx.entry.name))
    return tensor_list, tensor_name_list

def pack_operator(graph: Graph, memory_planer: Dict[int, List[int]], arch=None) -> Tuple[List[tOperator], List[tDMA]]:
    node_index = 0
    dma_list = []
    operator_list = []
    for node in graph.nodes.values():
        if not hasattr(node, "op_type"):
            continue
        op_attrs = node.op.pack_attrs()
        temp_tensor_ids = []
        dma_tensor_ids = []
        if node.op.get_workspace() != None and len(node.op.get_workspace()) != 0:
            for ctxt in memory_planer.entry_ctx_list:
                if "workspace" in ctxt.entry.name:
                    temp_tensor_ids.append(ctxt.entry.index)
                    break
        tensor_ids = ([x.index for x in node.inputs] + [x.index for x in node.outputs] + temp_tensor_ids)
        size = 0
        mem_type = None
        param_ids = []
        if node.op_type in {"Conv2dInt", "ConvTranspose2dInt", "LinearInt", "LSTMInt", "GRUInt", "LayerNormInt", "Conv1dInt"}:
        # if node.op_type in {"Conv2dInt", "ConvTranspose2dInt", "LinearInt", "LayerNormInt", "Conv1dInt"}:
            for x in node.inputs:
                if (x.tensor.mem_type != MemType.SHARE_MEM and x.is_constant() and x.tensor.data is not None):
                    param_ids.append(x.index)
                    if x.tensor.bits != 0.5:
                        size += x.tensor.nbytes 
                    else:
                        size += x.tensor.nbytes//2
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
                    dma_list.append([tDMA(mem_type, MemType.SHARE_MEM,param_ids[0], dma_tensor_ids[0],size,)])
                node_index += 1
        # elif node.op_type == "LSTMInt":
        #     for x in node.inputs:
        #         size = 0
        #         mem_type = None
        #         param_ids = []
        #         dma_tensor_ids = []
        #         if (x.tensor.mem_type != MemType.SHARE_MEM and x.is_constant() and x.name.find("ih") != -1):
        #             param_ids.append(x.index)
        #             if x.tensor.bits != 0.5:
        #                 size += x.tensor.nbytes 
        #             else:
        #                 size += x.tensor.nbytes//2
        #             mem_type = x.tensor.mem_type
        #         if size != 0:
        #             if node_index % 2 == 0:
        #                 for ctxt in memory_planer.entry_ctx_list:
        #                     if ctxt.entry.name == DMA_BUFFER1_NAME:
        #                         dma_tensor_ids.append(ctxt.entry.index)
        #                         tensor_ids.append(ctxt.entry.index)
        #                         break
        #             else:
        #                 for ctxt in memory_planer.entry_ctx_list:
        #                     if ctxt.entry.name == DMA_BUFFER2_NAME:
        #                         dma_tensor_ids.append(ctxt.entry.index)
        #                         tensor_ids.append(ctxt.entry.index)
        #                         break
        #             if len(dma_tensor_ids) != 0:
        #                 dma_list.append([tDMA(mem_type, MemType.SHARE_MEM,param_ids[0], dma_tensor_ids[0],size,)])
        #             node_index += 1

        #     for x in node.inputs:
        #         size = 0
        #         mem_type = None
        #         param_ids = []
        #         dma_tensor_ids = []
        #         if (x.tensor.mem_type != MemType.SHARE_MEM and x.is_constant() and x.name.find("hh") != -1):
        #             param_ids.append(x.index)
        #             if x.tensor.bits != 0.5:
        #                 size += x.tensor.nbytes 
        #             else:
        #                 size += x.tensor.nbytes//2
        #             mem_type = x.tensor.mem_type
        #         if size != 0:
        #             if node_index % 2 == 0:
        #                 for ctxt in memory_planer.entry_ctx_list:
        #                     if ctxt.entry.name == DMA_BUFFER1_NAME:
        #                         dma_tensor_ids.append(ctxt.entry.index)
        #                         tensor_ids.append(ctxt.entry.index)
        #                         break
        #             else:
        #                 for ctxt in memory_planer.entry_ctx_list:
        #                     if ctxt.entry.name == DMA_BUFFER2_NAME:
        #                         dma_tensor_ids.append(ctxt.entry.index)
        #                         tensor_ids.append(ctxt.entry.index)
        #                         break
        #             if len(dma_tensor_ids) != 0:
        #                 dma_list.append([tDMA(mem_type, MemType.SHARE_MEM,param_ids[0], dma_tensor_ids[0],size,)])
        #             node_index += 1

        num_input = len(node.inputs)
        num_output = len(node.outputs)
        operator_list.append(tOperator(op_attrs, node.op_type, "HIFI", num_input, num_output, tensor_ids))
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

def pack_param(memory_planer: Dict[int, List[int]]) -> Tuple[List[tParameter], List[tMemory]]:
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
            if (tensor is not None and tensor.data is not None and tensor.mem_type == memory_list[d]):
                t = tensor.data

                # if ctx.entry.name.find('ih') != -1 or ctx.entry.name.find('hh') != -1:
                #     t_weight = t[:256,:].transpose(1, 0)
                #     t_bias   = t[256:,:].transpose(1,0)
                #     offset = ALIGN16(len(t_weight.tobytes())+len(t_bias.tobytes()))
                #     params_buff += t_weight.tobytes() + t_bias.tobytes() + b"\0" * (offset - len(t_weight.tobytes()) - len(t_bias.tobytes()))
                #     if tensor.nbytes != offset:
                #         print(ctx.entry.name, "the size of tensor({}) do not match offset({})!".format(tensor.nbytes, offset))    
                #     continue
                offset = ALIGN16(len(t.tobytes()))
                params_buff += t.tobytes() + b"\0" * (offset - len(t.tobytes()))

                if tensor.bits == 0.5:
                    if (tensor.nbytes * tensor.bits) != len(t.tobytes()):
                        print(ctx.entry.name, "the size of tensor({}) do not match offset({})!".format(tensor.nbytes * tensor.bits, len(t.tobytes())))
                else:
                    if tensor.nbytes != offset:
                        print(ctx.entry.name, "the size of tensor({}) do not match offset({})!".format(tensor.nbytes, offset))

        param_list.append(tParameter(memory_list[d].value, 0, params_buff))
        shared_memory_list.append(tMemory(memory_list[d].value, len(params_buff)))
        return param_list, shared_memory_list

def pack_shape(graph:Graph)->Tuple[List[tDyAxisInfo],List[tTenDimPair], ScalarGraph]:
    shape_exprs     = list()
    scalar_inputs   = dict()
    id_pairs_list   = []
    dy_axis_list    = []

    if len(graph.dynamic_shape.keys()) == 0:
        scalar_graph = ScalarGraph.from_exprs([])
        return dy_axis_list,id_pairs_list,scalar_graph

    for expr, dyshape in graph.dynamic_shape.items():
        for name, dim_id in dyshape:
            tensor_id = graph.entries[name].index
            id_pairs_list.append(tTenDimPair(tensor_id, dim_id))
            shape_exprs.append(expr)
            if name in graph.inputs:
                for sym in expr.atoms():
                    if sym.is_symbol:
                        if name not in scalar_inputs:
                            scalar_inputs[name] = []
                        scalar_inputs[name].append((sym, dim_id))

    scalar_graph = ScalarGraph.from_exprs(shape_exprs)
    

    for e in graph.inputs:
        tensor_id = e.index
        if e.name in scalar_inputs:
            for (expr, dim_id) in scalar_inputs[e.name]:
                input_id = scalar_graph.input_names.index(expr)
                dy_axis_list.append(tDyAxisInfo(tensor_id, dim_id, input_id))
    print('')
    return dy_axis_list, id_pairs_list, scalar_graph


__all__ = ["pack_memory", "pack_tensor", "pack_operator", "pack_io", "pack_param", "pack_shape"]
