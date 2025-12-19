# Copyright (C) 2025 listenai Co.Ltd
# All rights reserved.
# Created by leifang on 2022.09.31

from typing import Dict, List

from ..devices import Device
from ..graph import Graph
from ._type import (
    tModel,
    tMemoryList,
    tTensorList,
    tOperatorList,
    tIOInfo,
    tParameterList,
    tShapeInferHdr,
    tDebugList,
    tDMAList
)
from ..resource_packer.venus import (
    pack_param,
    pack_memory,
    pack_tensor,
    pack_operator,
    pack_io,
    pack_shape
)

def serialize_model(graph: Graph, memory_plan: Dict[int, List[int]], device: Device):
    # 打包参数
    param_list, shared_memory_list = pack_param(memory_plan)
    
    # 打包张量
    runtime_memory_list = pack_memory(memory_plan)
    tensor_list, tensor_name_list = pack_tensor(memory_plan)
    
    # 打包操作符
    operator_list, dma_list = pack_operator(graph, memory_plan)
    
    # 打包输入输出
    input_list, output_list = pack_io(graph)
    
    # 打包形状信息
    dy_axis_list, id_pairs_list, scalar_graph = pack_shape(graph)
    
    # 根据平台调整内存配置
    if graph.platform.lower() == "mars":
        shared_memory_list = []
        runtime_memory_list = []
        dma_list = []
    
    # 构建模型
    packed_model = tModel(
        graph.platform,
        tMemoryList(shared_memory_list, runtime_memory_list, device.sram_size),
        tTensorList(tensor_list),
        tOperatorList(operator_list),
        tIOInfo(input_list, output_list),
        tParameterList(param_list),
        tShapeInferHdr(dy_axis_list, id_pairs_list, scalar_graph),
        tDebugList(tensor_name_list),
        tDMAList(dma_list)
    )
    
    return packed_model

__all__ = ["serialize_model"]