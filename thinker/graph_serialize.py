# Copyright (C) 2022 listenai Co.Ltd
# All rights reserved. 
# Created by leifang on 2022.09.31

from typing import List, Dict

from .graph import Graph
from .resource_packer import (
    pack_param,
    pack_memory,
    pack_tensor,
    pack_operator,
    pack_io,
)
from .resource_packer import (
    tModel,
    tMemoryList,
    tTensorList,
    tOperatorList,
    tIOInfo,
    tParameterList,
    tDebugList,
    tDMAList,
)


def serialize(graph: Graph, memory_plan: Dict[int, List[int]]):
    print("pack param begin")
    param_list, shared_memory_list = pack_param(memory_plan)
    print("pack param success")
    print("pack tensor begin")
    runtime_memory_list = pack_memory(memory_plan)
    tensor_list, tensor_name_list = pack_tensor(memory_plan)
    print("pack tensor success")
    print("pack operator begin")
    operator_list, dma_list = pack_operator(graph, memory_plan)
    print("pack operator success")
    print("pack io begin")
    input_list, output_list = pack_io(graph)
    print("pack io success")
    packed_model = tModel(
        tMemoryList(shared_memory_list, runtime_memory_list),
        tTensorList(tensor_list),
        tOperatorList(operator_list),
        tIOInfo(input_list, output_list),
        tParameterList(param_list),
        tDebugList(tensor_name_list),
        tDMAList(dma_list),
    )
    return packed_model


__all__ = ["serialize"]
