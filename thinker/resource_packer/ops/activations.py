import numpy as np
from ...graph import Tensor
from .utils import QuantType
from ...enum_defines import DevType, MemType
from .base import Operator, UnaryOperator, BaseLayout, OperatorAttrs, register_op

@register_op
class Relu(UnaryOperator, BaseLayout):
    def get_workspace(self, dev_type: DevType):
        input = self.inputs[0]
        output = self.outputs[0]
        workspace_size = 0
        if input.mem_type != MemType.SHARE_MEM and dev_type == DevType.LUNA:
            workspace_size += input.nbytes
        if output.mem_type != MemType.SHARE_MEM and dev_type == DevType.LUNA:
            workspace_size += output.nbytes
        
        workspace_size = min(65536, workspace_size)
        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, dev_type)]
        else:
            return []
        
__all__ = ["Relu"]
