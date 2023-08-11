import math
import numpy as np

from ...graph import Tensor
from .._type._ctype import tffi
from ...enum_defines import DevType
from .base import iqBinaryOperator, OperatorAttrs, register_op

class iqSumAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        assert "dims" in self.attrs

    def serialize(self) -> bytes:
        attrs = tffi.new("iqSumAttrs *")

        attrs.axis = self.attrs["dims"]

        return bytes(tffi.buffer(attrs))

@register_op
class iqSum(iqBinaryOperator):
    def __init__(self, attrs={}):
        self.attrs = iqSumAttrs(attrs)

    def get_workspace(self, dev_type: DevType):
        workspace_size = 0
        shape = list(self.inputs[0].shape)
        axis = self.attrs["dims"]
        shape[axis] = 1
        workspace_size = 1
        for s in shape:
            workspace_size *= s
        workspace_size *=  4
        max_workspace = Tensor.from_shape([workspace_size], np.int8, dev_type)
        max_workspace.dev_type = dev_type

        return [max_workspace]


__all__ = ["iqSum"]
