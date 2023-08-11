import math
import numpy as np

from ...graph import Tensor
from .._type._ctype import tffi
from ...enum_defines import DevType
from .base import Operator, OperatorAttrs, iqUnaryOperator, register_op


class SoftmaxIntAttrs(OperatorAttrs):
    def serialize(self) -> bytes:
        attrs = tffi.new("SoftmaxIntAttrs *")
        attrs.axis = self.attrs["dim"]
        return bytes(tffi.buffer(attrs))


@register_op
class SoftmaxInt(iqUnaryOperator):
    def __init__(self, attrs={}):
        self.attrs = SoftmaxIntAttrs(attrs)

    def get_workspace(self, dev_type: DevType):
        axis = self.attrs.get("dim", -1)
        workspace_bytes = self.inputs[0].shape[axis] * 2
        max_workspace = Tensor.from_shape([workspace_bytes], np.int32, dev_type)
        return [max_workspace]


__all__ = ["SoftmaxInt"]
