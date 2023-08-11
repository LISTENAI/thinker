import math
import numpy as np

from ...graph import Tensor
from .._type._ctype import tffi
from ...enum_defines import DevType
from .base import Operator, OperatorAttrs, UnaryOperator, register_op

class LogSoftmaxAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        assert "axis" in self.attrs

    def serialize(self) -> bytes:
        attrs = tffi.new("LogSoftmaxAttrs *")
        attrs.axis = self.attrs["axis"]
        return bytes(tffi.buffer(attrs))

@register_op
class LogSoftmax(UnaryOperator):
    def __init__(self, attrs={}):
        self.attrs = LogSoftmaxAttrs(attrs)

        
__all__ = ["LogSoftmax"]