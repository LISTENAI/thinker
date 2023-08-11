import numpy as np
from typing import Any, Dict, Optional

from .._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op


class CastAttrs(OperatorAttrs):
    def serialize(self) -> bytes:
        attrs = tffi.new("CastAttrs")
        attrs.to = self.attrs["to"]
        return bytes(tffi.buffer(attrs))


@register_op
class Cast(Operator):
    def __init__(self, attrs={}):
        self.attrs = CastAttrs(attrs)

    def infer_tensor(self):
        X = self.inputs[0]
        yshape = X.shape
        Y = X.clone(shape=yshape, scale=int(X.scale))
        Y.data = X.data
        Y.dtype = np.dtype(self.attrs["to"])
        self.outputs = [Y]

__all__ = ["Cast"]
