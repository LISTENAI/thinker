import numpy as np
from typing import Any, Dict, Optional

from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op


class CastAttrs(OperatorAttrs):
    """Attributes handler for Cast operator."""
    
    def serialize(self) -> bytes:
        """Serialize attributes to bytes."""
        attrs = tffi.new("CastAttrs")
        attrs.to = self.attrs["to"]
        return bytes(tffi.buffer(attrs))


@register_op
class Cast(Operator):
    """Cast operator to change tensor data type."""
    
    def __init__(self, attrs: Dict = {}):
        super().__init__()
        self.attrs = CastAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor with specified data type."""
        X = self.inputs[0]
        Y = X.clone(shape=X.shape, scale=int(X.scale))
        Y.data  = X.data
        Y.dtype = np.dtype(self.attrs["to"])
        Y.bits = Y.dtype.itemsize
        self.outputs = [Y]


__all__ = ["Cast"]