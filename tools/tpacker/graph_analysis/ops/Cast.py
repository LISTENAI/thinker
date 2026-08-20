import numpy as np
from typing import Any, Dict, Optional

from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op


class CastAttrs(OperatorAttrs):
    """Attributes handler for Cast operator."""

    def normalize(self) -> None:
        self.attrs["to"] = np.dtype(self.attrs["to"])

    def checkparams(self) -> None:
        supported_dtypes = {
            np.dtype(np.float32),
            np.dtype(np.int8), np.dtype(np.int16), np.dtype(np.int32), np.dtype(np.int64),
            np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.uint32), np.dtype(np.uint64),
        }
        assert self.attrs["to"] in supported_dtypes, \
            f"Cast does not support target dtype {self.attrs['to']}"
    
    def serialize(self) -> bytes:
        """Serialize attributes to bytes."""
        attrs = tffi.new("CastAttrs *")
        dtype = self.attrs["to"]
        attrs.to = (ord(dtype.str[-2]) << 8) + int(dtype.str[-1])
        return bytes(tffi.buffer(attrs))


@register_op
class Cast(Operator):
    """Cast operator to change tensor data type."""
    
    def __init__(self, attrs: Dict = {}):
        super().__init__()
        self.attrs = CastAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor with specified data type."""
        assert len(self.inputs) == 1, "Cast requires exactly one input tensor"
        X = self.inputs[0]
        supported_dtypes = {
            np.dtype(np.float32),
            np.dtype(np.int8), np.dtype(np.int16), np.dtype(np.int32), np.dtype(np.int64),
            np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.uint32), np.dtype(np.uint64),
        }
        assert X.dtype in supported_dtypes, f"Cast does not support input dtype {X.dtype}"
        Y = X.clone(shape=X.shape, scale=X.scale)
        Y.dtype = self.attrs["to"]
        Y.bits = Y.dtype.itemsize
        if X.data is not None:
            Y.data = X.data.astype(Y.dtype)
        self.outputs = [Y]


__all__ = ["Cast"]
