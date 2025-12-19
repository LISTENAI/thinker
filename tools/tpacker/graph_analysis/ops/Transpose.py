import math
import numpy as np
from typing import Any, Dict, List, Optional

from ...graph import Tensor
from ...xsympy import is_sympy
from ...enum_defines import DevType, MemType, Layout
from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op, BaseLayout

class TransposeAttrs(OperatorAttrs):
    def __init__(self, attrs: Optional[Dict[str, Any]] = {}) -> None:
        """Initialize Transpose attributes with given parameters."""
        super().__init__(attrs, "TransposeAttrs")

    def checkparams(self) -> None:
        """Validate and set default parameters for Transpose operation."""
        self.attrs["perm"] = self.attrs.get("perm", [])
        assert len(self.attrs["perm"]) < 7, "perm length must be less than 7"
        self.attrs["ndim"] = len(self.attrs["perm"])

    def serialize(self) -> bytes:
        """Serialize Transpose attributes to bytes."""
        attrs = tffi.new("TransposeAttrs *")
        attrs.axes_ = self.attrs["perm"]
        attrs.ndim_ = self.attrs["ndim"]
        return bytes(tffi.buffer(attrs))

@register_op
class Transpose(Operator):
    def __init__(self, attrs={}):
        """Initialize Transpose operator with given attributes."""
        self.attrs = TransposeAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor shape by transposing input tensor according to specified axes."""
        inputs = self.inputs
        assert len(inputs) == 1, "Transpose operation requires exactly one input"
        X = inputs[0]
        shape = list(X.shape)
        axes = self.attrs["perm"]

        if not axes:
            axes = list(range(len(shape) - 1, -1, -1))

        assert len(axes) <= len(shape), "Number of axes exceeds input dimensions"
        new_shape = [shape[x] for x in axes]
        new_shape += shape[len(axes):]

        Y = X.clone(shape=tuple(new_shape), scale=X.scale)

        # Handle layout conversion
        if X.layout == Layout.NCHW and tuple(axes) == (0, 1, 3, 2):
            Y.layout = Layout.NCWH
        elif X.layout == Layout.NCWH and tuple(axes) == (0, 1, 3, 2):
            Y.layout = Layout.NCHW

        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace size for the operation."""
        X = self.inputs[0]
        Y = self.outputs[0]
        platform = self.attrs.get("platform", "venus")

        workspace_size = 0
        if X.mem_type != MemType.SHARE_MEM:
            workspace_size += X.nbytes
        if Y.mem_type != MemType.SHARE_MEM and X.nbytes >= 65536:
            workspace_size += Y.nbytes

        workspace_size = min(workspace_size, 65536)
        if workspace_size != 0:
            max_workspace = Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)
            return [max_workspace]

__all__ = ["Transpose"]