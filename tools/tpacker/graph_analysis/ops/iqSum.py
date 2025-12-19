import math
import numpy as np
from typing import List

from ...graph import Tensor
from ...enum_defines import DevType, MemType
from ...resource_packer._type._ctype import tffi
from .base import iqBinaryOperator, OperatorAttrs, register_op

class iqSumAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        """Check if required parameters are present."""
        assert "dims" in self.attrs, "Missing required attribute: dims"

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the iqSum operation."""
        attrs = tffi.new("iqSumAttrs *")
        attrs.axis = self.attrs["dims"]
        return bytes(tffi.buffer(attrs))

@register_op
class iqSum(iqBinaryOperator):
    def __init__(self, attrs={}):
        """Initialize the iqSum operator with given attributes."""
        self.attrs = iqSumAttrs()
        self.attrs.attrs = attrs

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        assert len(inputs) == 1, "iqSum operator must have exactly one input"

        X = inputs[0]
        axis = self.attrs["dims"]

        # Ensure axis is within valid range
        assert -len(X.shape) <= axis < len(X.shape), "Axis out of bounds"

        # Calculate output shape
        output_shape = list(X.shape)
        output_shape[axis] = 1

        # Create output tensor
        Y = X.clone(shape=tuple(output_shape))
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the iqSum operation."""
        shape = list(self.inputs[0].shape)
        axis = self.attrs["dims"]
        shape[axis] = 1

        workspace_size = 1
        for s in shape:
            workspace_size *= s
        workspace_size *= 4  # Assuming 4 bytes per element

        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

__all__ = ["iqSum"]