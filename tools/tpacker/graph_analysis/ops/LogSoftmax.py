import math
import numpy as np
from typing import List

from ...graph import Tensor
from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, UnaryOperator, register_op

class LogSoftmaxAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        """Check if required parameters are present."""
        assert "axis" in self.attrs, "Missing required attribute: axis"

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the LogSoftmax operation."""
        attrs = tffi.new("LogSoftmaxAttrs *")
        attrs.axis = self.attrs["axis"]
        return bytes(tffi.buffer(attrs))

@register_op
class LogSoftmax(UnaryOperator):
    def __init__(self, attrs={}):
        """Initialize the LogSoftmax operator with given attributes."""
        self.attrs = LogSoftmaxAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on input."""
        X = self.inputs[0]
        axis = self.attrs["axis"]

        # Ensure axis is within valid range
        assert -len(X.shape) <= axis < len(X.shape), "Axis out of bounds"

        # Create output tensor with the same shape as input
        Y = X.clone()
        self.outputs = [Y]

        # Perform LogSoftmax computation if data is available
        if X.has_data():
            X_data = X.data
            max_val = np.max(X_data, axis=axis, keepdims=True)
            exp = np.exp(X_data - max_val)
            sum_exp = np.sum(exp, axis=axis, keepdims=True)
            Y.data = np.log(exp / sum_exp)

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the LogSoftmax operation."""
        # No additional workspace is needed for LogSoftmax
        return []

    def flops_counter(self, dynamic_shape) -> int:
        """Calculate the number of floating-point operations (FLOPs) for the LogSoftmax operation."""
        X = self.inputs[0]
        xshape = list(X.shape)
        
        # Calculate FLOPs based on input shape
        flops = int(np.prod(xshape)) * 3  # exp, sum, log operations
        return flops

__all__ = ["LogSoftmax"]