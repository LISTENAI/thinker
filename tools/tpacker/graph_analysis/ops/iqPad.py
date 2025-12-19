import math
import numpy as np
from typing import List
from ...graph import Tensor
from ...enum_defines import DevType
from ...resource_packer._type._ctype import tffi
from .base import iqBinaryOperator, OperatorAttrs, register_op

class iqPadAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        """Check if all required parameters are present and valid."""
        assert "mode" in self.attrs, "Missing required attribute: mode"
        assert self.attrs["mode"] in {'constant', 'reflect', 'replicate'}, "Invalid mode"

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the iqPad operation."""
        attrs = tffi.new("iqPadAttrs *")
        mode = self.attrs["mode"]
        attrs.mode = 0 if mode == 'constant' else 1 if mode == 'reflect' else 2
        return bytes(tffi.buffer(attrs))

@register_op
class iqPad(iqBinaryOperator):
    def __init__(self, attrs={}):
        """Initialize the iqPad operator with given attributes."""
        self.attrs = iqPadAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape based on inputs and padding."""
        inputs = self.inputs
        assert len(inputs) == 3, "iqPad operator must have exactly three inputs"

        X = inputs[0]
        pads = inputs[1]
        shape = list(X.shape)

        assert len(X.shape) >= len(pads.shape) and len(X.shape) <= 4, "Input shape must be between 1D and 4D"
        assert len(pads.shape) == 1, "Padding must be a 1D tensor"

        pad_size = pads.shape[0]
        if pad_size == 2:
            shape[-1] += pads.data[1] * 2
            shape[-2] += pads.data[0] * 2
        elif pad_size == 4:
            shape[-1] += pads.data[1] + pads.data[3]
            shape[-2] += pads.data[0] + pads.data[2]
        elif pad_size == 6:
            shape[-1] += pads.data[2] + pads.data[5]
            shape[-2] += pads.data[1] + pads.data[4]
            shape[-3] += pads.data[0] + pads.data[3]
        elif pad_size == 8:
            shape[-1] += pads.data[3] + pads.data[7]
            shape[-2] += pads.data[2] + pads.data[6]
            shape[-3] += pads.data[1] + pads.data[5]
            shape[-4] += pads.data[0] + pads.data[4]

        Y = X.clone(shape=tuple(shape))
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the iqPad operation."""
        input_data = self.inputs[0]
        workspace_size = input_data.nbytes
        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, input_data.mem_type)]
        return []

__all__ = ["iqPad"]