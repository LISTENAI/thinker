import onnx
import numpy as np
from onnx import TensorProto
from typing import Dict, Any

from ...xsympy import is_sympy
from .base import Operator, OperatorAttrs, register_op


def parse_tensor(t: TensorProto) -> np.ndarray:
    """Convert ONNX TensorProto to NumPy array."""
    try:
        from onnx.numpy_helper import to_array
    except ImportError:
        raise ImportError("ONNX and protobuf must be installed. "
                          "Installation instructions: https://github.com/onnx/onnx")

    if len(t.dims) > 0:
        return to_array(t).reshape(tuple(t.dims))
    else:
        # Handle scalar values without dimensions
        return np.array([to_array(t)])


class ConstantAttrs(OperatorAttrs):
    """Attributes handler for Constant operator."""
    
    def normalize(self):
        """Normalize attributes to prepare for tensor creation."""
        valuetensor = self.attrs.get('value')
        if isinstance(valuetensor, TensorProto):
            value = parse_tensor(valuetensor)
            data_type = value.dtype
            if data_type == np.float32:
                self.attrs['value'] = float(value[0])
                self.attrs['type'] = np.float32
            elif data_type == np.int64:
                self.attrs['value'] = int(value[0])
                self.attrs['type'] = np.int64
        else:
            self.attrs['type'] = np.float32
            self.attrs['value'] = 0

        nptype = self.attrs['type']
        # Calculate pack type based on dtype string
        self.attrs['pack_type'] = (ord(nptype.str[-2]) << 8) + int(nptype.str[-1])


@register_op
class Constant(Operator):
    """Generate a constant tensor with specified value and shape."""
    
    def __init__(self, attrs: Dict = {}):
        super().__init__()
        self.attrs = ConstantAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor shape and data."""
        assert len(self.inputs) == 1, "Constant operator expects exactly one input tensor"
        X = self.inputs[0]
        xshape = list(X.data)
        Y = X.clone(shape=tuple(xshape), dtype=self.attrs['type'])

        # Check if all dimensions are resolved (not symbolic)
        if all(not is_sympy(s) for s in xshape):
            Y.data = np.full(shape=xshape, fill_value=self.attrs['value'], dtype=self.attrs['type'])

        self.outputs = [Y]


__all__ = ['Constant']