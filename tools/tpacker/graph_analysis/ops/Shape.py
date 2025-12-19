import numpy as np

from ...graph import Tensor
from .base import Operator, OperatorAttrs, register_op


@register_op
class Shape(Operator):
    def __init__(self, attrs={}):
        """Initialize Shape operator with given attributes"""
        self.attrs = OperatorAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the shape of the input tensor and output it as a tensor"""
        assert len(self.inputs) == 1
        X = self.inputs[0]
        shape = np.array(X.shape)
        t = Tensor.from_numpy(shape)
        t.dtype = np.dtype(np.int64)
        t.data = np.array(shape)
        t.scale = X.scale
        self.outputs = [t]


__all__ = ["Shape"]