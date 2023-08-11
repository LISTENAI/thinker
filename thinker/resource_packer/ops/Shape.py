import numpy as np

from ...graph import Tensor
from .base import Operator, OperatorAttrs, register_op

@register_op
class Shape(Operator):
    def __init__(self, attrs = {}):
        self.attrs = OperatorAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) == 1
        X = inputs[0]
        self.inputs = inputs
        self.shape = np.array(X.shape)
        t = Tensor.from_numpy(self.shape)
        t.dtype = np.dtype(np.int64)
        t.data = np.array(self.shape)
        t.scale = X.scale
        self.outputs = [t]

__all__ = ["Shape"]