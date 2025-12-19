import onnx
import numpy as np
from onnx import TensorProto
from ...xsympy import is_sympy
from .base import Operator, OperatorAttrs, register_op

def parse_tensor(t: TensorProto) -> np.array:
    """Extract data from TensorProto and convert to numpy array."""
    from onnx.numpy_helper import to_array
    if t.dims:
        return to_array(t).reshape(tuple(t.dims))
    else:
        return np.array([to_array(t)])

class ConstantOfShapeAttrs(OperatorAttrs):
    def normalize(self):
        valuetensor = self.attrs.get('value')
        if isinstance(valuetensor, onnx.TensorProto):
            v = parse_tensor(valuetensor)
            data_type = v.dtype
            if data_type == np.float32:
                self.attrs['value'] = float(v[0])
                self.attrs['type'] = np.dtype('float32')
            elif data_type == np.int64:
                self.attrs['value'] = int(v[0])
                self.attrs['type'] = np.dtype('int64')
        else:
            self.attrs['type'] = np.dtype('float32')
            self.attrs['value'] = 0
        nptype = self.attrs['type']
        self.attrs['pack_type'] = (ord(nptype.str[-2]) << 8) + int(nptype.str[-1])

@register_op
class ConstantOfShape(Operator):
    def __init__(self, attrs={}):
        self.attrs = ConstantOfShapeAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        assert len(self.inputs) == 1
        X = self.inputs[0]
        xshape = list(X.data)
        Y = X.clone(shape=tuple(xshape), dtype=self.attrs['type'])
        if all(not is_sympy(s) for s in xshape):
            Y.data = np.full(shape=xshape, fill_value=self.attrs['value'], dtype=self.attrs['type'])
        self.outputs = [Y]

__all__ = ['ConstantOfShape']