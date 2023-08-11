import numpy as np

from .base import Operator, OperatorAttrs, register_op

@register_op
class Expand(Operator):
    def __init__(self, attrs = {}):
        self.attrs = OperatorAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        X = self.inputs[0]
        shape1 = list(X.shape)
        shape2 = list(inputs[1].data)

        if len(shape1) > len(shape2):
            shape2 = [1]*(len(shape1)-len(shape2)) + shape2
        else:
            shape1 = [1]*(len(shape2)-len(shape1)) + shape1
      
        assert len(shape1) == len(shape2)
        shape = [1]*len(shape1)
        for i in range(len(shape1)):
            if shape1[i] == 1:
                shape[i] = shape2[i]
            elif shape2[i] == 1:
                shape[i] = shape1[i]
            elif shape1[i] == shape2[i]:
                shape[i] = shape2[i]
            else:
                raise AttributeError
        Y = X.clone(shape=tuple(shape),scale=X.scale)
        
        if X.has_data():
            input_shape =  [1]*(len(shape)-len(X.data.shape)) + list(X.data.shape)
            tile_shape = np.array(shape) // np.array(input_shape)

            Y.data = np.tile(X.data,(tile_shape))
        self.outputs = [Y]

__all__ = ['Expand']