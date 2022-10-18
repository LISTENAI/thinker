from .base import Operator, OperatorAttrs, register_op


@register_op
class Flatten(Operator):
    def __init__(self, attrs={}):
        self.attrs = OperatorAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) == 1
        X = inputs[0]

        shape = list(X.shape)
        size = 1
        for i in range(1, len(shape)):
            size *= shape[i]
        Y_shape = (shape[0], size)
        Y = X.clone(shape=tuple(Y_shape))
        self.outputs = [Y]


__all__ = ["Flatten"]
