from ...enum_defines import DevType
from .base import Operator, UnaryOperator, BaseLayout, OperatorAttrs, register_op


@register_op
class Relu(UnaryOperator, BaseLayout):
    pass


@register_op
class PRelu(Operator):
    def __init__(self, attrs={}):
        self.attrs = OperatorAttrs(attrs)

    def infer_tensor(self):
        assert len(self.inputs) == 2, "num of inputs in PRelu must be 2"
        X1 = self.inputs[0]
        X2 = self.inputs[1]

        shape1 = list(X1.shape)
        shape2 = list(X2.shape)
        # expand to same dim
        if len(shape1) >= len(shape2):
            shape2 = [1] * (len(shape1) - len(shape2)) + shape2
        else:
            raise AttributeError

        assert len(shape1) == len(shape2)
        shape = [1] * len(shape1)
        for i in range(len(shape1)):
            if shape2[i] == 1:
                shape[i] = shape1[i]
            elif shape1[i] == shape2[i]:
                shape[i] = shape2[i]
            else:
                raise AttributeError
        Y = X1.clone(shape=tuple(shape))
        self.outputs = [Y]


__all__ = ["Relu", "PRelu"]
