from functools import reduce

from .base import Operator, OperatorAttrs, register_op


def _prod(L):
    return reduce(lambda x, y: x * y, L)


@register_op
class Reshape(Operator):
    def __init__(self, attrs={}):
        self.attrs = OperatorAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs

        X = self.inputs[0]
        tShape = list(X.shape)
        yshape = list(inputs[1].data)

        i_1 = None
        for i, s in enumerate(yshape):
            if s == 0:  # copy shape 0
                yshape[i] = tShape[i]
            elif s == -1:  # record shape -1 to i_1
                assert i_1 is None, "Only one -1 accept in reshape"
                yshape[i] = 1
                i_1 = i

        xsize = _prod(tShape)
        ysize = _prod(yshape)
        if i_1 is not None:
            yshape[i_1] = int(xsize / ysize)

        Y = X.clone(shape=tuple(yshape))
        if X.has_data():
            Y.data = X.data.reshape(yshape)
        self.outputs = [Y]

    def is_inplace(self):
        return True


__all__ = ["Reshape"]
