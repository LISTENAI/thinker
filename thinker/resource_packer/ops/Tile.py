from .base import Operator, OperatorAttrs, register_op

@register_op
class Tile(Operator):
    def __init__(self, attrs={}):
        self.attrs = OperatorAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs)==2
        X=inputs[0]
        repeats=inputs[1]
        yshape = X.shape * repeats.data
        for i, s in enumerate(yshape):
            yshape[i] = int(yshape[i])
        Y = X.clone(shape=tuple(yshape))
        self.outputs = [Y]