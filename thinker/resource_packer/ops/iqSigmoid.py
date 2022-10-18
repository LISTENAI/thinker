import numpy as np
import math
from .utils import QuantType
from .base import Operator, OperatorAttrs, register_op, OperatorAttrs


@register_op
class iqSigmoid(Operator):
    def infer_tensor(self):
        assert len(self.inputs) == 1
        X = self.inputs[0]
        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001
        assert X.scale == int(temp)

        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001

        Y = X.clone(scale=int(temp))
        self.outputs = [Y]
        Y.dtype = np.dtype("int8")
        if all([x.has_data() for x in self.inputs]):
            self.forward()


__all__ = ["iqSigmoid"]
