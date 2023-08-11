import math
import numpy as np
from typing import Any, Dict, Optional

from ..utils import QuantType
from .Operator import Operator, OperatorAttrs


class UnaryOperator(Operator):
    def __init__(self, attrs={}):
        self.attrs = OperatorAttrs(attrs)

    def infer_tensor(self):
        assert len(self.inputs) == 1
        X = self.inputs[0]
        Y = X.clone()
        self.outputs = [Y]
        if all([x.has_data() for x in self.inputs]):
            self.forward()

    def forward(self):
        raise NotImplementedError


class BinaryOperator(Operator):
    def __init__(self, attrs={}):
        self.attrs = OperatorAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) == 2
        X1 = inputs[0]
        X2 = inputs[1]

        shape1 = list(X1.shape)
        shape2 = list(X2.shape)

        # expand to same dim
        if len(shape1) > len(shape2):
            shape2 = [1] * (len(shape1) - len(shape2)) + shape2
        else:
            shape1 = [1] * (len(shape2) - len(shape1)) + shape1

        assert len(shape1) == len(shape2)
        shape = [1] * len(shape1)
        for i in range(len(shape1)):
            if shape1[i] == 1:
                shape[i] = shape2[i]
            elif shape2[i] == 1:
                shape[i] = shape1[i]
            elif shape1[i] == shape2[i]:
                shape[i] = shape2[i]
            else:
                raise AttributeError
        Y = X1.clone(shape=tuple(shape))
        self.inputs = inputs
        self.outputs = [Y]
        if all([x.has_data() for x in inputs]):
            self.forward()

    def forward(self):
        raise NotImplementedError

class LogicalOperator(BinaryOperator):
    def infer_tensor(self):
        super().infer_tensor()
        self.outputs[0].dtype = np.dtype(np.int8)

class MultiOperator(Operator):
    def __init__(self, attrs={}):
        self.attrs = OperatorAttrs(attrs)

class iqUnaryOperatorAttrs(OperatorAttrs):
    def normalize(self):
        attrs = self.attrs
        quant_type = attrs.get("platform_quant", "normal_quant")
        attrs["quant_type"] = QuantType.from_str(quant_type)


class iqBinaryOperatorAttrs(OperatorAttrs):
    def __init__(self, attrs: Optional[Dict[str, Any]] = {}) -> None:
        super().__init__(attrs, "iqBinaryAttrs")

    def normalize(self):
        attrs = self.attrs
        quant_type = attrs.get("platform_quant", "normal_quant")
        attrs["quant_type"] = QuantType.from_str(quant_type)


class iqUnaryOperator(UnaryOperator):
    def __init__(self, attrs={}):
        self.attrs = iqUnaryOperatorAttrs(attrs)

    def infer_tensor(self):
        assert len(self.inputs) == 1
        X = self.inputs[0]
        # Y = X.clone(scale=self.attrs["scale_o"])
        # self.outputs = [Y]

        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001
        if self.inputs[0].scale != -1:
            assert X.scale == int(temp)
        else:
            self.inputs[0].scale = int(temp)

        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001

        Y = X.clone(scale=int(temp))
        self.outputs = [Y]

        if all([x.has_data() for x in self.inputs]):
            self.forward()


class iqBinaryOperator(BinaryOperator):
    def __init__(self, attrs={}):
        self.attrs = iqBinaryOperatorAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) == 2
        X1 = inputs[0]
        X2 = inputs[1]

        shape1 = list(X1.shape)
        shape2 = list(X2.shape)

        # expand to same dim
        if len(shape1) > len(shape2):
            shape2 = [1] * (len(shape1) - len(shape2)) + shape2
        else:
            shape1 = [1] * (len(shape2) - len(shape1)) + shape1

        assert len(shape1) == len(shape2)
        shape = [1] * len(shape1)
        for i in range(len(shape1)):
            if shape1[i] == 1:
                shape[i] = shape2[i]
            elif shape2[i] == 1:
                shape[i] = shape1[i]
            elif shape1[i] == shape2[i]:
                shape[i] = shape2[i]
            else:
                raise AttributeError

        scale_x = self.attrs.get('scale_x', 1.0)
        temp = math.log(scale_x, 2)
        assert(abs(temp - int(temp)) < 0.000001)
        if self.inputs[0].scale != -1:
            assert self.inputs[0].scale == int(temp)
        else:
            self.inputs[0].scale = int(temp)

        scale_y = self.attrs.get('scale_y', 1.0)
        temp = math.log(scale_y, 2)
        assert(abs(temp - int(temp)) < 0.000001)
        self.inputs[1].scale = temp

        scale_o = self.attrs.get("scale_o", 1.0)
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001

        Y = X1.clone(shape=tuple(shape), scale=temp)
        self.inputs = inputs
        self.outputs = [Y]
        if all([x.has_data() for x in inputs]):
            self.forward()

__all__ = [
    "UnaryOperator",
    "BinaryOperator",
    "LogicalOperator",
    "MultiOperator",
    "iqUnaryOperatorAttrs",
    "iqBinaryOperatorAttrs",
    "iqUnaryOperator",
    "iqBinaryOperator",
]
