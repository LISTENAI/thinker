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
        Y = X.clone(scale=self.attrs["scale_o"])
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
        Y = X1.clone(shape=tuple(shape), scale=self.attrs["scale_o"])
        self.inputs = inputs
        self.outputs = [Y]
        if all([x.has_data() for x in inputs]):
            self.forward()


__all__ = [
    "UnaryOperator",
    "BinaryOperator",
    "iqUnaryOperatorAttrs",
    "iqBinaryOperatorAttrs",
    "iqUnaryOperator",
    "iqBinaryOperator",
]
