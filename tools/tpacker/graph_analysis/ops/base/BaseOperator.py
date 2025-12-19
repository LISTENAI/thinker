import math
import numpy as np
from typing import Any, Dict, Optional

from ....xsympy import is_sympy
from .Operator import Operator, OperatorAttrs
from ..utils import QuantType, calc_expr, RoundMethod


class UnaryOperator(Operator):
    """Base class for unary operators."""
    def __init__(self, attrs: Optional[Dict[str, Any]] = None):
        self.attrs = OperatorAttrs(attrs or {})

    def infer_tensor(self, dynamic_shape: Dict[str, int]):
        """Infer output tensor shape and data."""
        assert len(self.inputs) == 1, "Unary operator must have exactly one input"
        X = self.inputs[0]
        Y = X.clone()
        self.outputs = [Y]
        if all(x.has_data() for x in self.inputs):
            self.forward()

    def flops_counter(self, dynamic_shape: Dict[str, int]) -> int:
        """Calculate the number of floating-point operations."""
        return 0

    def forward(self):
        """Perform the forward computation."""
        raise NotImplementedError


class BinaryOperator(Operator):
    """Base class for binary operators."""
    def __init__(self, attrs: Optional[Dict[str, Any]] = None):
        self.attrs = OperatorAttrs(attrs or {})

    def infer_tensor(self, dynamic_shape: Dict[str, int]):
        """Infer output tensor shape and data."""
        assert len(self.inputs) == 2, "Binary operator must have exactly two inputs"
        X1, X2 = self.inputs

        shape1 = list(X1.shape)
        shape2 = list(X2.shape)

        # Expand to the same dimension
        if len(shape1) > len(shape2):
            shape2 = [1] * (len(shape1) - len(shape2)) + shape2
        else:
            shape1 = [1] * (len(shape2) - len(shape1)) + shape1

        assert len(shape1) == len(shape2), "Input shapes must have the same dimensions after expansion"

        shape = []
        for s1, s2 in zip(shape1, shape2):
            if s1 == 1:
                shape.append(s2)
            elif s2 == 1:
                shape.append(s1)
            elif s1 == s2:
                shape.append(s1)
            elif is_sympy(s1) and is_sympy(s2):
                s1_val = calc_expr(str(s1), dynamic_shape)
                s2_val = calc_expr(str(s2), dynamic_shape)
                assert s1_val == s2_val, "Dynamic shapes must match"
                shape.append(s1_val)
            else:
                raise AttributeError("Incompatible shapes")

        Y = X1.clone(shape=tuple(shape))
        self.outputs = [Y]
        if all(x.has_data() for x in self.inputs):
            self.forward()

    def flops_counter(self, dynamic_shape: Dict[str, int]) -> int:
        """Calculate the number of floating-point operations."""
        X = self.inputs[0]
        Y = self.outputs[0]
        xshape = list(X.shape)
        yshape = list(Y.shape)

        # Convert symbolic expressions to actual values
        for i, s in enumerate(xshape):
            if is_sympy(s):
                xshape[i] = calc_expr(str(s), dynamic_shape)
        for i, s in enumerate(yshape):
            if is_sympy(s):
                yshape[i] = calc_expr(str(s), dynamic_shape)

        output_dims = yshape[1:]
        return int(np.prod(output_dims))

    def forward(self):
        """Perform the forward computation."""
        raise NotImplementedError


class LogicalOperator(BinaryOperator):
    """Base class for logical operators."""
    def __init__(self, attrs: Optional[Dict[str, Any]] = None):
        self.attrs = OperatorAttrs(attrs or {})

    def infer_tensor(self, dynamic_shape: Dict[str, int]):
        """Infer output tensor shape and data."""
        super().infer_tensor(dynamic_shape)
        self.outputs[0].dtype = np.dtype(np.int8)


class MultiOperator(Operator):
    """Base class for multi-input operators."""
    def __init__(self, attrs: Optional[Dict[str, Any]] = None):
        self.attrs = OperatorAttrs(attrs or {})


class iqUnaryOperatorAttrs(OperatorAttrs):
    """Attributes for iqUnaryOperator."""
    def normalize(self):
        """Normalize operator attributes."""
        platform = self.attrs.get("platform", "venus")
        if platform in {"arcs", "venusA"}:
            quant_type = self.attrs.get("quant_mode", "floor_add")
            self.attrs["quant_mode"] = RoundMethod.from_str(quant_type)
        elif platform == "venus":
            quant_type = QuantType.from_str(self.attrs.get("platform_quant"))
            self.attrs["quant_mode"] = quant_type


class iqBinaryOperatorAttrs(OperatorAttrs):
    """Attributes for iqBinaryOperator."""
    def __init__(self, attrs: Optional[Dict[str, Any]] = None):
        super().__init__(attrs or {}, "iqBinaryAttrs")

    def normalize(self):
        """Normalize operator attributes."""
        platform = self.attrs.get("platform", "venus")
        if platform in {"arcs", "venusA"}:
            quant_type = self.attrs.get("quant_mode", "floor_add")
            self.attrs["quant_mode"] = RoundMethod.from_str(quant_type)
        elif platform == "venus":
            quant_type = QuantType.from_str(self.attrs.get("platform_quant"))
            self.attrs["quant_mode"] = quant_type


class iqUnaryOperator(UnaryOperator):
    """Intelligence-Quantized unary operator."""
    def __init__(self, attrs: Optional[Dict[str, Any]] = None):
        self.attrs = iqUnaryOperatorAttrs(attrs or {})

    def infer_tensor(self, dynamic_shape: Dict[str, int]):
        """Infer output tensor shape and data."""
        assert len(self.inputs) == 1, "iqUnaryOperator must have exactly one input"
        X = self.inputs[0]

        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 1e-6, "scale_x must be a power of 2"
        if X.scale != -1:
            assert X.scale == int(temp), "Input scale mismatch"
        else:
            X.scale = int(temp)

        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 1e-6, "scale_o must be a power of 2"

        Y = X.clone(scale=int(temp))
        self.outputs = [Y]
        if all(x.has_data() for x in self.inputs):
            self.forward()


class iqBinaryOperator(BinaryOperator):
    """Intelligence-Quantized binary operator."""
    def __init__(self, attrs: Optional[Dict[str, Any]] = None):
        self.attrs = iqBinaryOperatorAttrs(attrs or {})

    def infer_tensor(self, dynamic_shape: Dict[str, int]):
        """Infer output tensor shape and data."""
        inputs = self.inputs
        assert len(inputs) == 2, "iqBinaryOperator must have exactly two inputs"
        X1, X2 = inputs

        shape1 = list(X1.shape)
        shape2 = list(X2.shape)

        # Expand to the same dimension
        if len(shape1) > len(shape2):
            shape2 = [1] * (len(shape1) - len(shape2)) + shape2
        else:
            shape1 = [1] * (len(shape2) - len(shape1)) + shape1

        assert len(shape1) == len(shape2), "Input shapes must have the same dimensions after expansion"

        shape = []
        for s1, s2 in zip(shape1, shape2):
            if s1 == 1:
                shape.append(s2)
            elif s2 == 1:
                shape.append(s1)
            elif s1 == s2:
                shape.append(s1)
            elif is_sympy(s1) and is_sympy(s2):
                s1_val = calc_expr(str(s1), dynamic_shape)
                s2_val = calc_expr(str(s2), dynamic_shape)
                assert s1_val == s2_val, "Dynamic shapes must match"
                shape.append(s1_val)
            else:
                raise AttributeError("Incompatible shapes")

        scale_x = self.attrs.get('scale_x', 1.0)
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 1e-6, "scale_x must be a power of 2"
        if self.inputs[0].scale != -1:
            assert self.inputs[0].scale == int(temp), "Input scale mismatch"
        else:
            self.inputs[0].scale = int(temp)

        scale_y = self.attrs.get('scale_y', 1.0)
        temp = math.log(scale_y, 2)
        assert abs(temp - int(temp)) < 1e-6, "scale_y must be a power of 2"
        self.inputs[1].scale = int(temp)

        scale_o = self.attrs.get("scale_o", 1.0)
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 1e-6, "scale_o must be a power of 2"

        Y = X1.clone(shape=tuple(shape), scale=temp)
        self.outputs = [Y]
        if all(x.has_data() for x in inputs):
            self.forward()


__all__ = [
    "UnaryOperator", "BinaryOperator", "LogicalOperator", "MultiOperator",
    "iqUnaryOperatorAttrs", "iqBinaryOperatorAttrs", "iqUnaryOperator", "iqBinaryOperator"
]