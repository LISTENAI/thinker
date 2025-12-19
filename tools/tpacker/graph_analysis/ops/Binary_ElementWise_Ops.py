import numpy as np
from typing import List
from ...resource_packer._type import *
from ...xsympy import is_sympy
from .base import BinaryOperator, LogicalOperator, MultiOperator, register_op


# Logical Operators
@register_op
class Greater(LogicalOperator):
    """Element-wise greater comparison operator.
    
    Compares two input tensors element-wise and returns a boolean tensor
    indicating where the first tensor is greater than the second.
    """
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 > X2
        if is_sympy(self.outputs[0].data):
            self.outputs[0].data = np.array(self.outputs[0].data)


@register_op
class GreaterOrEqual(LogicalOperator):
    """Element-wise greater or equal comparison operator."""
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 >= X2
        if is_sympy(self.outputs[0].data):
            self.outputs[0].data = np.array(self.outputs[0].data)


@register_op
class Less(LogicalOperator):
    """Element-wise less comparison operator."""
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 < X2
        if is_sympy(self.outputs[0].data):
            self.outputs[0].data = np.array(self.outputs[0].data)


@register_op
class LessOrEqual(LogicalOperator):
    """Element-wise less or equal comparison operator."""
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 <= X2
        if is_sympy(self.outputs[0].data):
            self.outputs[0].data = np.array(self.outputs[0].data)


@register_op
class Equal(LogicalOperator):
    """Element-wise equality comparison operator."""
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 == X2
        if is_sympy(self.outputs[0].data):
            self.outputs[0].data = np.array(self.outputs[0].data)


@register_op
class NotEqual(LogicalOperator):
    """Element-wise inequality comparison operator."""
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 != X2
        if is_sympy(self.outputs[0].data):
            self.outputs[0].data = np.array(self.outputs[0].data)


@register_op
class Or(LogicalOperator):
    """Element-wise logical OR operator."""
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 or X2
        if is_sympy(self.outputs[0].data):
            self.outputs[0].data = np.array(self.outputs[0].data)


@register_op
class Xor(LogicalOperator):
    """Element-wise logical XOR operator."""
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 ^ X2
        if is_sympy(self.outputs[0].data):
            self.outputs[0].data = np.array(self.outputs[0].data)


# Binary Operators
@register_op
class Add(BinaryOperator):
    """Element-wise addition operator."""
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 + X2
        if is_sympy(self.outputs[0].data):
            self.outputs[0].data = np.array(self.outputs[0].data)
        else:
            self.outputs[0].data = self.outputs[0].data.astype(X1.dtype)


@register_op
class Sub(BinaryOperator):
    """Element-wise subtraction operator."""
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 - X2
        if is_sympy(self.outputs[0].data):
            self.outputs[0].data = np.array(self.outputs[0].data)
        else:
            self.outputs[0].data = self.outputs[0].data.astype(X1.dtype)


@register_op
class Mul(BinaryOperator):
    """Element-wise multiplication operator."""
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 * X2
        if is_sympy(self.outputs[0].data):
            self.outputs[0].data = np.array(self.outputs[0].data)
        else:
            self.outputs[0].data = self.outputs[0].data.astype(X1.dtype)


@register_op
class Div(BinaryOperator):
    """Element-wise division operator."""
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 / X2
        if is_sympy(self.outputs[0].data):
            self.outputs[0].data = np.array(self.outputs[0].data)
        else:
            self.outputs[0].data = self.outputs[0].data.astype(X1.dtype)


@register_op
class Pow(BinaryOperator):
    """Element-wise power operator."""
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = pow(X1, X2)
        if is_sympy(self.outputs[0].data):
            self.outputs[0].data = np.array(self.outputs[0].data)
        else:
            self.outputs[0].data = self.outputs[0].data.astype(X1.dtype)


@register_op
class And(BinaryOperator):
    """Element-wise logical AND operator."""
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 and X2
        if is_sympy(self.outputs[0].data):
            self.outputs[0].data = np.array(self.outputs[0].data)


# Multi Operators
@register_op
class Max(MultiOperator):
    """Computes the maximum value across multiple inputs."""
    pass


@register_op
class Min(MultiOperator):
    """Computes the minimum value across multiple inputs."""
    pass


@register_op
class Sum(MultiOperator):
    """Computes the sum of all input values."""
    pass


@register_op
class Mean(MultiOperator):
    """Computes the mean value of all input values."""
    pass


__all__ = [
    'Greater', 'GreaterOrEqual', 'Less', 'LessOrEqual', 'Equal', 'NotEqual',
    'Or', 'Xor', 'Add', 'Sub', 'Mul', 'Div', 'Pow', 'And', 'Max', 'Min', 'Sum', 'Mean'
]