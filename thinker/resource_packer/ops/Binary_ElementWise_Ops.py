import numpy as np

from .base import *
from .._type import *

#logical
@register_op
class Greater(LogicalOperator):
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 > X2
        if is_sympy(self.outputs[0].data):
            self.outputs[0].data = np.array(self.outputs[0].data)

@register_op
class GreaterOrEqual(LogicalOperator):
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 >= X2

@register_op
class Less(LogicalOperator):
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 < X2

@register_op
class LessOrEqual(LogicalOperator):
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 <= X2

@register_op
class Equal(LogicalOperator):
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 == X2

@register_op
class NotEqual(LogicalOperator):
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 != X2

@register_op
class Or(LogicalOperator):
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 or X2

@register_op
class Xor(LogicalOperator):
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 ^ X2

#binary
@register_op
class Add(BinaryOperator):
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 + X2
        self.outputs[0].data = self.outputs[0].data.astype(X1.dtype)

@register_op
class Sub(BinaryOperator):
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 - X2
        self.outputs[0].data = self.outputs[0].data.astype(X1.dtype)

@register_op
class Mul(BinaryOperator):
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 * X2
        self.outputs[0].data = self.outputs[0].data.astype(X1.dtype)

@register_op
class Div(BinaryOperator):
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = X1 / X2
        self.outputs[0].data = self.outputs[0].data.astype(X1.dtype)

@register_op
class Pow(BinaryOperator):
    def forward(self):
        inputs = self.inputs
        X1 = inputs[0].data
        X2 = inputs[1].data
        self.outputs[0].data = pow(X1, X2)
        self.outputs[0].data = self.outputs[0].data.astype(X1.dtype)

@register_op
class And(BinaryOperator):
    pass

#multi
@register_op
class Max(MultiOperator):
    pass

@register_op
class Min(MultiOperator):
    pass

@register_op
class Sum(MultiOperator):
    pass

@register_op
class Mean(MultiOperator):
    pass

__all__ = ['Greater', 'GreaterOrEqual', 'Less', 'LessOrEqual', 'Equal', 'NotEqual', 'Or', 'Xor',\
            'Add', 'Sub', 'Mul', 'Div', 'Pow', 'And', 'Max', 'Min', 'Sum', 'Mean']