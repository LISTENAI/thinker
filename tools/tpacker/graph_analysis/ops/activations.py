import numpy as np

from ...graph import Tensor
from ...enum_defines import DevType, MemType
from .base import UnaryOperator, BaseLayout, register_op


@register_op
class Relu(UnaryOperator, BaseLayout):
    """Rectified Linear Unit (ReLU) activation function."""
    
    def get_workspace(self):
        """Calculate the required workspace size."""
        input = self.inputs[0]
        output = self.outputs[0]
        workspace_size = 0

        if input.mem_type != MemType.SHARE_MEM:
            workspace_size += input.nbytes
        if output.mem_type != MemType.SHARE_MEM:
            workspace_size += output.nbytes

        workspace_size = min(65536, workspace_size)
        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]


@register_op
class PRelu(UnaryOperator, BaseLayout):
    """Parametric Rectified Linear Unit (PReLU) activation function."""
    pass


@register_op
class Sigmoid(UnaryOperator, BaseLayout):
    """Sigmoid activation function."""
    pass


@register_op
class Tanh(UnaryOperator, BaseLayout):
    """Hyperbolic Tangent (Tanh) activation function."""
    pass


__all__ = ["Relu", "PRelu", "Sigmoid", "Tanh"]