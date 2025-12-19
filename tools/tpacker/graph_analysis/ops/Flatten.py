from .base import Operator, OperatorAttrs, register_op

@register_op
class Flatten(Operator):
    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor by flattening the input tensor."""
        inputs = self.inputs
        assert len(inputs) == 1, "Flatten operator must have exactly one input"

        X = inputs[0]
        shape = list(X.shape)
        size = 1
        for dim in shape[1:]:
            size *= dim
        Y = X.clone(shape=(shape[0], size))
        self.outputs = [Y]

__all__ = ["Flatten"]