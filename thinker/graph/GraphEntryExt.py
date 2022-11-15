from .GraphEntry import GraphEntry
from ..enum_defines import TensorType


class ConstantEntry(GraphEntry):
    def __init__(self, name, tensor):
        super().__init__(name, tensor=tensor, is_constant_tensor=True)


class ScalarEntry(GraphEntry):
    def __init__(self, name, tensor):
        super().__init__(name, tensor=tensor, is_constant_tensor=True)
        self.tensor.shape = ()


class InputEntry(GraphEntry):
    def __init__(self, name, tensor):
        super().__init__(
            name, tensor=tensor, is_constant_tensor=False, tensor_type=TensorType.Input
        )


class OutputEntry(GraphEntry):
    def __init__(self, name, tensor):
        super().__init__(
            name, tensor=tensor, is_constant_tensor=False, tensor_type=TensorType.Output
        )


class EmptyEntry(GraphEntry):
    def __init__(self):
        super().__init__(name="null" + str(id(self)))
        self.set2_emptry()


__all__ = ["ConstantEntry", "ScalarEntry", "InputEntry", "OutputEntry", "EmptyEntry"]
