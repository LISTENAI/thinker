# Copyright (C) 2025 listenai Co.Ltd
# All rights reserved.
# Created by leifang on 2022.09.31

from .GraphEntry import GraphEntry
from ..enum_defines import TensorType

class ConstantEntry(GraphEntry):
    """Entry for constant tensors."""
    def __init__(self, name, tensor):
        super().__init__(name, tensor=tensor, is_constant_tensor=True)

class ScalarEntry(GraphEntry):
    """Entry for scalar values."""
    def __init__(self, name, tensor):
        super().__init__(name, tensor=tensor, is_constant_tensor=True)
        self.tensor.shape = ()

class InputEntry(GraphEntry):
    """Entry for input tensors."""
    def __init__(self, name, tensor):
        super().__init__(name, tensor=tensor, is_constant_tensor=False, tensor_type=TensorType.Input)

class OutputEntry(GraphEntry):
    """Entry for output tensors."""
    def __init__(self, name, tensor):
        super().__init__(name, tensor=tensor, is_constant_tensor=False, tensor_type=TensorType.Output)

class EmptyEntry(GraphEntry):
    """Entry for empty tensors."""
    def __init__(self):
        super().__init__(name=f"null_{id(self)}")
        self.set_empty()

__all__ = ["ConstantEntry", "ScalarEntry", "InputEntry", "OutputEntry", "EmptyEntry"]