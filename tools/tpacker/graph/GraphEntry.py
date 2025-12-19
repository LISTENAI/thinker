# Copyright (C) 2025 listenai Co.Ltd
# All rights reserved.
# Created by leifang on 2022.09.31

import numpy as np
from .Tensor import Tensor
from ..enum_defines import TensorType

class GraphEntry(object):
    """Base class for graph entries representing tensors in the graph."""
    
    def __init__(self, name="", tensor=None, is_constant_tensor=False, tensor_type=TensorType.Normal):
        """
        Initialize a GraphEntry object.
        
        Args:
            name (str): Name of the entry.
            tensor (Tensor): Tensor associated with the entry.
            is_constant_tensor (bool): Whether the tensor is constant.
            tensor_type (TensorType): Type of the tensor.
        """
        self.name = name
        self.dst_nodes = []
        self.src_node = None
        self.index = -1
        self.tensor_type = tensor_type
        self.is_constant_tensor = is_constant_tensor
        self.tensor = Tensor(shape=(), dtype=np.float32) if tensor is None else tensor

    @property
    def layout(self):
        """Get the layout of the tensor."""
        return self.tensor.layout

    @layout.setter
    def layout(self, value):
        """Set the layout of the tensor."""
        self.tensor.layout = value

    @property
    def data(self):
        """Get the data of the tensor."""
        return self.tensor.data

    @data.setter
    def data(self, value):
        """Set the data of the tensor."""
        self.tensor.data = value

    def is_constant(self):
        """Check if the tensor is constant."""
        return self.is_constant_tensor

    def set_constant(self):
        """Set the tensor as constant."""
        if not self.tensor.has_data():
            raise ValueError("Cannot set tensor_type to constant for empty data entry.")
        self.is_constant_tensor = True

    def is_graph_input(self):
        """Check if the entry is a graph input."""
        return not self.is_constant_tensor and self.tensor_type == TensorType.Input

    def set_graph_input(self):
        """Set the entry as a graph input."""
        self.is_constant_tensor = False
        self.tensor_type = TensorType.Input

    def is_graph_output(self):
        """Check if the entry is a graph output."""
        return not self.is_constant_tensor and self.tensor_type == TensorType.Output

    def set_graph_output(self):
        """Set the entry as a graph output."""
        self.is_constant_tensor = False
        self.tensor_type = TensorType.Output

    def set_graph_normal(self):
        """Set the entry as a normal tensor."""
        self.is_constant_tensor = False
        self.tensor_type = TensorType.Normal

    def set_empty(self):
        """Set the entry as an empty tensor."""
        self.tensor.data = np.array([])
        self.tensor.shape = ()
        self.is_constant_tensor = True
        self.tensor_type = TensorType.Empty

    def is_empty(self):
        """Check if the entry is an empty tensor."""
        return self.is_constant_tensor and self.tensor_type == TensorType.Empty

    def add_dst_node(self, node):
        """Add a destination node to the entry."""
        self.dst_nodes.append(node)

    def copy(self, src: "GraphEntry") -> "GraphEntry":
        """Copy attributes from another GraphEntry."""
        self.name = src.name
        self.tensor = src.tensor.clone()
        if isinstance(src.tensor.data, np.ndarray):
            self.tensor.data = np.array(src.tensor.data)
        else:
            self.tensor.data = src.tensor.data
        self.is_constant_tensor = src.is_constant_tensor
        self.tensor_type = src.tensor_type
        self.index = src.index
        return self

    def clone(self) -> "GraphEntry":
        """Clone the GraphEntry object."""
        return GraphEntry().copy(self)

    def __hash__(self):
        """Hash based on the entry name."""
        return hash(self.name)

    def __eq__(self, other):
        """Equality check based on name."""
        if isinstance(other, str):
            return self.name == other
        if isinstance(other, GraphEntry):
            return self.name == other.name
        return False

    def __str__(self):
        """String representation of the entry."""
        return f"name={self.name}, tensor={self.tensor}, is_constant={self.is_constant_tensor}"

    def __repr__(self):
        """Representation of the entry."""
        return self.__str__()

__all__ = ["GraphEntry"]