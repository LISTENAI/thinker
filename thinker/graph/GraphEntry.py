import numpy as np
from .Tensor import Tensor
from ..enum_defines import TensorType


class GraphEntry(object):
    def __init__(
        self,
        name="",
        tensor=None,
        is_constant_tensor=False,
        tensor_type=TensorType.Normal,
    ):
        self.tensor = Tensor(shape=(), dtype=np.float32) if tensor is None else tensor
        self.name = name
        self.is_constant_tensor = is_constant_tensor
        self.tensor_type = tensor_type
        self.dst_nodes = []
        self.src_node = None
        self.index = -1

    @property
    def layout(self):
        return self.tensor.layout

    @layout.setter
    def layout(self, value):
        self.tensor.layout = value

    @property
    def data(self):
        return self.tensor.data

    @data.setter
    def data(self, value):
        self.tensor.data = value

    def is_constant(self):
        return self.is_constant_tensor

    def set2_constant(self):
        if not self.tensor.has_data():
            raise ValueError("Can't set tensor_type to constant for empty data entry.")
        self.is_constant_tensor = True

    def is_graph_input(self):
        return self.is_constant_tensor == False and self.tensor_type == TensorType.Input

    def set2_graph_input(self):
        self.is_constant_tensor = False
        self.tensor_type = TensorType.Input

    def is_graph_output(self):
        return (
            self.is_constant_tensor == False and self.tensor_type == TensorType.Output
        )

    def set2_graph_output(self):
        self.is_constant_tensor = False
        self.tensor_type = TensorType.Output

    def set2_graph_normal(self):
        self.is_constant_tensor = False
        self.tensor_type = TensorType.Normal

    def set2_emptry(self):
        self.tensor.data = np.array([])
        self.tensor.shape = ()
        self.is_constant_tensor = True
        self.tensor_type = TensorType.Emptry

    def is_emptry(self):
        return self.is_constant_tensor == True and self.tensor_type == TensorType.Emptry

    def add_dst_node(self, node):
        self.dst_nodes.append(node)

    def copy(self, src: "GraphEntry") -> "GraphEntry":
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
        return GraphEntry().copy(self)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        if isinstance(other, GraphEntry):
            return self.name == other.name
        return False

    def __str__(self):
        return "name={}, tensor={}, is_constant={}".format(
            self.name, self.tensor, self.is_constant_tensor
        )

    def __repr__(self):
        return self.__str__()


__all__ = ["GraphEntry"]
