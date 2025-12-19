# Copyright (C) 2025 listenai Co.Ltd
# All rights reserved.
# Created by leifang on 2022.09.31

from .GraphEntry import GraphEntry

class GraphNode(object):
    """Node class representing operations in the graph."""
    
    def __init__(self, op_type, name, attrs={}, inputs=[], outputs=[]):
        """
        Initialize a GraphNode object.
        
        Args:
            op_type (str): Type of the operation.
            name (str): Name of the node.
            attrs (dict): Attributes of the node.
            inputs (list): Input entries to the node.
            outputs (list): Output entries from the node.
        """
        self.name = name
        self.op = None
        self.op_type = op_type
        self.dev_type = None
        self.index = -1
        self.attrs = attrs.copy()
        self.inputs = [x for x in inputs]
        self.outputs = [x for x in outputs]

    def __hash__(self):
        """Hash based on the node name."""
        return hash(self.name)

    def __eq__(self, other):
        """Equality check based on name."""
        if isinstance(other, str):
            return self.name == other
        if isinstance(other, GraphNode):
            return self.name == other.name
        return False

    def __str__(self):
        """String representation of the node."""
        return f"{self.op_type}('{self.name}'): attrs={self.attrs}"

    def __repr__(self):
        """Representation of the node."""
        return str(self)

    def ref_input_nodes(self):
        """Get reference input nodes."""
        nodes = []
        for entry in self.inputs:
            n = entry.src_node
            if n is not None:
                nodes.append(n)
        return nodes

    def ref_output_nodes(self):
        """Get reference output nodes."""
        nodes = []
        for entry in self.outputs:
            for n in entry.dst_nodes:
                nodes.append(n)
        return nodes

    def get_input_index(self, entry: GraphEntry):
        """Get the index of an input entry."""
        return self.inputs.index(entry)

    def get_output_index(self, entry: GraphEntry):
        """Get the index of an output entry."""
        return self.outputs.index(entry)

    def copy(self, src: "GraphNode") -> "GraphNode":
        """Copy attributes from another GraphNode."""
        self.name = src.name
        self.op = src.op
        self.index = src.index
        self.op_type = src.op_type
        self.dev_type = src.dev_type
        self.attrs = src.attrs.copy()
        self.outputs = [entry.clone() for entry in src.outputs]
        self.inputs = [entry.clone() for entry in src.inputs]
        return self

    def clone(self) -> "GraphNode":
        """Clone the GraphNode object."""
        return GraphNode(None, None).copy(self)

    def create_op(self) -> None:
        """Create an operator for the node."""
        from ..graph_analysis.ops.base import create_operator
        inputs = [entry.tensor for entry in self.inputs]
        outputs = [entry.tensor for entry in self.outputs]
        self.op = create_operator(self.op_type, self.attrs, inputs, outputs)

__all__ = ["GraphNode"]