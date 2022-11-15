from .GraphEntry import GraphEntry


class GraphNode(object):
    def __init__(self, op, name, attrs={}, inputs=[], outputs=[]):
        self.name = name  # name with scope
        self.op_type = op
        self.dev_type = None
        self.attrs = attrs.copy()
        self.inputs: list[GraphEntry] = [x for x in inputs]
        self.outputs: list[GraphEntry] = [x for x in outputs]
        self.index = -1
        self.op = None

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        if isinstance(other, GraphNode):
            return self.name == other.name
        return False

    def __str__(self):
        return "{}('{}'): attrs={}".format(self.op_type, self.name, self.attrs)

    def __repr__(self):
        return str(self)

    def ref_input_nodes(self):
        nodes = []
        for entry in self.inputs:
            n = entry.src_node
            if n is not None:
                nodes.append(n)
        return nodes

    def ref_output_nodes(self):
        nodes = []
        for entry in self.outputs:
            for n in entry.dst_nodes:
                nodes.append(n)
        return nodes

    def get_input_index(self, entry: GraphEntry):
        return self.inputs.index(entry)

    def get_output_index(self, entry: GraphEntry):
        return self.outputs.index(entry)

    def copy(self, src: "GraphNode") -> "GraphNode":
        self.name = src.name
        self.op_type = src.op_type
        self.dev_type = src.dev_type
        self.attrs = src.attrs.copy()
        self.outputs = [output.clone() for output in src.outputs]
        self.inputs = [input.clone() for input in src.inputs]
        self.index = src.index
        self.op = src.op
        return self

    def clone(self) -> "GraphNode":
        return GraphNode(None, None).copy(self)

    def create_op(self) -> None:
        inputs = [input.tensor for input in self.inputs]
        outputs = [output.tensor for output in self.outputs]
        from ..resource_packer.ops.base import create_operator

        self.op = create_operator(self.op_type, self.attrs, inputs, outputs)


__all__ = ["GraphNode"]
