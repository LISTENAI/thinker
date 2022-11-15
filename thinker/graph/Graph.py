from ..graph import Graph
from .GraphEntry import GraphEntry
from .GraphNode import GraphNode
from .indexed import IndexedOrderedDict as IndexedDict


class Graph(object):
    def __init__(self):
        self.nodes: IndexedDict = IndexedDict()
        self.entries: IndexedDict = IndexedDict()
        self.inputs: list[GraphEntry] = []
        self.outputs: list[GraphEntry] = []

        # attrs
        self.performance = 0
        self.dynamic_args_max = {}

    def add_node(self, node: GraphNode):
        if node.name in self.nodes:
            raise KeyError(f"Node named with {node.name} already exists in graph.")
        self.nodes[node.name] = node

    def add_entry(self, e: GraphEntry):
        if e.name in self.entries:
            raise KeyError(f"Tensor named with {e.name} already exits in graph.")
        self.entries[e.name] = e

    # copy
    def copy_attrs(self, src: "Graph") -> "Graph":
        self.performance = src.performance
        self.dynamic_args_max = src.dynamic_args_max

    def copy(self, src: "Graph") -> "Graph":
        self.copy_attrs(src)

        for entry in src.entries.values():
            self.entries[entry.name] = entry.clone()

        for node in src.nodes.values():
            new_node = node.clone()
            for i, x in enumerate(node.inputs):
                new_node.inputs[i] = self.entries[x.name]
            for i, x in enumerate(node.outputs):
                new_node.outputs[i] = self.entries[x.name]
            self.nodes[new_node.name] = new_node

        for input in src.inputs:
            self.inputs.append(self.entries[input.name])

        for output in src.outputs:
            self.outputs.append(self.entries[output.name])
        return self

    def clone(self, is_update=False) -> "Graph":
        graph = Graph().copy(self)
        if is_update == True:
            graph.update()
        return graph

    def init_tensor(self):
        for node in self.nodes.values():
            inputs = [self.entries[x].tensor for x in node.inputs]
            if node.op is None:
                node.create_op()
                if node.op is None:
                    raise AttributeError(
                        'not support Operator: "{}" yet'.format(node.op_type)
                    )
            node.op.set_input(inputs)
            node.op.infer_tensor()

            outputs = node.op.get_output()
            for i, x in enumerate(node.outputs):
                self.entries[x].tensor.shape = outputs[i].shape
                self.entries[x].tensor.data = outputs[i].data
                self.entries[x].tensor.scale = outputs[i].scale
                self.entries[x].tensor.dtype = outputs[i].dtype
                self.entries[x].tensor.layout = outputs[i].layout

    def _del_useless_entry(self):
        use_list = set()
        for x in self.inputs:
            use_list.add(x)
        for x in self.outputs:
            use_list.add(x)
        for node in self.nodes.values():
            for e in node.inputs:
                use_list.add(e.name)
            for e in node.outputs:
                use_list.add(e.name)

        useless_list = []
        for e in self.entries.values():
            if e.name not in use_list:
                useless_list.append(e.name)

        for x in useless_list:
            del self.entries[x]

    def _update_entries(self):
        for e in self.entries.values():
            e.src_node = None
            e.dst_nodes = []

        for node in self.nodes.values():
            for input in node.inputs:
                self.entries[input].add_dst_node(node)
            for output in node.outputs:
                self.entries[output].src_node = node

        for i, e in enumerate(self.entries.values()):
            e.index = i

    def _update_nodes(self):
        self._sort_nodes()

        for i, input in enumerate(self.inputs):
            self.inputs[i] = self.entries[input]
        for i, output in enumerate(self.outputs):
            self.outputs[i] = self.entries[output]

        for i, node in enumerate(self.nodes.values()):
            node.inputs = [self.entries[x] for x in node.inputs]
            node.outputs = [self.entries[x] for x in node.outputs]
            node.index = i

    def _sort_nodes(self):
        # self._update_entries()
        new_nodes = IndexedDict()

        def recurse_sort_nodes(name_list):
            for x in name_list:
                e = self.entries[x]
                if e.is_graph_input() or e.is_constant():
                    continue  # inputs or params

                node = e.src_node
                if node == None:
                    raise ("node is null")
                if node.name not in new_nodes:
                    recurse_sort_nodes(node.inputs)
                    new_nodes[node.name] = node

        recurse_sort_nodes([x for x in self.outputs])
        self.nodes = new_nodes

    def _constant_fold(self):
        for e in self.entries.values():
            if e.tensor.has_data():
                e.set2_constant()

    def _update_op_tensors(self):
        for node in self.nodes.values():
            if node.op == None:
                continue
            node.op.inputs = [input.tensor for input in node.inputs]
            node.op.outputs = [output.tensor for output in node.outputs]

    def pack_params(self):
        for node in self.nodes.values():
            if node.op == None:
                continue
            node.op.pack_params(node.dev_type)

    def update(self):
        self._constant_fold()
        self._update_entries()
        self._update_nodes()
        self._del_useless_entry()
        self._update_op_tensors()


__all__ = ["Graph"]
