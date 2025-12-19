# Copyright (C) 2025 listenai Co.Ltd
# All rights reserved.
# Created by leifang on 2022.09.31

import sympy
import numpy as np

from ..xsympy import is_sympy
from .GraphEntry import GraphEntry
from .GraphNode import GraphNode
from .indexed import IndexedOrderedDict as IndexedDict

class Graph(object):
    """Main graph class representing a computational graph."""
    
    def __init__(self):
        """Initialize an empty graph."""
        self.name = str
        self.nodes: IndexedDict = IndexedDict()
        self.entries: IndexedDict = IndexedDict()
        self.inputs: list[GraphEntry] = []
        self.outputs: list[GraphEntry] = []
        
        # Graph attributes
        self.performance = 0
        self.dynamic_args_max = {}
        self.dynamic_args_opt = {}
        self.dynamic_shape = {}
        self.platform = None

    def add_node(self, node: GraphNode):
        """Add a node to the graph."""
        if node.name in self.nodes:
            raise KeyError(f"Node '{node.name}' already exists in graph.")
        self.nodes[node.name] = node

    def add_entry(self, e: GraphEntry):
        """Add an entry to the graph."""
        if e.name in self.entries:
            raise KeyError(f"Entry '{e.name}' already exists in graph.")
        self.entries[e.name] = e

    def copy_attrs(self, src: "Graph") -> "Graph":
        """Copy attributes from another graph."""
        self.name = src.name
        self.performance = src.performance
        self.dynamic_args_max = src.dynamic_args_max
        self.dynamic_args_opt = src.dynamic_args_opt
        self.dynamic_shape = src.dynamic_shape
        self.platform = src.platform
        return self

    def copy(self, src: "Graph") -> "Graph":
        """Copy another graph into this one."""
        self.copy_attrs(src)
        for entry in src.entries.values():
            self.entries[entry.name] = entry.clone()
        for node in src.nodes.values():
            new_node = node.clone()
            new_node.inputs = [self.entries[x.name] for x in node.inputs]
            new_node.outputs = [self.entries[x.name] for x in node.outputs]
            self.nodes[new_node.name] = new_node
        self.inputs = [self.entries[input.name] for input in src.inputs]
        self.outputs = [self.entries[output.name] for output in src.outputs]
        return self

    def clone(self, is_update=False) -> "Graph":
        """Clone the graph with optional update."""
        graph = Graph().copy(self)
        if is_update:
            graph.update()
        return graph

    def _get_symbol_name(self, symbol):
        """Get the name of a sympy symbol."""
        return symbol.name if is_sympy(symbol) and hasattr(symbol, 'name') else None

    def _set_inputs_by_limit(self, condition):
        """Set input dimensions based on dynamic conditions."""
        dict_symbol = {}
        for input in self.inputs:
            for s in input.tensor.shape:
                name = self._get_symbol_name(s)
                if name:
                    dict_symbol[name] = s
        for key in condition:
            if key['type'] == 'equal':
                dst = key['dst']
                src = key['src']
                for input in self.inputs:
                    shape = list(input.tensor.shape)
                    for i, s in enumerate(shape):
                        name = self._get_symbol_name(s)
                        if name == dst and src in dict_symbol:
                            shape[i] = dict_symbol[src]
                    input.tensor.shape = tuple(shape)

    def _apply_dynamic_axes(self, dynamic_args):
        """Apply dynamic axes to the graph."""
        if dynamic_args is None:
            return
        dynamic_dict = {}
        for x, ctx in dynamic_args.items():
            xmin, xmax, xfactor = ctx
            assert xmax % xfactor == 0 and xmin % xfactor == 0
            sx = sympy.Symbol(x, integer=True)
            if xmin != xmax:
                self.dynamic_args_max[x] = xmax
                self.dynamic_args_opt[x] = xmin
                sx = (sx // xfactor) * xfactor
                dynamic_dict[x] = sx
            else:
                dynamic_dict[x] = xmin
        for input in self.inputs:
            shape = list(input.tensor.shape)
            for i, s in enumerate(shape):
                name = self._get_symbol_name(s)
                if name and name in dynamic_dict:
                    shape[i] = dynamic_dict[name]
            input.tensor.shape = tuple(shape)

    def _dynamic_data_tensor(self):
        """Handle dynamic data tensors in the graph."""
        if self.dynamic_args_max is None:
            return
        self._update_entries()
        def _recurse_dynamic_entry(entry):
            if entry.tensor.has_data():
                for data in entry.tensor.data.reshape(-1):
                    if is_sympy(data):
                        node = entry.src_node
                        if node:
                            for input in node.inputs:
                                _recurse_dynamic_entry(input)
        for entry in self.entries.values():
            if entry.tensor.is_dynamic_data:
                _recurse_dynamic_entry(entry)

    def init_tensor(self):
        """Initialize tensors for all nodes in the graph."""
        for node in self.nodes.values():
            inputs = [self.entries[x].tensor for x in node.inputs]
            if node.op is None:
                node.create_op()
            node.op.set_input(inputs)
            node.op.infer_tensor(self.dynamic_args_max)
            outputs = node.op.get_output()
            for i, x in enumerate(node.outputs):
                output_tensor = outputs[i]
                self.entries[x].tensor.shape = output_tensor.shape
                self.entries[x].tensor.data = output_tensor.data
                self.entries[x].tensor.scale = output_tensor.scale
                self.entries[x].tensor.dtype = output_tensor.dtype
                self.entries[x].tensor.bits = output_tensor.bits
                self.entries[x].tensor.layout = output_tensor.layout
        self._dynamic_data_tensor()
        self.update()

    def acquire_dynamic_shape(self):
        """Acquire dynamic shapes for the graph."""
        self.dynamic_shape = {}
        if not self.dynamic_args_max:
            return
        dynamic_args_max = {sympy.Symbol(k, integer=True): v for k, v in self.dynamic_args_max.items()}
        for e in self.entries.values():
            tensor = e.tensor
            if not tensor.has_data():
                shape = list(tensor.shape)
                for i, x in enumerate(shape):
                    if is_sympy(x):
                        if x not in self.dynamic_shape:
                            self.dynamic_shape[x] = []
                        self.dynamic_shape[x].append((e.name, i))
                        shape[i] = int(x.subs(dynamic_args_max))
                if tensor.dtype == np.dtype('O'):
                    tensor.dtype = np.dtype('i8')
                tensor.shape = shape

    def _del_useless_entry(self):
        """Remove useless entries from the graph."""
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
        useless_list = [e.name for e in self.entries.values() if e.name not in use_list]
        for x in useless_list:
            del self.entries[x]

    def _update_entries(self):
        """Update entries in the graph."""
        for e in self.entries.values():
            e.src_node = None
            e.dst_nodes = []
        for node in self.nodes.values():
            for entry in node.inputs:
                self.entries[entry.name].add_dst_node(node)
            for entry in node.outputs:
                self.entries[entry.name].src_node = node
        for i, e in enumerate(self.entries.values()):
            e.index = i

    def _update_nodes(self):
        """Update nodes in the graph."""
        self._sort_nodes()
        for i, node in enumerate(self.nodes.values()):
            node.index = i
            node.inputs = [self.entries[x] for x in node.inputs]
            node.outputs = [self.entries[x] for x in node.outputs]

    def _sort_nodes(self):
        """Sort nodes topologically."""
        new_nodes = IndexedDict()
        def recurse_sort_nodes(name_list):
            for x in name_list:
                e = self.entries[x]
                if e.is_graph_input() or e.is_constant():
                    continue
                node = e.src_node
                if node:
                    if node.name not in new_nodes:
                        recurse_sort_nodes(node.inputs)
                        new_nodes[node.name] = node
        recurse_sort_nodes([x for x in self.outputs])
        self.nodes = new_nodes

    def _constant_fold(self):
        """Perform constant folding on the graph."""
        for e in self.entries.values():
            if e.tensor.has_data() and not is_sympy(e):
                e.set_constant()

    def _update_op_tensors(self):
        """Update tensors for all operators in the graph."""
        for node in self.nodes.values():
            if node.op:
                node.op.inputs = [input.tensor for input in node.inputs]
                node.op.outputs = [output.tensor for output in node.outputs]

    def pack_params(self):
        """Pack parameters for the graph."""
        for node in self.nodes.values():
            if node.op:
                node.op.pack_params()

    def update(self):
        """Update the graph by performing constant folding and other optimizations."""
        self._constant_fold()
        self._update_entries()
        self._update_nodes()
        self._del_useless_entry()
        self._update_op_tensors()

__all__ = ["Graph"]