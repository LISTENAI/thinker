# Copyright (C) 2025 listenai Co.Ltd
# All rights reserved.
# Created by leifang on 2022.09.31

import sympy
import numpy as np
from enum import Enum

class ScalarOpType(Enum):
    """Enumeration of scalar operation types."""
    ADD = 1
    MUL = 2
    DIV = 3
    POW = 4
    FLOOR = 5
    CEILING = 6
    SQRT = 7
    MIN = 8
    MAX = 9

    @classmethod
    def from_str(cls, name):
        """Convert a string to a ScalarOpType."""
        name_ = str(name).upper()
        if name_ not in cls.__dict__:
            raise LookupError(f"Unsupported scalar op: {name}")
        return cls.__dict__[name_]

class ScalarOpNode:
    """Node class for scalar operations."""
    def __init__(self):
        self.inputs = []
        self.output = None
        self.op = None
        self.expr = ""

    def __repr__(self):
        """Representation of the node."""
        return self.expr

    def serialize(self):
        """Serialize the node into bytes."""
        data = [self.op.value] + self.inputs + [self.output]
        return np.array(data, dtype=np.int32).tobytes()

class ScalarGraph:
    """Graph class for scalar operations."""
    
    def __init__(self):
        """Initialize an empty scalar graph."""
        self.inputs = []
        self.outputs = []
        self.scalars = []
        self.nodes = []
        self.input_names = []

    def serialize(self):
        """Serialize the graph into bytes."""
        num_input = len(self.inputs)
        num_output = len(self.outputs)
        num_scalars = len(self.scalars)
        num_node = len(self.nodes)
        
        # Serialize header
        header = np.array([num_input, num_output, num_node, num_scalars], dtype=np.int32)
        buffer = header.tobytes()
        
        # Serialize input and output IDs
        buffer += np.array(self.inputs, dtype=np.int32).tobytes()
        buffer += np.array(self.outputs, dtype=np.int32).tobytes()
        
        # Serialize input names
        for name in self.input_names:
            name_bytes = bytes(str(name), encoding='utf-8')
            assert len(name_bytes) <= 32, f"Name: {name} exceeds size limit."
            buffer += name_bytes + b'\x00' * (32 - len(name_bytes))
        
        # Serialize scalars
        scalars_np = np.zeros(num_scalars, dtype=np.float64)
        for index, scalar in enumerate(self.scalars):
            if scalar.is_Number:
                scalars_np[index] = float(scalar)
        buffer += scalars_np.tobytes()
        
        # Serialize nodes
        node_metas = []
        node_buffers = b''
        for node in self.nodes:
            node_buffers += node.serialize()
            node_metas.append(len(node.inputs) + 2)
        buffer += np.array(node_metas, dtype=np.int32).tobytes() + node_buffers
        
        return buffer

    @staticmethod
    def from_exprs(sym_exprs) -> "ScalarGraph":
        """Create a ScalarGraph from sympy expressions."""
        if not isinstance(sym_exprs, (list, tuple)):
            sym_exprs = [sym_exprs]
        
        g = ScalarGraph()
        scalar_id_map = {}

        def make_nodes(expr):
            if expr in scalar_id_map:
                return scalar_id_map[expr]
            input_ids = []
            for arg in expr.args:
                input_ids.append(make_nodes(arg))
            output_id = len(g.scalars)
            scalar_id_map[expr] = output_id
            if expr.is_Number:
                g.scalars.append(expr)
            elif expr.is_Symbol:
                g.scalars.append(expr)
                g.input_names.append(expr)
                g.inputs.append(output_id)
            else:
                op_name = type(expr).__name__
                op_type = ScalarOpType.from_str(op_name)
                node = ScalarOpNode()
                node.inputs = input_ids
                node.output = output_id
                node.op = op_type
                node.expr = str(expr)
                g.nodes.append(node)
                g.scalars.append(expr)
            return output_id

        for expr in sym_exprs:
            sympify_expr = sympy.sympify(expr)
            output_index = make_nodes(sympify_expr)
            g.outputs.append(output_index)
        return g

__all__ = ["ScalarGraph"]