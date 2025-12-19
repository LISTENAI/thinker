import onnx
from onnx import helper, numpy_helper
from onnx import TensorProto, AttributeProto
from typing import Dict, List, Optional, Any, Tuple
import copy
import numpy as np
from .onnx_graph import Graph, Node, Tensor


class ONNXShapeInference:
    """Complete ONNX Shape Inference implementation based on onnx_graph.py"""

    def __init__(self):
        self.custom_op_handlers = {}
        self.value_info_map = {}
        self.dynamic_info = {}
        self.op_before_handlers = []
        self.op_after_handlers = []
        self.graph = None
        self.std_op_handlers = {
            "Conv": self._infer_conv_shape,
            "Relu": self._infer_elementwise_shape,
            "Add": self._infer_elementwise_shape,
            "Mul": self._infer_elementwise_shape,
            "MatMul": self._infer_matmul_shape,
            "Reshape": self._infer_reshape_shape,
            "Shape": self._infer_shape_shape,
            "Gather": self._infer_gather_shape,
            "Unsqueeze": self._infer_unsqueeze_shape,
            "Squeeze": self._infer_squeeze_shape,
            "Concat": self._infer_concat_shape,
            "Constant": self._infer_constant_shape,
            "MaxPool": self._infer_pool_shape,
            "AveragePool": self._infer_pool_shape,
            "GlobalMaxPool": self._infer_globalpool_shape,
            "GlobalAveragePool": self._infer_globalpool_shape,
            "Flatten": self._infer_flatten_shape,
            "Gemm": self._infer_gemm_shape,
            "Linear": self._infer_linear_shape,
            # thinker ops
            "Quant": self._infer_quant_shape,
            "Dequant": self._infer_dequant_shape,
            "Conv2dInt": self._infer_conv_shape,
            "iqAdd": self._infer_elementwise_shape,
            "AvgPool2dInt": self._infer_pool_shape,
            "LinearInt": self._infer_linear_shape,
            "iqCat": self._infer_concat_shape,
        }

    def register_custom_op(self, op_type: str, shape_inference_func):
        """Register custom operator shape inference function"""
        self.custom_op_handlers[op_type] = shape_inference_func

    def resgister_op_before_handler(
        self, op_type: str, handler
    ) -> "ONNXShapeInference":
        """Register operator pre-processing handler"""
        self.op_before_handlers.append(handler)
        return self

    def resgister_op_after_handler(self, op_type: str, handler) -> "ONNXShapeInference":
        """Register operator post-processing handler"""
        self.op_after_handlers.append(handler)
        return self

    def infer_shapes(self, onnx_path_or_graph: Graph = None, dynamic_info={}) -> Graph:
        """Execute complete shape inference, first perform topological sorting on nodes"""
        # Use provided graph or instance graph
        if isinstance(onnx_path_or_graph, str):
            # Load from ONNX file path
            self.graph = Graph(onnx_path_or_graph)
        elif isinstance(onnx_path_or_graph, Graph):
            # Use existing Graph object
            self.graph = onnx_path_or_graph
        elif onnx_path_or_graph is not None:
            raise ValueError(
                "Input must be either a file path string or a Graph object"
            )

        # Save dynamic information, dynamic_info = {'dim_param': value}
        self.dynamic_info = dynamic_info
        # Create graph copy
        inferred_graph = copy.deepcopy(self.graph)
        # Initialize value_info mapping
        self._initialize_value_info(inferred_graph)
        # Topologically sort nodes
        sorted_nodes = self._topological_sort_nodes(inferred_graph.get_nodes())
        # Process nodes in topological order
        processed_nodes = set()
        for id, node in enumerate(sorted_nodes):
            if (id, node.name) in processed_nodes:
                continue
            if self._are_inputs_ready(node):
                self._infer_node_shapes(node, inferred_graph)
                processed_nodes.add((id, node.name))
        # Finally update output node shape information
        inferred_graph = self._update_output_shapes(inferred_graph)
        return inferred_graph

    def _check_onnx_version(self, min_version="1.8.0"):
        """Check if ONNX version meets minimum requirement"""
        current_version = tuple(int(x) for x in onnx.__version__.split("."))
        min_version_tuple = tuple(int(x) for x in min_version.split("."))
        if current_version >= min_version_tuple:
            return True
        else:
            return False

    def _initialize_value_info(self, graph: Graph):
        """Initialize value_info mapping"""
        self.value_info_map = {}

        # Add inputs
        for input_tensor in graph.get_input():
            self.value_info_map[input_tensor.name] = input_tensor
            self.value_info_map[input_tensor.name].set_shape(
                self._get_shape_with_dynamic(input_tensor)
            )

        # Add initializers
        for init_tensor in graph.get_initializer():
            self.value_info_map[init_tensor.name] = init_tensor

        # Add existing value_info
        for vi_tensor in graph.value_info:
            self.value_info_map[vi_tensor.name] = vi_tensor

    def _get_shape_with_dynamic(self, tensor: Tensor) -> List[int]:
        """Get shape from Tensor object"""
        shape = []
        for dim in tensor.shape:
            if isinstance(dim, str):
                # Dynamic dimension parameter
                if dim in self.dynamic_info:
                    shape.append(self.dynamic_info[dim])
                else:
                    shape.append(None)  # Unknown dynamic dimension
            else:
                shape.append(dim)
        return shape

    def _topological_sort_nodes(self, nodes: List[Node]) -> List[Node]:
        """Perform topological sorting on nodes, return sorted node list"""
        from collections import defaultdict, deque

        # Ensure unique node names
        for id, node in enumerate(nodes):
            if not node.name or node.name == "":
                node.name = f"{node.op_type}_{id}"
            else:
                node.name = f"{node.name}_{id}"

        # Build dependency graph
        name_to_node = {node.name: node for node in nodes}
        output_to_node = {}
        for node in nodes:
            for out in node.output:
                output_to_node[out] = node

        in_degree = {node.name: 0 for node in nodes}
        dependents = defaultdict(list)
        for node in nodes:
            for inp in node.input:
                if inp in output_to_node:
                    in_degree[node.name] += 1
                    dependents[output_to_node[inp].name].append(node.name)

        # Kahn's algorithm
        queue = deque(
            [name_to_node[name] for name, deg in in_degree.items() if deg == 0]
        )
        sorted_nodes = []
        while queue:
            node = queue.popleft()
            sorted_nodes.append(node)
            for dep in dependents[node.name]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(name_to_node[dep])
        if len(sorted_nodes) != len(nodes):
            raise RuntimeError("Graph has cycles or disconnected components.")
        return sorted_nodes

    def _find_producer(self, graph: Graph, tensor_name: str) -> Optional[Node]:
        """Find the node that produces the tensor"""
        return graph.get_node_byoutput(tensor_name)

    def _are_inputs_ready(self, node: Node) -> bool:
        """Check if all inputs of the node are ready for inference"""
        # TODO
        for input_name in node.input:
            if input_name not in self.value_info_map:
                return False
            if self.value_info_map[input_name].get_shape() is None:
                return False
        return True

    def _infer_node_shapes(self, node: Node, graph: Graph):
        """Infer output shapes of the node"""
        # Get input information
        input_info = []
        for input_name in node.input:
            if input_name in self.value_info_map:
                input_info.append(self.value_info_map[input_name])
            else:
                raise RuntimeError(
                    f"Input '{input_name}' for node '{node.name}' not found in value_info_map."
                )

        # Get node attributes
        attrs = node.attributes

        for handler in self.op_before_handlers:
            handler(node.op_type, input_info, attrs)

        # Perform shape inference based on operator type
        if node.op_type in self.custom_op_handlers:
            output_info = self.custom_op_handlers[node.op_type](node, input_info, attrs)
        else:
            output_info = self._infer_standard_op_shape(node.op_type, input_info, attrs)
        for handler in self.op_after_handlers:
            handler(node.op_type, input_info, attrs, output_info)
        # print(f"Node '{node.name}' [{node.op_type}]:")
        # print(f"  Inputs:")
        # for inp in input_info:
        #     print(f"    {inp.name}: shape = {inp.shape}, dtype = {inp.dtype}")
        # print(f"  Outputs:")
        # for out in output_info:
        #     print(f"    {out.name}: shape = {out.shape}, dtype = {out.dtype}")

        # Update output information
        for i, output_name in enumerate(node.output):
            if i < len(output_info):
                self.value_info_map[output_name] = output_info[i]
                self._update_graph_value_info(graph, output_name, output_info[i])

    def _infer_standard_op_shape(
        self, op_type: str, input_info: List[Tensor], attrs: Dict[str, Any]
    ) -> List[Tensor]:
        """Standard operator shape inference (automatic registration and dispatch)"""
        if op_type in self.std_op_handlers:
            return self.std_op_handlers[op_type](input_info, attrs)
        else:
            raise NotImplementedError(
                f"Unsupported op_type '{op_type}', skipping shape inference."
            )

    def _update_graph_value_info(self, graph: Graph, tensor_name: str, info: Tensor):
        """Update ValueInfo in the graph"""
        # Find existing value_info
        existing_vi = graph.get_value_info_byname(tensor_name)
        if existing_vi:
            # Update existing tensor
            self._update_tensor_info(existing_vi, info)
        else:
            # Create new value_info tensor
            info.name = tensor_name  # Set tensor name
            graph.value_info.append(info)

    def _update_tensor_info(self, tensor: Tensor, info: Dict):
        """Update Tensor object with new info"""
        tensor.set_dtype(info.get_dtype())
        tensor.set_shape(info.get_shape())
        tensor.set_value(info.get_value())

    def _update_output_shapes(self, graph: Graph) -> Graph:
        """Update shape information in graph.output"""
        # Update each output tensor with inferred shape information
        for output_tensor in graph.get_output():
            if output_tensor.name in self.value_info_map:
                # Update shape information from value_info_map
                info = self.value_info_map[output_tensor.name]
                self._update_tensor_info(output_tensor, info)

        return graph

    def _infer_conv_shape(
        self, input_info: List[Tensor], attrs: Dict[str, Any]
    ) -> List[Tensor]:
        """Conv operator shape inference"""
        assert (
            len(input_info) >= 2
        ), "Conv operator requires at least 2 inputs (input and weights)."

        input_shape = input_info[0].get_shape()
        weight_shape = input_info[1].get_shape()

        assert (
            input_shape and weight_shape
        ), "Input shape and weight shape are required for conv shape inference."

        # Get attributes
        strides = attrs.get("strides", [1, 1])
        pads = attrs.get("pads", [0, 0, 0, 0])
        dilations = attrs.get("dilations", [1, 1])

        batch_size = input_shape[0]
        out_channels = weight_shape[0]

        def _calc_conv_output_size(
            input_size: int, kernel_size: int, pad: int, stride: int, dilation: int
        ) -> int:
            """Calculate convolution output size"""
            effective_kernel_size = (kernel_size - 1) * dilation + 1
            return (input_size + pad - effective_kernel_size) // stride + 1

        # Calculate output spatial dimensions
        h_out = _calc_conv_output_size(
            input_shape[2], weight_shape[2], pads[0] + pads[2], strides[0], dilations[0]
        )
        w_out = _calc_conv_output_size(
            input_shape[3], weight_shape[3], pads[1] + pads[3], strides[1], dilations[1]
        )

        output_shape = [batch_size, out_channels, h_out, w_out]
        return [Tensor("", dtype=input_info[0].dtype, shape=tuple(output_shape))]

    def _infer_elementwise_shape(
        self, input_info: List[Tensor], attrs: Dict[str, Any]
    ) -> List[Tensor]:
        """Element-wise operator shape inference"""
        assert input_info, "Input info is required for elementwise shape inference."
        # Quant bits.
        data_bits = attrs.get("data_bits", 8)
        assert data_bits in (8, 16, 32), "Only support 8/16/32 bits data"
        output_dtype = (
            np.dtype("int8")
            if data_bits == 8
            else (np.dtype("int16") if data_bits == 16 else np.dtype("int32"))
        )
        # Broadcast shape inference
        output_shape = self._broadcast_shapes(
            [list(info.get_shape()) for info in input_info]
        )
        return [Tensor("", dtype=input_info[0].dtype, shape=tuple(output_shape))]

    def _infer_quant_shape(
        self, input_info: List[Tensor], attrs: Dict[str, Any]
    ) -> List[Tensor]:
        """Element-wise operator shape inference"""
        assert input_info, "Input info is required for elementwise shape inference."
        # Quant bits.
        data_bits = attrs.get("data_bits", 8)
        assert data_bits in (8, 16, 32), "Only support 8/16/32 bits data"
        output_dtype = (
            np.dtype("int8")
            if data_bits == 8
            else (np.dtype("int16") if data_bits == 16 else np.dtype("int32"))
        )
        # Broadcast shape inference
        output_shape = self._broadcast_shapes(
            [list(info.get_shape()) for info in input_info]
        )
        return [Tensor("", dtype=output_dtype, shape=tuple(output_shape))]

    def _infer_dequant_shape(
        self, input_info: List[Tensor], attrs: Dict[str, Any]
    ) -> List[Tensor]:
        """Element-wise operator shape inference"""
        assert input_info, "Input info is required for elementwise shape inference."
        # Quant bits.
        data_bits = attrs.get("data_bits", 8)
        output_dtype = np.dtype("float32")
        # Broadcast shape inference
        output_shape = self._broadcast_shapes(
            [list(info.get_shape()) for info in input_info]
        )
        return [Tensor("", dtype=output_dtype, shape=tuple(output_shape))]

    def _broadcast_shapes(self, shapes: List[List[int]]) -> List[int]:
        """Broadcast shape calculation"""
        if not shapes:
            return None

        # Find maximum rank
        max_rank = max(len(shape) for shape in shapes if shape)

        # Pad all shapes to same rank
        padded_shapes = []
        for shape in shapes:
            if shape is None:
                return None
            padded_shape = [1] * (max_rank - len(shape)) + shape
            padded_shapes.append(padded_shape)

        # Calculate broadcasted shape
        output_shape = []
        for i in range(max_rank):
            sizes = [shape[i] for shape in padded_shapes]
            if all(s == 1 for s in sizes):
                output_shape.append(1)
            else:
                # Find non-1 sizes
                non_one_sizes = [s for s in sizes if s != 1]
                if not non_one_sizes:
                    output_shape.append(1)
                else:
                    # Check if all non-1 sizes are the same
                    if len(set(non_one_sizes)) > 1:
                        raise ValueError(
                            f"Incompatible shapes for broadcasting: {shapes}"
                        )
                    output_shape.append(non_one_sizes[0])

        return output_shape

    def _infer_matmul_shape(
        self, input_info: List[Tensor], attrs: Dict[str, Any]
    ) -> List[Tensor]:
        """MatMul operator shape inference"""
        assert len(input_info) >= 2, "MatMul operator requires 2 inputs."

        A_shape = input_info[0].get_shape()
        B_shape = input_info[1].get_shape()

        if not A_shape or not B_shape:
            raise ValueError("Input shapes are required for MatMul shape inference.")

        # Matrix multiplication shape inference
        if len(A_shape) == 1:
            A_shape = [1] + A_shape  # Promote to 2D
        if len(B_shape) == 1:
            B_shape = B_shape + [1]  # Promote to 2D

        # Check if inner dimensions match
        if A_shape[-1] != B_shape[-2]:
            raise ValueError(f"MatMul shape mismatch: {A_shape} and {B_shape}")

        # Calculate output shape
        output_shape = A_shape[:-1] + [B_shape[-1]]
        if len(output_shape) == 1:
            output_shape = output_shape[0]  # Demote to 1D

        return [Tensor("", input_info[0].get_dtype(), output_shape, None)]

    def _infer_reshape_shape(
        self, input_info: List[Tensor], attrs: Dict[str, Any]
    ) -> List[Tensor]:
        """Reshape operator shape inference"""
        assert (
            len(input_info) >= 2
        ), "Reshape operator requires 2 inputs (data and shape)."

        input_shape = input_info[0].get_shape()
        shape_value = input_info[1].get_value()

        if shape_value is None:
            # Try to get shape from attributes
            if "shape" in attrs:
                new_shape = attrs.get_shape()
            else:
                raise ValueError("Shape value is required for Reshape shape inference.")
        else:
            new_shape = list(shape_value)

        # Handle -1 dimension (auto-inference)
        if -1 in new_shape:
            total_elements = np.prod(input_shape) if input_shape else 1
            specified_elements = np.prod([dim for dim in new_shape if dim != -1])
            inferred_dim = total_elements // specified_elements
            new_shape = [inferred_dim if dim == -1 else dim for dim in new_shape]
        new_shape = [int(x) for x in new_shape]

        return [Tensor("", input_info[0].get_dtype(), new_shape, None)]

    def _infer_shape_shape(
        self, input_info: List[Tensor], attrs: Dict[str, Any]
    ) -> List[Tensor]:
        """Shape operator shape inference"""
        assert len(input_info) >= 1, "Shape operator requires at least 1 input."

        input_shape = input_info[0].get_shape()
        if input_shape is None:
            raise ValueError("Input shape is required for Shape operator inference.")

        start = attrs.get("start", 0)
        end = attrs.get("end", len(input_shape))
        output_shape = [end - start]
        value = (
            np.array(input_shape[start:end], dtype=np.int64)
            if input_shape is not None
            else None
        )
        return [Tensor("", np.dtype("int64"), output_shape, value)]

    def _infer_gather_shape(
        self, input_info: List[Tensor], attrs: Dict[str, Any]
    ) -> List[Tensor]:
        """Gather operator shape inference, supports value calculation"""
        assert len(input_info) >= 2, "Gather operator requires at least 2 inputs."

        data_shape = input_info[0].get_shape()
        indices_shape = input_info[1].get_shape()
        data_value = input_info[0].get_value()
        indices_value = input_info[1].get_value()

        if data_shape is None or indices_shape is None:
            raise ValueError("Input shapes are required for Gather shape inference.")

        axis = attrs.get("axis", 0)
        if axis < 0:
            axis += len(data_shape)

        """
        Meaning explanation:
        ```
        This is the core code for ONNX Gather operator shape inference.
        Assume data's shape is data_shape, indices' shape is indices_shape, axis is the axis for gather operation.

        data_shape[:axis]: Take the first axis dimensions of data_shape.
        indices_shape: Insert indices' shape.
        data_shape[axis+1:]: Take all dimensions from axis+1 onwards of data_shape.
        Final effect:
        Output shape = data_shape with axis dimension replaced by indices_shape. For example:

        data_shape = [2, 3, 4]
        indices_shape = [5, 6]
        axis = 1
        then output shape = [2] + [5, 6] + [4] = [2, 5, 6, 4]

        This is exactly the shape rule of ONNX Gather operator.
        ```
        """
        output_shape = data_shape[:axis] + indices_shape + data_shape[axis + 1 :]
        # Value calculation: only computable when both data_value and indices_value are not None
        value = None
        if data_value is not None and indices_value is not None:
            try:
                value = np.take(data_value, indices_value, axis=axis)
            except Exception:
                value = None

        return [Tensor("", input_info[0].get_dtype(), output_shape, value)]

    def _infer_unsqueeze_shape(
        self, input_info: List[Tensor], attrs: Dict[str, Any]
    ) -> List[Tensor]:
        """Unsqueeze operator shape inference, supports value calculation"""
        assert len(input_info) >= 1, "Unsqueeze operator requires at least 1 input."

        input_shape = input_info[0].get_shape()
        input_value = input_info[0].get_value()
        if input_shape is None:
            raise ValueError("Input shape is required for Unsqueeze shape inference.")

        axes = attrs.get("axes", [])
        output_shape = list(input_shape[:])
        # Record insertion points to avoid multiple insertions at the same position causing misalignment
        for axis in sorted(axes):
            if axis < 0:
                axis += len(output_shape) + 1
            output_shape.insert(axis, 1)

        # Value calculation
        value = None
        if input_value is not None and axes:
            value = np.expand_dims(input_value, tuple(axes))

        return [Tensor("", input_info[0].get_dtype(), output_shape, value)]

    def _infer_squeeze_shape(
        self, input_info: List[Tensor], attrs: Dict[str, Any]
    ) -> List[Tensor]:
        """Squeeze operator shape inference, supports value calculation"""
        assert len(input_info) >= 1, "Squeeze operator requires at least 1 input."

        input_shape = input_info[0].get_shape()
        input_value = input_info[0].get_value()
        if input_shape is None:
            raise ValueError("Input shape is required for Squeeze shape inference.")

        axes = attrs.get("axes", [])
        # ONNX Squeeze: if axes not specified, remove all dimensions with size 1
        if not axes:
            output_shape = [dim for dim in input_shape if dim != 1]
            axes_tuple = tuple(i for i, dim in enumerate(input_shape) if dim == 1)
        else:
            # axes may be negative, need to convert to positive
            axes_tuple = tuple(
                axis if axis >= 0 else axis + len(input_shape) for axis in axes
            )
            output_shape = [
                dim for i, dim in enumerate(input_shape) if i not in axes_tuple
            ]

        # Value calculation
        value = None
        if input_value is not None:
            try:
                value = np.squeeze(input_value, axis=axes_tuple if axes else None)
            except Exception:
                value = None

        return [Tensor("", input_info[0].get_dtype(), output_shape, value)]

    def _infer_concat_shape(
        self, input_info: List[Tensor], attrs: Dict[str, Any]
    ) -> List[Tensor]:
        """Concat operator shape inference, supports value calculation"""
        assert len(input_info) >= 1, "Concat operator requires at least 1 input."

        axis = attrs.get("axis", 0)
        # Handle negative axis
        rank = len(input_info[0].get_shape())
        if axis < 0:
            axis += rank
        # Calculate output shape
        output_shape = list(input_info[0].get_shape())
        output_shape[axis] = sum(
            info.get_shape()[axis]
            for info in input_info
            if info.get_shape() is not None
        )

        # Value calculation: only computable when all input values are not None
        if all(info.get_value() is not None for info in input_info):
            try:
                value = np.concatenate(
                    [info.get_value() for info in input_info], axis=axis
                )
            except Exception:
                value = None
        else:
            value = None

        return [Tensor("", input_info[0].get_dtype(), output_shape, value)]

    def _infer_constant_shape(
        self, input_info: List[Tensor], attrs: Dict[str, Any]
    ) -> List[Tensor]:
        """Constant operator shape inference, supports value calculation"""
        # Constant node has no inputs, value is in attrs['value']
        value = attrs.get("value", None)
        if value is None:
            raise ValueError("Constant operator requires 'value' attribute.")
        # value can be numpy.ndarray or scalar
        if isinstance(value, np.ndarray):
            shape = list(value.shape)
            dtype = value.dtype
        else:
            # Scalar
            shape = []
            dtype = np.array(value).dtype
            value = np.array(value)
        return [Tensor.from_array(value)]

    def _infer_pool_shape(
        self, input_info: List[Tensor], attrs: Dict[str, Any]
    ) -> List[Tensor]:
        """
        Pooling operator shape inference, supports MaxPool/AveragePool, supports value calculation
        """
        assert (
            len(input_info) >= 1 and input_info[0].get_shape()
        ), "Input shape is required for pool shape inference."

        input_shape = input_info[0].get_shape()
        input_dtype = input_info[0].get_dtype()
        input_value = input_info[0].get_value()
        rank = len(input_shape)
        # Only supports 2D pooling
        if rank < 4:
            raise ValueError("Pool operator only supports input with rank >= 4.")

        # Get attributes
        kernel_shape = attrs.get("kernel_shape", [1, 1])
        strides = attrs.get("strides", [1, 1])
        pads = attrs.get(
            "pads", [0, 0, 0, 0]
        )  # [pad_top, pad_left, pad_bottom, pad_right]

        N, C, H, W = input_shape
        kH, kW = kernel_shape
        sH, sW = strides
        pad_top, pad_left, pad_bottom, pad_right = (
            pads if len(pads) == 4 else (0, 0, 0, 0)
        )

        # Calculate output spatial dimensions
        out_H = int((H + pad_top + pad_bottom - kH) // sH + 1)
        out_W = int((W + pad_left + pad_right - kW) // sW + 1)
        output_shape = [N, C, out_H, out_W]

        # Value calculation (only when input value exists)
        value = None
        if input_value is not None:
            # First pad
            padded = np.pad(
                input_value,
                ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )
            out = np.zeros((N, C, out_H, out_W), dtype=input_value.dtype)
            for n in range(N):
                for c in range(C):
                    for i in range(out_H):
                        for j in range(out_W):
                            h_start = i * sH
                            h_end = h_start + kH
                            w_start = j * sW
                            w_end = w_start + kW
                            window = padded[n, c, h_start:h_end, w_start:w_end]
                            if attrs.get("op_type", "").lower() == "averagepool":
                                out[n, c, i, j] = np.mean(window)
                            else:  # Default MaxPool
                                out[n, c, i, j] = np.max(window)
            value = out
        return [Tensor("", input_dtype, output_shape, value)]

    def _infer_globalpool_shape(
        self, input_info: List[Tensor], attrs: Dict[str, Any]
    ) -> List[Tensor]:
        """
        Pooling operator shape inference, supports GlobalMaxPool/GlobalAveragePool, supports value calculation
        """
        assert (
            len(input_info) >= 1 and input_info[0].get_shape()
        ), "Input shape is required for pool shape inference."

        input_shape = input_info[0].get_shape()
        input_dtype = input_info[0].get_dtype()
        input_value = input_info[0].get_value()
        rank = len(input_shape)
        # Only supports 2D pooling
        assert rank >= 4, "GlobalPool operator only supports input with rank >= 4."

        # Get attributes
        kernel_shape = attrs.get("kernel_shape", [input_shape[2], input_shape[3]])
        strides = attrs.get("strides", [1, 1])
        pads = attrs.get(
            "pads", [0, 0, 0, 0]
        )  # [pad_top, pad_left, pad_bottom, pad_right]

        N, C, H, W = input_shape
        kH, kW = kernel_shape
        sH, sW = strides
        pad_top, pad_left, pad_bottom, pad_right = (
            pads if len(pads) == 4 else (0, 0, 0, 0)
        )

        # Calculate output spatial dimensions
        out_H = int((H + pad_top + pad_bottom - kH) // sH + 1)
        out_W = int((W + pad_left + pad_right - kW) // sW + 1)
        output_shape = [N, C, out_H, out_W]

        # Value calculation (only when input value exists)
        value = None
        if input_value is not None:
            # First pad
            padded = np.pad(
                input_value,
                ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )
            out = np.zeros((N, C, out_H, out_W), dtype=input_value.dtype)
            for n in range(N):
                for c in range(C):
                    for i in range(out_H):
                        for j in range(out_W):
                            h_start = i * sH
                            h_end = h_start + kH
                            w_start = j * sW
                            w_end = w_start + kW
                            window = padded[n, c, h_start:h_end, w_start:w_end]
                            if attrs.get("op_type", "").lower() == "globalaveragepool":
                                out[n, c, i, j] = np.mean(window)
                            else:  # Default MaxPool
                                out[n, c, i, j] = np.max(window)
            value = out

        return [Tensor("", input_dtype, output_shape, value)]

    def _infer_flatten_shape(
        self, input_info: List[Tensor], attrs: Dict[str, Any]
    ) -> List[Tensor]:
        """
        Flatten operator shape inference, supports value calculation
        """
        assert (
            len(input_info) >= 1 and input_info[0].get_shape()
        ), "Input shape is required for Flatten shape inference."

        input_shape = input_info[0].get_shape()
        input_dtype = input_info[0].get_dtype()
        input_value = input_info[0].get_value()
        axis = attrs.get("axis", 1)
        rank = len(input_shape)
        if axis < 0:
            axis += rank

        dim0 = int(np.prod(input_shape[:axis])) if axis > 0 else 1
        dim1 = int(np.prod(input_shape[axis:])) if axis < rank else 1
        output_shape = [dim0, dim1]

        value = None
        if input_value is not None:
            value = np.reshape(input_value, output_shape)

        return [Tensor("", input_dtype, output_shape, value)]

    def _infer_gemm_shape(
        self, input_info: List[Tensor], attrs: Dict[str, Any]
    ) -> List[Tensor]:
        """
        Gemm operator shape inference, supports value calculation
        """
        # Gemm: Y = alpha * A * B + beta * C
        # Supports A: (M, K), B: (K, N), C: (M, N) or broadcast
        assert (
            len(input_info) >= 2
            and input_info[0].get_shape()
            and input_info[1].get_shape()
        ), "Input shape is required for Gemm shape inference."

        A_shape = input_info[0].get_shape()
        B_shape = input_info[1].get_shape()
        C_shape = input_info[2].get_shape() if len(input_info) > 2 else None
        A_value = input_info[0].get_value()
        B_value = input_info[1].get_value()
        C_value = input_info[2].get_value() if len(input_info) > 2 else None
        dtype = input_info[0].get_dtype()

        # Get attributes
        alpha = attrs.get("alpha", 1.0)
        beta = attrs.get("beta", 1.0)
        transA = attrs.get("transA", 0)
        transB = attrs.get("transB", 0)

        # Handle transpose
        if transA:
            A_shape = [A_shape[1], A_shape[0]]
            if A_value is not None:
                A_value = np.transpose(A_value)
        if transB:
            B_shape = [B_shape[1], B_shape[0]]
            if B_value is not None:
                B_value = np.transpose(B_value)

        # Shape inference
        M = A_shape[0]
        N = B_shape[1]
        output_shape = [M, N]

        # Value calculation
        value = None
        if A_value is not None and B_value is not None:
            out = alpha * np.matmul(A_value, B_value)
            if C_value is not None:
                # Broadcast C to output shape
                C_broadcast = np.broadcast_to(C_value, out.shape)
                out = out + beta * C_broadcast
            value = out

        return [Tensor("", dtype, output_shape, value)]

    def _infer_linear_shape(
        self, input_info: List[Dict], attrs: Dict[str, Any]
    ) -> List[Dict]:
        """
        Linear operator shape inference, supports value calculation
        Linear: Y = X * W^T + b
        """
        # Linear requires at least 2 inputs (input and weight), bias is optional
        assert (
            len(input_info) >= 2
        ), "Linear operator requires at least 2 inputs (input and weight)."

        input_shape = input_info[0].get_shape()
        weight_shape = input_info[1].get_shape()
        bias_shape = input_info[2].get_shape() if len(input_info) > 2 else None

        input_value = input_info[0].get_value()
        weight_value = input_info[1].get_value()
        bias_value = input_info[2].get_value() if len(input_info) > 2 else None

        dtype = input_info[0].get_dtype()

        if not input_shape or not weight_shape:
            raise ValueError(
                "Input shape and weight shape are required for Linear shape inference."
            )

        # Linear layer shape inference
        # Input: (..., in_features)
        # Weight: (out_features, in_features)
        # Output: (..., out_features)

        in_features = input_shape[-1]
        out_features = weight_shape[0]

        # Check dimension compatibility
        if weight_shape[1] != in_features:
            raise ValueError(
                f"Linear shape mismatch: input last dim {in_features} != weight second dim {weight_shape[1]}"
            )

        # Calculate output shape
        output_shape = input_shape[:-1] + [out_features]

        # Value calculation
        value = None
        if input_value is not None and weight_value is not None:
            try:
                # Reshape input to 2D for matrix multiplication
                batch_size = (
                    int(np.prod(input_shape[:-1])) if len(input_shape) > 1 else 1
                )
                input_2d = input_value.reshape(batch_size, in_features)

                # Linear transformation: Y = X * W^T + b
                output_2d = np.matmul(input_2d, weight_value.T)

                # Add bias if present
                if bias_value is not None:
                    output_2d = output_2d + bias_value

                # Reshape back to output shape
                value = output_2d.reshape(output_shape)
            except Exception:
                value = None

        return [Tensor("", dtype, output_shape, value)]

    ####################################################################################

    def get_value_info(self) -> Dict[str, Dict]:
        """Get all value info for debugging purposes"""
        return self.value_info_map.copy()

    def save(self, output_path: str):
        """Save the inferred graph to ONNX file"""
        if self.graph:
            self.graph.save_onnx(output_path)
        else:
            raise ValueError("No graph available to save")

    def load_from_path(self, onnx_path: str):
        """Load graph from ONNX file path"""
        self.graph = Graph(onnx_path)
        return self


####################################################################################

# Custom operator shape inference example
def custom_op_shape_inference(node, input_info, attrs):
    """Custom operator shape inference example"""
    assert (
        input_info and input_info[0].get_shape()
    ), "Input shape is required for custom_op_shape_inference."
    input_shape = input_info[0].get_shape()
    input_dtype = input_info[0].get_dtype()
    # Calculate output shape based on attributes
    if "output_size" in attrs:
        output_shape = attrs["output_size"]
    else:
        output_shape = input_shape
    return [{"shape": output_shape, "dtype": input_dtype, "value": None}]


def create_test_model():
    """Create test model using onnx_graph"""
    # Create a simple test model and save it first
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 224, 224]
    )
    weight_value = np.random.randn(64, 3, 3, 3).astype(np.float32)
    weight_tensor = onnx.numpy_helper.from_array(weight_value, "weight")

    conv_node = helper.make_node(
        "Conv",
        name="conv_node",
        inputs=["input", "weight"],
        outputs=["conv_output"],
        kernel_shape=[3, 3],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
    )

    relu_node = helper.make_node(
        "Relu", name="relu_node", inputs=["conv_output"], outputs=["relu_output"]
    )

    custom_node = helper.make_node(
        "MyCustomOp",
        name="custom_node",
        inputs=["relu_output"],
        outputs=["custom_output"],
        output_size=[1, 64, 112 * 112],
    )

    output_tensor = helper.make_tensor_value_info(
        "custom_output", TensorProto.FLOAT, None
    )

    graph_def = helper.make_graph(
        nodes=[conv_node, relu_node, custom_node],
        name="test_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_tensor],
    )

    model = helper.make_model(graph_def, producer_name="shape-inference-test")
    model.ir_version = 6
    model.opset_import[0].version = 11

    # Save temporarily and load with Graph
    temp_path = "temp_model.onnx"
    onnx.save(model, temp_path)
    graph = Graph(temp_path)

    import os

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return graph


def main_debug():
    """Main function, execute tests"""
    test_graph = create_test_model()

    shape_inferencer = ONNXShapeInference()
    shape_inferencer.register_custom_op("MyCustomOp", custom_op_shape_inference)
    inferred_graph = shape_inferencer.infer_shapes(test_graph)

    # Get output shapes
    output_shapes = {}
    for output_tensor in inferred_graph.get_output():
        output_shapes[output_tensor.name] = (
            list(output_tensor.shape) if output_tensor.shape else None
        )

    print("Inferred output shapes:", output_shapes)

    # Save results
    inferred_graph.save_onnx("model_with_shapes.onnx")
    print("Model saved to model_with_shapes.onnx")


##################################################################################################
# Command line interface
##################################################################################################

import argparse
import json
import os
import sys


def parse_json_parameter(json_str_or_path: str):
    """Parse JSON parameter from string or file path"""
    if os.path.isfile(json_str_or_path):
        with open(json_str_or_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return json.loads(json_str_or_path)


def main(args: argparse.Namespace):
    """Main function, execute tests"""
    shape_inferencer = ONNXShapeInference()
    shape_inferencer.register_custom_op("MyCustomOp", custom_op_shape_inference)

    # Load config if provided
    config = parse_json_parameter(args.config) if args.config else {}
    dynamic_shape = config.get("dynamic_shape", {})
    # Infer shapes
    inferred_graph = shape_inferencer.infer_shapes(
        args.input, dynamic_info=dynamic_shape
    )
    # Get output shapes
    output_shapes = {}
    for output_tensor in inferred_graph.get_output():
        output_shapes[output_tensor.name] = (
            list(output_tensor.shape) if output_tensor.shape else None
        )
    print("Inferred output shapes:", output_shapes)
    # Save results
    inferred_graph.save_onnx(args.output)


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ONNX Shape Inference Tool")
    parser.add_argument(
        "--input", type=str, required=False, help="Path to input ONNX model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_with_shapes.onnx",
        help="Path to save the output ONNX model with inferred shapes",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run test with a sample model"
    )
    parser.add_argument(
        "--config", type=str, required=False, help="Path to config json str or file"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(parse_arguments()))

__all__ = ['']