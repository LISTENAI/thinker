import os
import onnx
from onnx import numpy_helper
import logging
import numpy as np
import argparse
from .onnx_infer import ONNXShapeInference
from .onnx_graph import Graph, Tensor, Node
from typing import Dict, List, Optional, Any, Tuple

from . import luna_profile_arcs as luna_profile

# ============================================================================================


class ONNXProfile:
    """Profile ONNX model to estimate the number of cycles for each operator"""

    def __init__(self):
        self.graph = None
        self.estimate_result = []
        self.op_cycle_funcs = {
            "ArgMax": self._estimate_argmax_cycles,
            "AvgPool2dInt": self._estimate_avgpool2dint_cycles,
            "BMMInt": self._estimate_bmmint_cycles,
            "BatchNormInt": self._estimate_batchnormint_cycles,
            "Cast": self._estimate_cast_cycles,
            "Concat": self._estimate_concat_cycles,
            "Constant": self._estimate_zero_cycles,
            "Conv1dInt": self._estimate_conv2d_cycles,
            "Conv2dInt": self._estimate_conv2d_cycles,
            "ConvTranspose2dInt": self._estimate_convtranspose2dint_cycles,
            "Deconv2dInt": self._estimate_conv2d_cycles,
            "Dequant": self._estimate_zero_cycles,
            "FFNInt": self._estimate_unknow_cycles,  # nlang
            "Flatten": self._estimate_flatten_cycles,
            "GLUInt": self._estimate_gluint_cycles,
            "GRUInt": self._estimate_gruint_cycles,
            "Gather": self._estimate_gather_cycles,
            "iqAdd": self._estimate_iqadd_cycles,
            "iqCat": self._estimate_iqcat_cycles,
            "iqDiv": self._estimate_iqdiv_cycles,
            "iqMul": self._estimate_iqmul_cycles,
            "iqPad": self._estimate_iqpad_cycles,
            "iqSigmoid": self._estimate_iqsigmoid_cycles,
            "iqSub": self._estimate_iqsub_cycles,
            "iqSum": self._estimate_iqsum_cycles,
            "iqTanh": self._estimate_iqtanh_cycles,
            "iqVar": self._estimate_iqvar_cycles,
            "LSTMInt": self._estimate_lstmint_cycles,  # nlang
            "LayerNormInt": self._estimate_layernormint_cycles,
            "LinearInt": self._estimate_linearint_cycles,
            "LogSoftmaxInt": self._estimate_logsoftmaxint_cycles,
            "MaxPool": self._estimate_maxpool_cycles,
            "multiheadattentionint": self._estimate_unknow_cycles,
            "Prelu": self._estimate_prelu_cycles,
            "Quant": self._estimate_quant_cycles,
            "ReduceMean": self._estimate_reducemean_cycles,
            "Relu": self._estimate_relu_cycles,
            "Requant": self._estimate_requant_cycles,
            "Reshape": self._estimate_reshape_cycles,
            "ShuffleChannel": self._estimate_shufflechannel_cycles,
            "Slice": self._estimate_slice_cycles,
            "SoftmaxInt": self._estimate_softmaxint_cycles,
            "sparifyffnint": self._estimate_unknow_cycles,
            "Split": self._estimate_split_cycles,
            "Tile": self._estimate_tile_cycles,
            "TopK": self._estimate_topn_cycles,
            "topn": self._estimate_topn_cycles,
            "topn2": self._estimate_topn2_cycles,
            "Transpose": self._estimate_transpose_cycles,
            "upsampleint": self._estimate_unknow_cycles,
        }

    def logging_node_info(self, node, shape_dict=None):
        logging.info(f"Name: {node.name}, OpType: {node.op_type}")

        for inp in node.input:
            shape = shape_dict.get(inp) if shape_dict else None
            logging.info(f"{inp}, shape={shape}")

        for out in node.output:
            shape = shape_dict.get(out) if shape_dict else None
            logging.info(f"{out}, shape={shape}")

        logging.info(node.attributes)

    def build_shape_dict(self, input_path):
        model = onnx.load(input_path)
        shape_dict = {}

        def extract(tensor_value_info):
            name = tensor_value_info.name
            dims = []
            for d in tensor_value_info.type.tensor_type.shape.dim:
                if d.dim_value > 0:
                    dims.append(d.dim_value)
                else:
                    dims.append("None")
            shape_dict[name] = dims

        for init in model.graph.initializer:
            arr = numpy_helper.to_array(init)
            shape_dict[init.name] = arr.shape
            
        for info in model.graph.input:            
            extract(info)
        for info in model.graph.output:
            extract(info)
        for info in model.graph.value_info:
            extract(info)

        return shape_dict

    def estimate(self, args: argparse.Namespace) -> int:
        """Estimate total cycles for all operators in the model"""
        asynchro = args.asynchro
        self.graph = Graph(args.input)
        shape_dict  = self.build_shape_dict(args.input)
        total_cycles = 0
        total_cpu_cycles = 0
        total_dma_cycles = 0
        total_luna_cycles = 0
        pre_luna_cycles = 0
        self.estimate_result = []
        index = 0
        for node in self.graph.node:
            logging.debug(f"estimate op, index:{index}, name:{node.name}")
            inputs = [self._get_value_info_byname(self.graph, x) for x in node.input]
            outputs = [self._get_value_info_byname(self.graph, x) for x in node.output]
            cpu_cycles = 0
            self.logging_node_info(node, shape_dict)
            luna_cycles, dma_cycles, *cpu = self._estimate_op_cycles(
                node, inputs, outputs
            )
            luna_cycles = int(luna_cycles)
            if cpu:
                cpu_cycles = cpu[0]

            if asynchro:  # max(luna, dma)
                dma_tail = (
                    dma_cycles - pre_luna_cycles if dma_cycles > pre_luna_cycles else 0
                )
            else:  # luan + dma
                dma_tail = dma_cycles

            pre_luna_cycles = luna_cycles
            op_cycles = cpu_cycles + dma_tail + luna_cycles
            # print(f"Estimated cycles for {node.name}[{node.op_type}]:\t\t\t {op_cycles}")
            total_cycles += op_cycles
            total_cpu_cycles += cpu_cycles
            total_dma_cycles += dma_tail
            total_luna_cycles += luna_cycles
            self.estimate_result.append(
                (node.name, op_cycles, cpu_cycles, dma_tail, luna_cycles)
            )
            index += 1
        # self.estimate_result.append(('', total_cycles, total_cpu_cycles, total_dma_cycles, total_luna_cycles))
        return self.estimate_result

    def get_result(self) -> List[Tuple[str, int, int, int, int]]:
        """Get the raw estimate result"""
        return self.esrtimate_result

    def get_format_result(self, format="speedscope") -> str:
        """Get the formatted estimate result"""
        if format == "speedscope":
            return self._format_speedscope(self.graph.name, self.estimate_result)
        elif format == "csv":
            return self._format_csv(self.graph.name, self.estimate_result)
        elif format == "debug":
            return self._format_debug(self.graph.name, self.estimate_result)
        else:
            raise NotImplementedError(f"Format {format} is not implemented.")

    def _get_value_info_byname(self, graph: Graph, name: str) -> Optional[Tensor]:
        for value_info in graph.value_info:
            if value_info.name == name:
                return value_info
        for input in graph.input:
            if input.name == name:
                return input
        for output in graph.output:
            if output.name == name:
                return output
        for initializer in graph.initializer:
            if initializer.name == name:
                return initializer
        raise ValueError(f"Value info {name} not found in graph.\nPlease use thinker tpacker onnx file firstly")
        return None

    def _estimate_op_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ) -> Tuple[int, int]:
        """Estimate the number of cycles for a given operator"""
        if node.op_type in self.op_cycle_funcs:
            return self.op_cycle_funcs[node.op_type](node, inputs, outputs)
        else:
            return self._estimate_unknow_cycles(node, inputs, outputs)

    def _estimate_unknow_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ) -> int:
        logging.warning(
            f"Unknown operator type: {node.op_type}, name: {node.name}. Assuming 0 cycles."
        )
        # raise NotImplementedError(f"Cycle estimation for operator {node.op_type} is not implemented.")
        return 0, 0

    def _estimate_conv2d_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ) -> int:
        """Estimate cycles for Conv2d operator"""
        assert len(inputs) >= 2, "Conv2d must have at least 2 inputs (input, weight)"
        assert len(outputs) >= 1, "Conv2d must have at least 1 output"
        input = inputs[0]
        weight = inputs[1]
        bias = inputs[2] if len(inputs) > 2 else None
        output = outputs[0]
        input_shape = input.get_shape()
        weight_shape = weight.get_shape()
        bias_shape = bias.get_shape() if bias else ()
        output_shape = output.get_shape()
        attrs = node.attributes
        input_n, input_c, input_h, input_w = input_shape
        output_n, output_c, output_h, output_w = output_shape
        weight_out_c, weight_in_c, weight_h, weight_w = weight_shape
        stride_h, stride_w = attrs["strides"]
        dilation_h, dilation_w = attrs["dilations"]
        padding_h_0, padding_w_0, padding_h_1, padding_w_1 = attrs["pads"]
        groups = attrs["group"]
        data_bits = attrs.get("data_bits", 8)
        parameter_bits = attrs.get("parameter_bits", 8)
        o_bits = attrs.get("o_bits", 8)
        scale_x = attrs.get("scale_x", 1.0)
        scale_w = attrs.get("scale_w", 1.0)
        scale_o = attrs.get("scale_o", 1.0)
        castor_mode = attrs.get("castor_mode", 0)
        platform_quant = attrs.get("platform_quant", 0)

        # print(f"Conv2d: {node.name}, input shape: {input_shape}, weight shape: {weight_shape}, output shape: {output_shape}, attrs: {attrs}")
        assert padding_h_0 == padding_h_1, "Only support symmetric padding"
        assert padding_w_0 == padding_w_1, "Only support symmetric padding"
        assert (
            input_c == weight_in_c * groups
        ), f"Input channels {input_c} must equal to weight in channels {weight_in_c} * groups {groups}"
        assert (
            output_c == weight_out_c
        ), f"Output channels {output_c} must equal to weight out channels {weight_out_c}"
        assert (
            input_n == output_n
        ), f"Input batch size {input_n} must equal to output batch size {output_n}"
        assert (
            input_c % groups == 0
        ), f"Input channels {input_c} must be divisible by groups {groups}"
        assert (
            weight_out_c % groups == 0
        ), f"Weight out channels {weight_out_c} must be divisible by groups {groups}"

        assert data_bits in (8,), "Only support 8 bits data"
        assert parameter_bits in (8,), "Only support 8 bits parameter"
        assert o_bits in (8, 32), "Only support 8 or 32 bits output"
        assert (
            scale_x * scale_w > scale_o
        ), "Scale of input * weight must be greater than scale of output"

        if groups > 1:
            if groups == input_c and groups == weight_out_c:
                # Depthwise Conv2d
                luna_cycles = luna_profile.est_nn_depthwise2d_cycles(
                    input_c,
                    input_h,
                    input_w,
                    output_c,
                    weight_h,
                    weight_w,
                    stride_h,
                    stride_w,
                    dilation_h,
                    dilation_w,
                    padding_h_0,
                    padding_h_1,
                    padding_w_0,
                    padding_w_1,
                    groups,
                    i_bits_i=data_bits,
                    i_bits_w=parameter_bits,
                    i_bits_b=32,
                    o_bits=o_bits,
                )
                # TODO:
                node.name += "_dw"
            else:
                raise NotImplementedError("Grouped Conv2d shape error.")
        elif dilation_h > 1 or dilation_w > 1:
            luna_cycles = luna_profile.est_nn_deconv2d_cycles(
                input_c,
                input_h,
                input_w,
                output_c,
                weight_h,
                weight_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                padding_h_0,
                padding_h_1,
                padding_w_0,
                padding_w_1,
                groups,
                data_bits,
                parameter_bits,
                32,
                o_bits,
            )
        else:
            luna_cycles = luna_profile.est_nn_conv2d_cycles(
                input_c,
                input_h,
                input_w,
                output_c,
                weight_h,
                weight_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                padding_h_0,
                padding_h_1,
                padding_w_0,
                padding_w_1,
                groups,
                i_bits_i=data_bits,
                i_bits_w=parameter_bits,
                i_bits_b=32,
                o_bits=o_bits,
            )
            # TODO:
            if weight_h == 1 and weight_w == 1:
                node.name += "_pw"
            else:
                node.name += "_com"

        weight_size = weight.get_size()
        dma_cycles = luna_profile.est_dma_cycles(
            size=weight_size,
            i_bits=parameter_bits,
            o_bits=o_bits,
            i_device=1,
            o_device=0,
        )
        cpu_cycles = 12500
        return luna_cycles, dma_cycles, cpu_cycles

    def _estimate_avgpool2dint_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        """Estimate cycles for argmax operator"""
        assert len(inputs) == 1, "argmax must equal 1 inputs"
        assert len(outputs) == 1, "argmax must equal 1 output"

        input1 = inputs[0]
        input1_shape = input1.get_shape()
        in_c = input1_shape[-3]

        output = outputs[0]
        input1_shape = input1.get_shape()
        output_shape = output.get_shape()
        ou_channel_size = output_shape[-2] * output_shape[-1]
        kernel_shape = node.attributes["kernel_shape"]
        one_kernel_size = kernel_shape[0] * kernel_shape[1]
        strides = node.attributes["strides"]
        pads = node.attributes["pads"]
        o_bits = node.attributes.get("o_bits", 8)
        data_bit = node.attributes.get("data_bits", 8)

        total_cycles = 0
        # need more consider thinker input_condition
        total_cycles += luna_profile.est_nn_avgpool2d_cycles(
            input_c=input1_shape[1],
            input_h=input1_shape[2],
            input_w=input1_shape[3],
            output_c=output_shape[1],
            kernel_h=kernel_shape[0],
            kernel_w=kernel_shape[1],
            stride_h=strides[0],
            stride_w=strides[1],
            padding_h_0=pads[0],
            padding_h_1=pads[1],
            padding_w_0=pads[2],
            padding_w_1=pads[3],
            i_bits_i=data_bit,
            o_bits=o_bits,
        )
        total_cycles += luna_profile.est_memset_cycles(
            size=one_kernel_size, i_bits=32, o_bits=32
        )
        total_cycles += luna_profile.est_vector_div_cycles(
            size=in_c * ou_channel_size, i_bits_0=32, i_bits_1=32, o_bits=32
        )
        total_cycles += luna_profile.est_vector_scale_cycles(
            size=output.get_size(), i_bits_0=32, i_bits_1=32, o_bits=8
        )
        cpu_cycles = 20505
        return total_cycles, 0, cpu_cycles

    def _estimate_convtranspose2dint_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        total_cycles = 0
        logging.warning(
            f"Unknown operator type: {node.op_type}, name: {node.name}. Assuming 0 cycles."
        )
        total_cycles += luna_profile.est_convtranspose2dint_cycles()
        return total_cycles, 0

    def _estimate_zero_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        total_cycles = 0
        return total_cycles, 0

    def _estimate_prelu_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        """Estimate cycles for prelu operator"""
        assert len(inputs) == 1, "prelu must equal 1 inputs"
        assert len(outputs) == 1, "prelu must equal 1 output"

        input1 = inputs[0]
        output = outputs[0]

        input_shape1 = input1.get_shape()
        output_shape = output.get_shape()

        assert (
            input_shape1 == output_shape
        ), "Output shape must match input shape for prelu"

        total_cycles = 0

        total_cycles += luna_profile.est_prelu_cycles(
            size=input1.get_size(),
            i_bits=input1.get_bitsize(),
            o_bits=output.get_bitsize(),
            i_device=0,
            o_device=0,
        )

        return total_cycles, 0

    def _estimate_maxpool_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        """Estimate cycles for argmax operator"""
        assert len(inputs) == 1, "argmax must equal 1 inputs"
        assert len(outputs) == 1, "argmax must equal 1 output"

        input1 = inputs[0]
        output = outputs[0]
        input1_shape = input1.get_shape()
        output_shape = output.get_shape()
        kernel_shape = node.attributes["kernel_shape"]
        strides = node.attributes["strides"]
        pads = node.attributes["pads"]
        o_bits = node.attributes.get("o_bits", 8)
        data_bit = node.attributes.get("data_bits", 8)

        total_cycles = 0
        total_cycles += luna_profile.est_nn_maxpool2d_cycles(
            input_c=input1_shape[1],
            input_h=input1_shape[2],
            input_w=input1_shape[3],
            output_c=output_shape[1],
            kernel_h=kernel_shape[0],
            kernel_w=kernel_shape[1],
            stride_h=strides[0],
            stride_w=strides[1],
            padding_h_0=pads[0],
            padding_h_1=pads[1],
            padding_w_0=pads[2],
            padding_w_1=pads[3],
            i_bits_i=data_bit,
            o_bits=o_bits,
        )
        total_cycles += luna_profile.est_vector_scale_cycles(
            size=output.get_size(), i_bits_0=data_bit, i_bits_1=data_bit, o_bits=o_bits
        )
        return total_cycles, 0

    def _estimate_linearint_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        """Estimate cycles for linearint operator"""
        assert len(inputs) == 1, "linearint must equal 3 inputs"
        assert len(outputs) == 1, "linearint must equal 1 output"
        input1 = inputs[0]
        input2 = inputs[1]
        input3 = inputs[2]
        output = outputs[0]

        input1_shape = input1.get_shape()
        output_shape = output.get_shape()

        data_bit = int(node.attributes["data_bits"])
        parameter_bits = int(node.attributes["parameter_bits"])
        o_bits = int(node.attributes["o_bits"])

        total_cycles = 0
        dma_cycles = 0
        total_cycles += luna_profile.est_mat_mul_bias_cycles(
            row=input1_shape[0],
            col=input1_shape[1],
            col2=output_shape[1],
            i_bits_0=data_bit,
            i_bits_1=parameter_bits,
            o_bits=o_bits,
        )
        dma_cycles += luna_profile.est_dma_cycles(
            size=input2.get_size(),
            i_bits=parameter_bits,
            o_bits=o_bits,
            i_device=1,
            o_device=0,
        )
        dma_cycles += luna_profile.est_dma_cycles(
            size=input3.get_size(),
            i_bits=parameter_bits,
            o_bits=o_bits,
            i_device=1,
            o_device=0,
        )
        return total_cycles, dma_cycles

    def _estimate_gluint_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        total_cycles = 0
        input0 = inputs[0]
        input_shape = input0.get_shape()

        axis = node.attributes["axis"]
        o_bits = node.attributes["o_bits"]

        axis = (len(input_shape) + axis) if axis < 0 else axis
        in_dims = len(input_shape)

        M = 1
        for i in range(axis):
            M *= input_shape[i]
        N = input_shape[axis]

        M = (M + 7) & ~7
        N = (N + 7) & ~7
        size = M * N
        half_size = size / 2

        total_cycles += luna_profile.est_mat_trans_cycles(row=M, col=N)
        total_cycles += luna_profile.est_vector_scale_cycles(size, o_bits=32)
        # bit 相同， scaler不同
        total_cycles += luna_profile.est_vector_scale_cycles(
            size=half_size, i_bits_0=32, i_bits_1=32, o_bits=32
        )
        total_cycles += luna_profile.est_sigmoid_cycles(
            size=half_size, i_bits=32, o_bits=32
        )
        total_cycles += luna_profile.est_vector_scale_cycles(
            size=half_size, i_bits_0=32, i_bits_1=32, o_bits=32
        )
        total_cycles += luna_profile.est_vector_mul_cycles(
            size=half_size, i_bits_0=32, i_bits_1=32, o_bits=o_bits
        )
        total_cycles += luna_profile.est_mat_trans_cycles(row=N / 2, col=M)
        return total_cycles, 0

    def _estimate_gruint_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        logging.warning(
            f"Unknown operator type: {node.op_type}, name: {node.name}. Assuming 0 cycles. Thinker not implement."
        )
        total_cycles = 0
        input0 = inputs[0]
        input0_shape = input0.get_shape()
        # direction 不考虑 bidirectional, forward和reverse效果相同
        # 考虑layout
        layout = node.attributes["layout"]
        hidden_size = node.attributes["hidden_size"]
        if layout == 0:
            # T B D
            seq_len = input0_shape[0]
            batch_size = input0_shape[1]
        else:
            # B T D
            seq_len = input0_shape[1]
            batch_size = input0_shape[0]

        gate_size = batch_size * hidden_size
        for i in range(seq_len):
            for j in range(2):  # input and hidden
                luna_profile.est_mat
                total_cycles += luna_profile.est_mat_mul_bias_cycles(
                    row=batch_size, col=input0_shape[2], col2=3 * hidden_size, o_bits=32
                )
                total_cycles += luna_profile.est_vector_add_cycles(
                    size=3 * gate_size, i_bits_0=32, i_bits_1=32, o_bits=32
                )
                total_cycles += luna_profile.est_vector_scale_cycles(
                    size=3 * gate_size, i_bits_0=32, i_bits_1=32, o_bits=32
                )
            total_cycles += luna_profile.est_vector_add_cycles(
                size=2 * gate_size, i_bits_0=32, i_bits_1=32, o_bits=32
            )
            total_cycles += luna_profile.est_sigmoid_cycles(
                size=hidden_size, i_bits=32, o_bits=8
            )
            total_cycles += luna_profile.est_sigmoid_cycles(
                size=hidden_size, i_bits=32, o_bits=8
            )

            # cal G_n
            total_cycles += luna_profile.est_vector_scale_cycles(
                size=hidden_size, o_bits=32
            )
            total_cycles += luna_profile.est_vector_mul_cycles(
                row=hidden_size, i_bits_0=32, i_bits_1=32, o_bits=32
            )
            total_cycles += luna_profile.est_vector_add_cycles(
                size=hidden_size, i_bits_0=32, i_bits_1=32, o_bits=32
            )
            total_cycles += luna_profile.est_tanh_cycles(
                hidden_size, i_bits=32, o_bits=8
            )

            # cal h_y
            total_cycles += luna_profile.est_vector_scale_cycles(size=hidden_size)
            total_cycles += luna_profile.est_vector_mul_cycles(
                row=hidden_size, i_bits_0=8, i_bits_1=8, o_bits=32
            )
            total_cycles += luna_profile.est_vector_scale_cycles(
                size=hidden_size, o_bits=32
            )
            total_cycles += luna_profile.est_vector_offset_cycles(
                size=hidden_size, i_bits_0=32, i_bits_1=32, o_bits=32
            )
            total_cycles += luna_profile.est_vector_scale_cycles(
                size=hidden_size, i_bits_0=8, i_bits_1=8, o_bits=32
            )
            total_cycles += luna_profile.est_vector_mul_cycles(
                row=hidden_size, i_bits_0=32, i_bits_1=32, o_bits=32
            )
            total_cycles += luna_profile.est_vector_add_cycles(
                size=hidden_size, i_bits_0=32, i_bits_1=32, o_bits=32
            )
            total_cycles += luna_profile.est_memcpy_cycles(size=hidden_size)

        return 0, 0

    def _estimate_lstmint_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        """Estimate cycles for lstmint operator"""
        assert len(inputs) >= 2, "bmmint must greater 2 inputs"
        assert len(outputs) == 2, "bmmint must equal 1 output"

        attr = node.attributes
        hidden_size = attr.get("hidden_size")

        input = inputs[0]
        input_shape = input.get_shape()

        total_cycles = 0
        dma_cycles = 0
        # reset history_c history_h
        total_cycles += luna_profile.est_memcpy_cycles(size=hidden_size)
        total_cycles += luna_profile.est_memset_cycles(size=hidden_size)

        # dma
        dma_cycles += luna_profile.est_dma_cycles(
            size=input_shape[-1] * hidden_size * 4, i_device=1, o_device=0
        )
        dma_cycles += luna_profile.est_dma_cycles(
            size=input_shape[-1] * hidden_size * 4, i_device=1, o_device=0
        )
        dma_cycles += luna_profile.est_dma_cycles(
            size=hidden_size * 4, i_device=1, o_device=0
        )
        dma_cycles += luna_profile.est_dma_cycles(
            size=hidden_size * 4, i_device=1, o_device=0
        )
        # cycle: input ht-1
        total_cycles += luna_profile.est_mat_mul_bias_cycles(
            row=input_shape[-2], col=input_shape[-1], col2=hidden_size * 4, o_bits=32
        )
        total_cycles += luna_profile.est_mat_mul_bias_cycles(
            row=hidden_size, col=hidden_size, col2=hidden_size * 4, o_bits=32
        )
        # scale
        total_cycles += 2 * luna_profile.est_vector_scale_cycles(
            size=hidden_size * 4, i_bits_0=32, i_bits_1=32, o_bits=32
        )
        total_cycles += luna_profile.est_vector_add_cycles(
            size=hidden_size * 4, i_bits_0=32, i_bits_1=32, o_bits=32
        )
        for i in range(3):
            total_cycles += luna_profile.est_sigmoid_cycles(
                size=hidden_size, i_bits=32, o_bits=32
            )
        total_cycles += luna_profile.est_tanh_cycles(
            size=hidden_size, i_bits=32, o_bits=32
        )
        for i in range(4):
            total_cycles += luna_profile.est_vector_scale_cycles(
                size=hidden_size, i_bits_0=32, i_bits_1=32, o_bits=32
            )

        # cell
        total_cycles += luna_profile.est_vector_mul_cycles(
            size=hidden_size, i_bits_0=32, i_bits_1=32, o_bits=32
        )
        total_cycles += luna_profile.est_vector_mul_cycles(
            size=hidden_size, i_bits_0=32, i_bits_1=32, o_bits=32
        )
        total_cycles += luna_profile.est_vector_add_cycles(
            size=hidden_size, i_bits_0=32, i_bits_1=32, o_bits=32
        )
        # hidden
        total_cycles += luna_profile.est_tanh_cycles(
            size=hidden_size, i_bits=32, o_bits=32
        )
        total_cycles += luna_profile.est_vector_scale_cycles(
            size=hidden_size, i_bits_0=32, i_bits_1=32, o_bits=32
        )
        total_cycles += luna_profile.est_vector_scale_cycles(
            size=hidden_size, i_bits_0=32, i_bits_1=32, o_bits=32
        )
        total_cycles += luna_profile.est_vector_mul_cycles(
            size=hidden_size, i_bits_0=32, i_bits_1=32, o_bits=8
        )
        total_cycles += luna_profile.est_vector_scale_cycles(size=hidden_size)

        return total_cycles, dma_cycles

    def _estimate_bmmint_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        """Estimate cycles for bmmint operator"""
        assert len(inputs) == 2, "bmmint must equal 1 inputs"
        assert len(outputs) == 1, "bmmint must equal 1 output"

        input1 = inputs[0]
        input2 = inputs[1]

        input1_shape = input1.get_shape()
        input2_shape = input2.get_shape()

        batch = 1
        if len(input1) == 3:
            batch = input1_shape[0]

        i_bit_w = node.attributes["parameter_bits"]
        i_bit_x = node.attributes["data_bits"]
        o_bit = node.attributes["o_bits"]

        total_cycles = 0
        for i in range(batch):
            total_cycles += luna_profile.est_mat_mul_bias_cycles(
                row=input1_shape[-2],
                col=input1_shape[-1],
                col2=input2_shape[-1],
                i_bits_0=i_bit_w,
                i_bits_1=i_bit_x,
                o_bits=o_bit,
            )

        return total_cycles, 0

    def _estimate_iqsigmoid_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        assert len(inputs) == 1, "iqSigmoid must equal 1 inputs"
        Q_INPUT = 27
        input1 = inputs[0]
        i_bit = node.attributes["data_bits"]
        o_bit = node.attributes["o_bits"]

        total_cycles = 0
        total_cycles += luna_profile.est_vector_scale_cycles(
            size=input1.get_size(),
            i_bits_0=i_bit,
            i_bits_1=Q_INPUT,
            o_bits=o_bit,
        )

        total_cycles += luna_profile.est_sigmoid_cycles(
            size=input1.get_size(), i_bits=i_bit, o_bits=o_bit
        )

        return total_cycles, 0

    def _estimate_iqtanh_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        assert len(inputs) == 1, "iqSigmoid must equal 1 inputs"
        Q_INPUT = 27
        input1 = inputs[0]
        i_bit = node.attributes["data_bits"]
        o_bit = node.attributes["o_bits"]

        total_cycles = 0
        total_cycles += luna_profile.est_vector_scale_cycles(
            size=input1.get_size(),
            i_bits_0=i_bit,
            i_bits_1=Q_INPUT,
            o_bits=o_bit,
        )

        total_cycles += luna_profile.est_tanh_cycles(
            size=input1.get_size(), i_bits=i_bit, o_bits=o_bit
        )

        return total_cycles, 0

    def _estimate_tanh_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        total_cycles = 0
        logging.warning(
            f"Unknown operator type: {node.op_type}, name: {node.name}. Assuming 0 cycles."
        )
        total_cycles += luna_profile.est_tanh_cycles()
        return total_cycles, 0

    def _estimate_iqpad_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):

        input1 = inputs[0]
        output = outputs[0]
        input1_shape = input1.get_shape()
        output_shape = output.get_shape()

        c_in = input1_shape[1]
        h_in = input1_shape[2]
        w_in = input1_shape[3]

        c_out = output_shape[1]
        h_out = output_shape[2]
        w_out = output_shape[3]

        total_cycles = 0
        total_cycles += luna_profile.est_mat_trans_cycles(
            row=c_in, col=h_in * w_in, i_bits=8, o_bits=8
        )
        # simplify thinker iqpad. mode:0, share memory
        total_cycles += luna_profile.est_memset_cycles(
            size=output.get_size(), i_bits=8, o_bits=8
        )

        for i in range(h_in):
            for j in range(w_in):
                total_cycles += luna_profile.est_memcpy_cycles(
                    size=c_in, i_bits=8, o_bits=8
                )

        total_cycles += luna_profile.est_mat_trans_cycles(
            row=h_out * w_out, col=c_out, i_bits=8, o_bits=8
        )

        return total_cycles, 0

    def _estimate_transpose_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        assert len(inputs) == 1, "transpose must equal 1 inputs"
        assert (
            inputs[0].get_size() == outputs[0].get_size()
        ), "transpose input and output shape must be the same"
        i_bit = node.attributes["data_bits"]
        o_bit = node.attributes["o_bits"]
        input1 = inputs[0]
        input1_shape = input1.get_shape()

        total_cycles = 0
        if input1_shape.__len__() == 2:
            total_cycles += luna_profile.est_mat_trans_cycles(
                row=input1_shape[0],
                col=input1_shape[1],
                i_bits=i_bit,
                o_bits=o_bit,
            )
        elif input1_shape.__len__() > 2:  # may communicate with thinker
            perm = node.attributes["perm"]
            total_cycles += luna_profile.est_mat_trans3d_cycles(
                d1=input1_shape[0],
                d2=input1_shape[1],
                d3=input1_shape[2],
                permte1=int(perm[0]),
                permute2=int(perm[1]),
                permute3=int(perm[2]),
                i_bits=i_bit,
                o_bits=o_bit,
            )

        return total_cycles, 0

    def _estimate_reshape_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        input1 = inputs[0]

        total_cycles = 0
        total_cycles += luna_profile.est_memcpy_cycles(size=input1.get_size())
        cpu_cycles = 139
        return total_cycles, 0, cpu_cycles

    def _estimate_flatten_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        i_bit = node.attributes["data_bits"]
        o_bit = node.attributes["o_bits"]
        input1 = inputs[0]
        total_cycles = 0
        total_cycles += luna_profile.est_memcpy_cycles(
            size=input1.get_size(), i_bits=i_bit, o_bits=o_bit
        )
        return total_cycles, 0

    def _estimate_slice_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        i_bit = node.attributes["data_bits"]
        o_bit = node.attributes["o_bits"]
        output = outputs[0]
        total_cycles = 0
        total_cycles += luna_profile.est_memcpy_cycles(
            size=output.get_size(), i_bits=i_bit, o_bits=o_bit
        )
        return total_cycles, 0

    def _estimate_iqsum_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        data_bits = node.attributes["data_bits"]
        o_bits = node.attributes["o_bits"]
        input1 = inputs[0]
        total_cycles = 0
        total_cycles += luna_profile.est_vector_sum_cycles(
            size=input1.get_size(), i_bits=8, o_bits=32
        )
        total_cycles += luna_profile.est_vector_scale_cycles(
            size=1, i_bits_0=32, i_bits_1=32, o_bits=8
        )
        return total_cycles, 0

    def _estimate_iqsub_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        # assume data in share memory
        assert len(inputs) == 2, "iqSub must equal 2 inputs"
        assert (
            input[0].get_size() == inputs[1].get_size()
        ), "iqSub input shapes must be the same"

        data_bits = node.attributes["data_bits"]
        o_bits = node.attributes["o_bits"]
        input1 = inputs[0]

        total_cycles = 0
        if data_bits != o_bits:
            total_cycles += luna_profile.est_vector_scale_cycles(
                size=input1.get_size(),
                i_bits_0=data_bits,
                i_bits_1=data_bits,
                o_bits=o_bits,
            )

        total_cycles += luna_profile.est_vector_sub_cycles(
            size=input1.get_size(),
            i_bits_0=data_bits,
            i_bits_1=data_bits,
            o_bits=o_bits,
        )
        return total_cycles, 0

    def _estimate_concat_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        num_input = len(inputs)
        total_cycles = 0

        for input in inputs:
            total_cycles += luna_profile.est_memcpy_cycles(size=input.get_size())

        cpu_cycles = 3325 / 8 * num_input
        return total_cycles, 0, cpu_cycles

    def _estimate_topn_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        total_cycles = 0

        leading = 1
        input = inputs[0]
        once_size = input_shape[-1] if axis == -1 else input_shape[axis]

        axis = node.attributes["axis"]
        input_shape = input.get_shape()

        if axis == -1 or len(input_shape) - 1 == axis:
            leading = input.get_size() / input_shape[-1]

        for i in range(leading):
            total_cycles += luna_profile.est_vector_max_cycles(
                size=once_size, o_bits=32
            )

        return total_cycles, 0

    def _estimate_topn2_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        total_cycles = 0

        leading = 1
        axis = node.attributes["axis"]
        once_size = input_shape[-1] if axis == -1 else input_shape[axis]

        input = inputs[0]
        input_shape = input.get_shape()

        if axis == -1 or len(input_shape) - 1 == axis:
            leading = input_shape[-2]

        for i in range(leading):
            total_cycles += luna_profile.est_vector_max_cycles(
                size=once_size, i_bits=32, o_bits=32
            )

        return total_cycles, 0

    def _estimate_tile_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        repeats = node.attributes["repeats"]
        input_shape = inputs[0].get_shape()
        assert len(repeats) == len(input_shape), "tile"

        total_cycle = 0
        after_shape = [shape * repeats[idx] for idx, shape in enumerate(input_shape)]
        size = 1
        for shape in after_shape:
            size *= shape

        total_cycle += luna_profile.est_memcpy_cycles(size=size)
        return total_cycle, 0

    def _estimate_iqadd_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        # assume data in share memory
        assert len(inputs) == 2, "iqadd must equal 2 inputs"
        assert (
            inputs[0].get_size() == inputs[1].get_size()
        ), "iqadd input shapes must be the same"
        total_cycles = 0

        attrs = node.attributes

        scale_x = attrs["scale_x"]
        scale_y = attrs["scale_y"]
        scale_o = attrs["scale_o"]
        input1 = inputs[0]
        input2 = inputs[1]
        output = outputs[0]

        if scale_x != scale_o:
            total_cycles += luna_profile.est_vector_scale_cycles(size=input1.get_size())
        if scale_y != scale_o:
            total_cycles += luna_profile.est_vector_scale_cycles(size=input2.get_size())

        total_cycles += luna_profile.est_vector_add_cycles(
            size=input1.get_size() * len(inputs),
            i_bits_0=input1.get_bitsize(),
            i_bits_1=input2.get_bitsize(),
            o_bits=output.get_bitsize(),
        )
        cpu_cycles = 1500
        return total_cycles, 0, cpu_cycles

    def _estimate_iqcat_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        # assume data in share memory
        assert len(inputs) > 1, "iqcat must greater 1 inputs"
        input0 = inputs[0]
        output = outputs[0]
        output_shape = output.get_shape()
        axis = node.attributes["axis"]
        scale_o = node.attributes["scale_o"]

        diffs = 0
        total_cycles = 0
        input_num = len(inputs)
        for i in range(input_num):
            if node.attributes["scale_x_" + str(i)] != scale_o:
                diffs += 1

        leading = 1
        for i in range(axis):
            leading *= output.get_shape()[i]

        trailing = 1
        for i in range(axis + 1, len(output_shape)):
            trailing *= output_shape[i]

        for i in range(diffs):
            total_cycles += luna_profile.est_vector_scale_cycles(size=input0.get_size())

        if output.dtype == np.int8:
            if leading == 1:
                for i in range(input_num):
                    total_cycles += luna_profile.est_memcpy_cycles(
                        size=input0.get_size()
                    )
            else:
                for i in range(input_num):
                    for j in range(leading):
                        total_cycles += luna_profile.est_memcpy_cycles(
                            size=trailing * output_shape[axis]
                        )
        else:
            logging.error(
                f"operator type: {node.op_type}, name: {node.name}.just support int8."
            )

        return total_cycles, 0

    def _estimate_iqmul_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        lhs = inputs[0]
        rhs = inputs[1]
        lhs_shape = lhs.get_shape()
        rhs_shape = rhs.get_shape()
        lhs_c = lhs_shape[1]
        lhs_h = lhs_shape[2]
        lhs_w = lhs_shape[3]
        attr = node.attributes
        data_bit = attr.get("data_bit", 8)
        o_bits = attr.get("o_bits", 8)

        total_cycles = 0
        if (
            len(lhs_shape) == 4
            and rhs_shape.__len__()
            and lhs_shape[1] == rhs_shape[1]
            and rhs_shape[2] == 1
            and rhs_shape[2] == rhs_shape[3]
        ):
            total_cycles += luna_profile.est_memset_cycles(
                size=lhs_h * lhs_w, i_bits=8, o_bits=8
            )
            total_cycles += luna_profile.est_mat_mul_bias_cycles(
                row=lhs_c, col1=1, col2=lhs_h * lhs_w
            )
            total_cycles += luna_profile.est_vector_mul_cycles(
                size=lhs_c * lhs_h * lhs_w
            )
        elif lhs_shape.__len__() == 0:
            total_cycles += luna_profile.est_vector_scale_cycles(
                size=lhs.get_size(), i_bits_0=data_bit, i_bits_1=data_bit, o_bits=o_bits
            )
        else:
            total_cycles += luna_profile.est_vector_mul_cycles(
                size=lhs.get_size(), i_bits_0=data_bit, i_bits_1=data_bit, o_bits=o_bits
            )
        return total_cycles, 0

    def _estimate_iqdiv_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        attr = node.attributes
        data_bit = attr.get("data_bit", 8)
        o_bits = attr.get("o_bits", 8)

        total_cycles = 0
        if len(inputs) == 2:
            total_cycles += luna_profile.est_vector_div_cycles(
                size=inputs[0].get_size(),
                i_bits_0=data_bit,
                i_bits_1=data_bit,
                o_bits=o_bits,
            )
        elif len(inputs) == 1:
            total_cycles += luna_profile.est_vector_scale_cycles(
                size=inputs[0].get_size(),
                i_bits_0=data_bit,
                i_bits_1=data_bit,
                o_bits=o_bits,
            )
            logging.warning(
                f"operator type: {node.op_type}, name: {node.name}. len(inputs) == 1, error"
            )
        else:
            logging.warning(
                f"operator type: {node.op_type}, name: {node.name}. len(inputs) > 2, error"
            )
        return total_cycles, 0

    def _estimate_gather_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        data_bits = node.attributes["data_bits"]
        o_bits = node.attributes["o_bits"]
        output = outputs[0]

        total_cycles = 0
        total_cycles += luna_profile.est_memcpy_cycles(
            size=output.get_size(), i_bits=data_bits, o_bits=o_bits
        )

        return total_cycles, 0

    def _estimate_quant_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        input1 = inputs[0]
        data_bits = node.attributes["data_bits"]
        o_bits = node.attributes["o_bits"]

        total_cycles = 0
        # total_cycles += luna_profile.est_vector_scale_cycles(
        #     input1.get_size(), i_bits_0=data_bits, i_bits_1=data_bits, o_bits=o_bits
        # )
        return total_cycles, 0

    def _estimate_dequant_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        input1 = inputs[0]
        data_bits = node.attributes["data_bits"]
        o_bits = node.attributes["o_bits"]

        total_cycles = 0
        total_cycles += luna_profile.est_vector_scale_cycles(
            input1.get_size(), i_bits_0=data_bits, i_bits_1=data_bits, o_bits=o_bits
        )
        return total_cycles, 0

    def _estimate_softmaxint_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        Q_INPUT = 27
        input1 = inputs[0]
        data_bits = node.attributes["data_bits"]
        o_bits = node.attributes["o_bits"]

        total_cycles = 0
        total_cycles += luna_profile.est_vector_scale_cycles(
            size=input1.get_size(),
            i_bits_0=data_bits,
            i_bits_1=Q_INPUT,
            o_bits=o_bits,
        )
        total_cycles += luna_profile.est_softmax_cycles(
            size=input1.get_size(), i_bits=data_bits, o_bits=o_bits
        )
        return total_cycles, 0

    def _estimate_shufflechannel_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        total_cycles = 0
        logging.warning(
            f"Unknown operator type: {node.op_type}, name: {node.name}. Assuming 0 cycles."
        )
        # total_cycles += luna_profile.est_shufflechannel_cycles()
        return total_cycles, 0

    def _estimate_logsoftmaxint_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        Q_INPUT = 27
        input1 = inputs[0]
        data_bits = node.attributes["data_bits"]
        o_bits = node.attributes["o_bits"]

        total_cycles = 0
        total_cycles += luna_profile.est_vector_scale_cycles(
            size=input1.get_size(),
            i_bits_0=data_bits,
            i_bits_1=Q_INPUT,
            o_bits=o_bits,
        )

        total_cycles += luna_profile.est_logsoftmax_cycles(
            size=input1.get_size(), i_bits=data_bits, o_bits=o_bits
        )
        return total_cycles, 0

    def _estimate_split_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        dims = node.attributes["num_outputs"]
        assert len(dims) == len(outputs), "dms need equal outputs"
        if dims is None:
            logging.error(
                f"operator type: {node.op_type}, name: {node.name}. attributes[num_outputs] error"
            )

        total_cycles = 0
        for output in outputs:
            total_cycles += luna_profile.est_memcpy_cycles(size=output.get_size())
        return total_cycles, 0

    def _estimate_batchnormint_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        input1 = inputs[0]
        input1_shape = input1.get_shape()
        one_batch_size = input1_shape[2] * input1_shape[3]
        total_cycles = 0
        logging.warning(
            f"operator type: {node.op_type}, name: {node.name}. bias use offset or add?"
        )

        for i in range(input1_shape[0]):
            for j in range(input1_shape[1]):
                total_cycles += luna_profile.est_vector_scale_cycles(
                    size=one_batch_size, o_bits=32
                )
                total_cycles += luna_profile.est_vector_offset_cycles(
                    size=one_batch_size, i_bits_0=32, i_bits_1=32
                )
        return total_cycles, 0

    def _estimate_reducemean_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        total_cycles = 0
        logging.warning(
            f"Unknown operator type: {node.op_type}, name: {node.name}. Assuming 0 cycles."
        )
        return total_cycles, 0

    def _estimate_requant_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        input1 = inputs[0]
        data_bits = node.attributes["data_bits"]
        o_bits = node.attributes["o_bits"]

        total_cycles = 0
        total_cycles += luna_profile.est_vector_scale_cycles(
            size=input1.get_size(),
            i_bits_0=data_bits,
            i_bits_1=data_bits,
            o_bits=o_bits,
        )

        return total_cycles, 0

    def _estimate_layernormint_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        assert len(inputs) == 3, "layernormint must 3 inputs"
        input1 = inputs[0]
        input1_shape = input1.get_shape()
        input2 = inputs[1]
        input3 = inputs[2]

        attrs = node.attributes
        axis = attrs.get("axis", -1)
        data_bits = attrs.get("data_bits", 8)
        parameter_bits = attrs.get("parameter_bits", 8)
        o_bits = attrs.get("o_bits", 8)

        dma_cycles = 0
        dma_cycles += luna_profile.est_dma_cycles(
            size=input2.get_size(), i_bits=parameter_bits, o_bits=o_bits
        )
        dma_cycles += luna_profile.est_dma_cycles(
            size=input3.get_size(), i_bits=parameter_bits, o_bits=o_bits
        )

        leading = 1
        T = 1
        if axis == -1:
            T = input1_shape[-1]
            leading = input1_shape[0] * input1_shape[1]
        elif axis == -1:
            T = input1_shape[-1] * input1_shape[-2]
            leading = input1_shape[0]
        else:
            raise NotImplementedError(
                f"operator type: {node.op_type}, name: {node.name}. not 3 dims"
            )

        total_cycles = 0
        for i in range(leading):
            total_cycles += luna_profile.est_vector_sum_cycles(
                size=T, i_bits=data_bits, o_bits=32
            )
            total_cycles += luna_profile.est_vector_mul_cycles(
                size=T, i_bits_0=data_bits, i_bits_1=data_bits, o_bits=32
            )
            total_cycles += luna_profile.est_vector_sum_cycles(
                size=T, i_bits=32, o_bits=32
            )
            # 计算完分子
            total_cycles += luna_profile.est_vector_scale_cycles(
                size=T, i_bits_0=parameter_bits, i_bits_1=parameter_bits, o_bits=32
            )
            total_cycles += luna_profile.est_vector_scale_cycles(
                size=T, i_bits_0=32, i_bits_1=32, o_bits=32
            )
            total_cycles += luna_profile.est_vector_offset_cycles(
                size=T, i_bits_0=32, i_bits_1=32, o_bits=32
            )
            total_cycles += luna_profile.est_vector_scale_cycles(
                size=T, i_bits_0=32, i_bits_1=32, o_bits=32
            )
            # weight to 32
            total_cycles += luna_profile.est_vector_scale_cycles(
                size=T, i_bits_0=parameter_bits, i_bits_1=parameter_bits, o_bits=32
            )
            total_cycles += luna_profile.est_vector_mul_cycles(
                size=T, i_bits_0=32, i_bits_1=32, o_bits=32
            )
            total_cycles += luna_profile.est_vector_add_cycles(
                size=T, i_bits_0=32, i_bits_1=32, o_bits=o_bits
            )
        return total_cycles, dma_cycles

    def _estimate_iqvar_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        input1 = inputs[0]
        input_shape = input1.get_shape()
        leading = input_shape[-3] * input_shape[-2]
        F = input_shape[-1]

        total_cycles = 0
        for i in range(leading):
            total_cycles += luna_profile.est_vector_sum_cycles(
                size=F, i_bits=32, o_bits=32
            )
            total_cycles += luna_profile.est_vector_sum_cycles(size=F, o_bits=32)
        return total_cycles, 0

    def _estimate_cast_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        total_cycles = 0
        total_cycles += luna_profile.est_vector_scale_cycles(size=inputs[0].get_size())
        return total_cycles, 0

    def _estimate_argmax_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        """Estimate cycles for argmax operator"""
        assert len(inputs) == 1, "argmax must equal 1 inputs"
        assert len(outputs) == 1, "argmax must equal 1 output"
        total_cycles = 0

        input1 = inputs[0]
        output = outputs[0]
        input1_shape = input1.get_shape()

        axis = int(node.attributes["axis"])
        n_dims = len(input1_shape)
        leading = 1
        once_size = input1_shape[-1] if axis == -1 else input1_shape[axis]

        if axis != -1 or axis != n_dims - 1:
            logging.warning(
                "argmax axis is not -1 or last dimension, leading dimension is not supported yet."
            )

        for i in range(n_dims - 1):
            leading *= input1_shape[i]

        for i in range(0, leading, once_size):
            total_cycles += luna_profile.est_vector_max_cycles(
                once_size, input1.get_bitsize(), output.get_bitsize(), 0, 0
            )

        return total_cycles, 0

    def _estimate_relu_cycles(
        self, node: Node, inputs: List[Tensor], outputs: List[Tensor]
    ):
        """Estimate cycles for relu operator"""
        assert len(inputs) == 1, "relu must equal 1 inputs"
        assert len(outputs) == 1, "relu must equal 1 output"

        input1 = inputs[0]
        output = outputs[0]

        input_shape1 = input1.get_shape()
        output_shape = output.get_shape()

        assert (
            input_shape1 == output_shape
        ), "Output shape must match input shape for relu"

        total_cycles = 0

        total_cycles += luna_profile.est_relu_cycles(
            size=input1.get_size(),
            i_bits=input1.get_bitsize(),
            o_bits=output.get_bitsize(),
        )

        cpu_cycles = 1800
        return total_cycles, 0, cpu_cycles

    def _format_speedscope(
        self, model_name: str, result: List[Tuple[str, int, int, int, int]]
    ) -> str:
        # https://www.speedscope.app/
        """
        root;conv1 5
        root;conv1;cpu 15
        root;conv1;luna 70
        root;conv1;dma 10

        root;conv2 5
        root;conv2;cpu 15
        root;conv2;luna 70
        root;conv2;dma 10
        """
        format_result = ""
        for op_name, op_cycles, cpu_cycles, dma_cycles, luna_cycles in result:
            format_result += f"{model_name};{op_name} {0}\n"
            format_result += f"{model_name};{op_name};cpu {cpu_cycles}\n"
            format_result += f"{model_name};{op_name};dma {dma_cycles}\n"
            format_result += f"{model_name};{op_name};luna {luna_cycles}\n"
            format_result += "\n"
        return format_result

    def _format_debug(
        self, model_name: str, result: List[Tuple[str, int, int, int, int]]
    ) -> str:
        format_result = ""
        for op_name, op_cycles, cpu_cycles, dma_cycles, luna_cycles in result:
            format_result += f"{model_name};{op_name}\nluna {luna_cycles}\n"
            format_result += "\n"
        return format_result

    def _format_csv(
        self, model_name: str, result: List[Tuple[str, int, int, int, int]]
    ) -> str:
        format_result = "op_name, op_cycles, cpu_cycles, dma_cycles, luna_cycles,\n"
        for op_name, op_cycles, cpu_cycles, dma_cycles, luna_cycles in result:
            format_result += (
                f"{op_name},{op_cycles},{cpu_cycles},{dma_cycles},{luna_cycles},\n"
            )
        return format_result


def main_test():
    # Example usage
    model_path = "./data.ignore/test_thinker_onnx_infer.py.temp/resnet18_shape.onnx"
    assert os.path.isfile(model_path), f"{model_path} is not a file."

    profiler = ONNXProfile()
    profiler.estimate(model_path)
    s = profiler.get_format_result()

    print(s)


##################################################################################################
# Command line interface
##################################################################################################

import argparse
import json
import os
import sys
import logging


def parse_json_parameter(json_str_or_path: str):
    """Parse JSON parameter from string or file path"""
    if os.path.isfile(json_str_or_path):
        with open(json_str_or_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return json.loads(json_str_or_path)
def main():
    args = parse_arguments()
    tprofile(args)

def tprofile(args: argparse.Namespace):
    """Main function, execute tests"""
    # Load config if provided
    # config = parse_json_parameter(args.config) if args.config else {}
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        filename="log_onnx_profile.txt",
        filemode="w",
    )
    assert os.path.isfile(args.input), f"{args.input} is not a file."
    assert args.format in [
        "speedscope",
        "csv",
        "debug",
    ], f"Format {args.format} is not supported."
    assert args.platform in ["arcs"], f"Platform {args.platform} is not supported."

    profiler = ONNXProfile()
    profiler.estimate(args)
    format_result = profiler.get_format_result(format=args.format)

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(format_result)


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ONNX Shape Inference Tool")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input ONNX model"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the output profile file",
    )
    parser.add_argument(
        "--format",
        type=str,
        help="Output format, e.g., csv, speedscope",
    )
    parser.add_argument(
        "--platform", type=str, default="arcs", help="Target platform, e.g., arcs"
    )
    parser.add_argument(
        "--test", action="store_false", help="Run test with a sample model"
    )
    parser.add_argument(
        "--config", type=str, required=False, help="Path to config json str or file"
    )
    parser.add_argument(
        "--asynchro",
        type=bool,
        default=True,
        required=False,
        help="luna/dma, sync or asynchro",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())

__all__ = ['tprofile']