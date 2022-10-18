import math
import numpy as np

from ...graph import Tensor
from .utils import QuantType
from .._type._ctype import tffi
from ...enum_defines import DevType, Layout, ALIGN2, ALIGN8, ALIGN16
from .utils import attr2tuple, calc_conv2d_output_shape
from .base import Operator, OperatorAttrs, register_op, CPUConvLayout


class Conv2dIntAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        assert "data_bits" in self.attrs
        assert "o_bits" in self.attrs
        assert "parameter_bits" in self.attrs
        assert "kernel_shape" in self.attrs
        assert "pads" in self.attrs
        assert "strides" in self.attrs
        assert "dilations" in self.attrs
        assert "group" in self.attrs
        assert "platform_quant" in self.attrs
        assert "scale_x" in self.attrs
        assert "scale_w" in self.attrs
        assert "scale_o" in self.attrs

        dilations = self.attrs.get("dilations")
        dilations = attr2tuple(dilations, (1, 1))
        assert dilations == (1, 1), "not support dilation Conv2dInt!"

        kernels = self.attrs.get("kernel_shape")
        kernels = attr2tuple(kernels, (1, 1))
        assert kernels[0] in {1, 2, 3, 4, 5}, "kernel_w for Conv2dInt exceed limit"
        assert kernels[1] in {1, 2, 3, 4, 5}, "kernel_h for Conv2dInt exceed limit"

        pads = self.attrs.get("pads")
        pads = attr2tuple(pads, (0, 0, 0, 0))
        assert pads[0] in {0, 1, 2, 3, 4}, "pad_left for Conv2dInt exceed limit"
        assert pads[1] in {0, 1, 2, 3, 4}, "pad_right for Conv2dInt exceed limit"
        assert pads[2] in {0, 1, 2, 3, 4}, "pad_up for Conv2dInt exceed limit"
        assert pads[3] in {0, 1, 2, 3, 4}, "pad_down for Conv2dInt exceed limit"

        strides = self.attrs.get("strides")
        strides = attr2tuple(strides, (1, 1))
        assert strides[0] in {1, 2, 4}, "stride_h for Conv2dInt exceed limit"
        assert strides[1] in {1, 2, 4}, "stride_w for Conv2dInt exceed limit"

        assert (
            kernels[-1] >= strides[-1] and kernels[-2] >= strides[-2]
        ), "weight and stride size of Conv2dInt do not match"
        assert (
            pads[0] <= kernels[-2] and pads[2] <= kernels[-2]
        ), "pad_h and weight_h size of Conv2dInt do not match"
        assert (
            pads[1] <= kernels[-1] and pads[3] <= kernels[-1]
        ), "pad_w and weight_w size of Conv2dInt do not match"

        assert self.attrs.get("platform_quant") == "luna_quant"
        quant_type = QuantType.from_str(self.attrs.get("platform_quant"))
        act_type = int(self.attrs.get("act_type", 0))
        self.attrs["quant_type"] = quant_type
        self.attrs["act_type"] = act_type

    def serialize(self) -> bytes:
        attrs = tffi.new("Conv2dIntAttrs *")

        attrs.dilation = self.attrs["dilations"]
        attrs.kernel = self.attrs["kernel_shape"]
        attrs.pad = self.attrs["pads"]
        attrs.stride = self.attrs["strides"]
        attrs.group = self.attrs["group"]
        attrs.quant_type = self.attrs["quant_type"].value
        attrs.act_type = self.attrs["act_type"]

        return bytes(tffi.buffer(attrs))


@register_op
class Conv2dInt(Operator, CPUConvLayout):
    def __init__(self, attrs={}):
        self.attrs = Conv2dIntAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) in {2, 3}
        X = inputs[0]
        W = inputs[1]
        if len(inputs) == 3:
            assert (
                inputs[2].dtype == np.int32
            ), "data type of bias in Conv2dInt must be int32"

        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001
        assert X.scale == int(
            temp
        ), "scale of tensor must be same with scale_x in attribute"

        scale_y = self.attrs.get("scale_w")
        temp = math.log(scale_y, 2)
        assert abs(temp - int(temp)) < 0.000001
        self.inputs[1].scale = int(temp)

        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001

        assert (
            len(X.shape) == 4 and X.dtype == np.int8
        ), "data type of input in Conv2dInt must be int8 and ndims of input shape must be 4 "
        assert (
            len(W.shape) == 4 and W.dtype == np.int8
        ), "data type of weight in Conv2dInt must be int8 and ndims of input shape must be 4 "

        group = self.attrs.get("group", 1)
        kernels = self.attrs.get("kernel_shape")

        assert (
            X.shape[1] == W.shape[1] * group
        ), "channel in from input:{}, channel in from weight:{}, group:{}".format(
            X.shape[1], W.shape[1], group
        )
        assert (
            kernels[-1] >= W.shape[-1] and kernels[-2] >= W.shape[-2]
        ), "kernel_size:{} must same with input of weight:{}".format(
            (kernels[-1], kernels[-2]), (W.shape[-1], W.shape[-2])
        )
        assert (
            X.shape[-1] >= W.shape[-1] and X.shape[-2] >= W.shape[-2]
        ), "input and weight size of Conv2dInt do not match".format(
            (X.shape[-1], X.shape[-2]), (W.shape[-1], W.shape[-2])
        )

        shape = calc_conv2d_output_shape(
            input_shape=X.shape,
            weight_shape=W.shape,
            kernels=self.attrs["kernel_shape"],
            strides=self.attrs["strides"],
            dilations=self.attrs["dilations"],
            pads=self.attrs["pads"],
            groups=self.attrs["group"],
        )

        Y = X.clone(shape=tuple(shape), scale=int(temp))
        self.outputs = [Y]

    def get_workspace(self, dev_type: DevType):
        workspace_size = 0
        input_data = self.inputs[0]
        weight = self.inputs[1]
        kernel_num = weight.shape[0]
        kernel_c = weight.shape[1]

        group = self.attrs["group"]
        group_align = ALIGN16(group)
        strides = self.attrs["strides"]
        stride_h = strides[0]
        stride_w = strides[1]
        kernels = self.attrs["kernel_shape"]
        kernel_h = kernels[0]
        kernel_w = kernels[1]
        pads = self.attrs["pads"]
        pad_l = pads[0]
        pad_r = pads[1]
        pad_t = pads[2]
        pad_b = pads[3]
        h = input_data.shape[2]
        w = input_data.shape[3]
        data_size = (
            ((w + 8 * stride_w - 1) // (8 * stride_w))
            * (8 * stride_w)
            * ((kernel_c + 7) // 8)
            * 8
            * h
        )
        out_size = self.outputs[0].nbytes

        if 1 == kernel_c and (
            group == kernel_num
            or (group_align == kernel_num and weight.layout == Layout.NHWC8)
        ):  # depthwise conv
            num_input_align = kernel_c
            num_output_align = ALIGN16(kernel_num)
            kernel_size = num_output_align * kernel_h * kernel_w
            assert kernel_size * weight.dtype.itemsize <= 32768
            workspace_size = max(out_size, 65536)
        elif group != 1:  # group conv
            assert kernel_num % group == 0
            num_input_align = ALIGN8(kernel_c)
            num_output_align = ((kernel_num // group + 1) // 2) * 2
            kernel_size = num_input_align * num_output_align * kernel_h * kernel_w
            workspace_size = max(out_size, 65536)
        elif data_size > 64 * 1024 and h * w >= 64 * 1024:  # conv split h
            k_h = kernel_h
            s_h = stride_h
            split_num = 1
            tmp_in_h = h
            ou_h = (h + pad_t + pad_b - kernel_h) // stride_h + 1
            input_limit_without_h = (
                ((w + 8 * stride_w - 1) // (8 * stride_w))
                * (8 * stride_w)
                * ((kernel_c + 7) // 8)
                * 8
            )
            while tmp_in_h * input_limit_without_h > 65536 or (ou_h % split_num) != 0:
                split_num = split_num + 1
                tmp_in_h = ((ou_h * s_h) / split_num) + k_h - s_h
            tmp_ou_h = ou_h / split_num
            out_size = math.ceil(out_size / split_num)
            workspace_size = max(out_size, 65536)
        else:
            return []

        max_workspace = Tensor.from_shape([workspace_size], np.int8, dev_type)
        max_workspace.dev_type = dev_type

        return [max_workspace]

    def pack_params(self, dev_type: DevType):
        input_data = self.inputs[0]

        kernels = self.attrs["kernel_shape"]
        kernel_h = kernels[0]
        kernel_w = kernels[1]

        strides = self.attrs["strides"]
        stride_h = strides[0]
        stride_w = strides[1]

        group = self.attrs["group"]

        if len(self.inputs) >= 3:
            bias = self.inputs[2]
            assert bias.dtype == np.int32

        weight = self.inputs[1]
        if input_data.layout == Layout.NCWH:
            weight.data = weight.data.transpose(0, 1, 3, 2)
            shape = weight.shape
            assert len(shape) == 4
            weight.shape = (shape[0], shape[1], shape[3], shape[2])

        if weight.layout == Layout.NHWC8 or weight.layout == Layout.NWHC8:
            self.inputs[1].update(
                shape=weight.shape, data=weight.data, layout=Layout.NHWC8
            )
        else:
            kernel_c = weight.shape[1]
            kernel_num = weight.shape[0]
            assert kernel_c * group == input_data.shape[1]
            assert kernel_h == weight.shape[2]
            assert kernel_w == weight.shape[3]

            h = input_data.shape[2]
            w = input_data.shape[3]
            data_size = (
                ((w + 8 * stride_w - 1) // (8 * stride_w))
                * (8 * stride_w)
                * ((kernel_c + 7) // 8)
                * 8
                * h
            )
            assert ~(
                data_size * input_data.dtype.itemsize <= 65536
                and ALIGN8(kernel_c) < 512
            ), "input size of conv exceed limit"

            if 1 == kernel_c and group == kernel_num:  # depthwise conv
                num_input_align = kernel_c
                num_output_align = ALIGN16(kernel_num)
                kernel_size = num_output_align * kernel_h * kernel_w
                group = num_output_align
                assert kernel_size * weight.dtype.itemsize <= 32768

                new_weight_data = np.zeros(
                    (num_output_align, num_input_align, kernel_h, kernel_w),
                    weight.dtype,
                )
                shape = new_weight_data.shape
                for p in range(kernel_num):
                    for q in range(kernel_c):
                        new_weight_data[p, q, :, :] = weight.data[p, q, :, :]

                temp = new_weight_data.reshape(
                    -1, 16, num_input_align, kernel_h, kernel_w
                )
                new_weight_data = temp.transpose(0, 2, 3, 4, 1)
                self.inputs[1].update(
                    shape=shape, data=new_weight_data, layout=Layout.NHWC8
                )

            elif group != 1:  # group conv
                assert kernel_num % group == 0
                num_input_align = ALIGN8(kernel_c)
                num_output_align = ((kernel_num // group + 1) // 2) * 2
                kernel_size = num_input_align * num_output_align * kernel_h * kernel_w
                assert kernel_size * weight.dtype.itemsize <= 32768

                new_weight_data = np.zeros(
                    (group, num_output_align, num_input_align, kernel_h * kernel_w),
                    weight.dtype,
                )
                shape = (group * num_output_align, kernel_h, kernel_w, num_input_align)
                weight.data = weight.data.reshape(
                    group, -1, kernel_c, kernel_h * kernel_w
                )

                for g in range(group):
                    for p in range(kernel_num // group):
                        for q in range(kernel_c):
                            new_weight_data[g, p, q, :] = weight.data[g, p, q, :]

                new_weight_data = new_weight_data.transpose(0, 1, 3, 2)
                temp = new_weight_data.reshape(
                    group, -1, 2, kernel_h * kernel_w, num_input_align
                )
                temp1 = temp.transpose(0, 1, 3, 2, 4)
                shape1 = temp1.shape
                temp2 = temp1.reshape(
                    shape1[0], shape1[1] * shape1[2], shape1[3], -1, 8
                )
                new_weight_data = temp2.transpose(0, 1, 3, 2, 4)
                self.inputs[1].update(
                    shape=shape, data=new_weight_data, layout=Layout.NHWC8
                )

            else:  # common conv
                num_input_align = ALIGN8(kernel_c)
                num_output_align = ALIGN2(kernel_num)
                kernel_size = num_input_align * num_output_align * kernel_h * kernel_w
                if (
                    kernel_size * weight.dtype.itemsize > 32768
                ) and data_size * input_data.dtype.itemsize <= 65536:
                    temp_size = num_input_align * kernel_h * kernel_w
                    max_size = 32768 // temp_size
                    if max_size % 2:
                        max_size -= 1
                    split_num = (kernel_num + max_size - 1) // max_size

                    out_size = (kernel_num + split_num - 1) // split_num
                    if out_size % 2:
                        out_size += 1
                    num_output_align = out_size * split_num

                new_weight_data = np.zeros(
                    (num_output_align, num_input_align, kernel_h, kernel_w),
                    weight.dtype,
                )

                for p in range(kernel_num):
                    for q in range(kernel_c):
                        new_weight_data[p, q, :, :] = weight.data[p, q, :, :]
                new_weight_data = new_weight_data.transpose(0, 2, 3, 1)
                shape = new_weight_data.shape
                temp = new_weight_data.reshape(
                    -1, 2, kernel_h, kernel_w, num_input_align
                )
                temp1 = temp.transpose(0, 2, 3, 1, 4)
                shape1 = temp1.shape
                temp2 = temp1.reshape(
                    shape1[0], shape1[1] * shape1[2], shape1[3], -1, 8
                )
                new_weight_data = temp2.transpose(0, 1, 3, 2, 4)
                assert input_data.layout in {Layout.NCHW, Layout.NCWH}
                if input_data.layout == Layout.NCHW:
                    self.inputs[1].update(
                        shape=shape,
                        data=new_weight_data,
                        layout=Layout.NHWC8,
                    )
                elif input_data.layout == Layout.NCWH:
                    self.inputs[1].update(
                        shape=shape, data=new_weight_data, layout=Layout.NWHC8
                    )


__all__ = ["Conv2dInt"]
