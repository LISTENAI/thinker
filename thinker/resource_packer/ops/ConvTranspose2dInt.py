import math
import numpy as np
from enum import Enum

from ...graph import Tensor
from .._type._ctype import tffi
from ...enum_defines import DevType, Layout
from .utils import get_deconv_workspace_desc
from .base import Operator, OperatorAttrs, register_op
from .utils import attr2tuple, calc_deconv2d_output_shape


class ConvTranspose2dIntAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        attrs = self.attrs
        if "auto_pad" in attrs and attrs["auto_pad"] != "NOTSET":
            auto_pad = attrs["auto_pad"]
            raise (f"[ConvTranspose2dInt] Not supported yet: auto_pad:{auto_pad}.")
        if "output_shape" in attrs and tuple(attrs["output_shape"]) != (0, 0):
            output_shape = tuple(attrs["output_shape"])
            raise (
                f"[ConvTranspose2dInt] Not supported yet: outputshape:{output_shape}."
            )
        if (
            "output_padding" in attrs
            and tuple(attrs["output_padding"]) != (0, 0)
            and tuple(attrs["output_padding"]) != (0, 0, 0, 0)
            and tuple(attrs["output_padding"]) != (0, 0, 0, 0, 0, 0)
        ):
            output_padding = tuple(attrs["output_padding"])
            raise (
                f"[ConvTranspose2dInt] Not supported yet: output_padding:{output_padding}."
            )
        else:
            output_padding = (0, 0, 0, 0, 0, 0)

        dilations = attrs.get("dilations")
        dilations = attr2tuple(dilations, (1, 1))
        assert dilations == (1, 1), "not support dilation ConvTranspose2dInt!"

        kernels = attrs.get("kernel_shape")
        kernels = attr2tuple(kernels, (1, 1))
        assert kernels[0] in {
            1,
            2,
            3,
            4,
            5,
        }, "kernel_w for ConvTranspose2dInt exceed limit"
        assert kernels[1] in {
            1,
            2,
            3,
            4,
            5,
        }, "kernel_h for ConvTranspose2dInt exceed limit"

        pads = attrs.get("pads")
        pads = attr2tuple(pads, (0, 0, 0, 0))
        assert pads[0] in {
            0,
            1,
            2,
            3,
            4,
        }, "pad_left for ConvTranspose2dInt exceed limit"
        assert pads[1] in {
            0,
            1,
            2,
            3,
            4,
        }, "pad_right for ConvTranspose2dInt exceed limit"
        assert pads[2] in {0, 1, 2, 3, 4}, "pad_up for ConvTranspose2dInt exceed limit"
        assert pads[3] in {
            0,
            1,
            2,
            3,
            4,
        }, "pad_down for ConvTranspose2dInt exceed limit"

        strides = attrs.get("strides")
        strides = attr2tuple(strides, (1, 1))
        assert strides[0] in {1, 2, 4}, "stride_h for ConvTranspose2dInt exceed limit"
        assert strides[1] in {1, 2, 4}, "stride_w for ConvTranspose2dInt exceed limit"

        if strides[0] == 2:
            assert kernels[0] in {2, 3, 4, 5}, "stride_h and kernel_h do not match"
        if strides[0] == 4:
            assert kernels[0] in {4, 5}, "stride_h and kernel_h do not match"
        if strides[1] == 2:
            assert kernels[1] in {2, 3, 4, 5}, "stride_w and kernel_w do not match"
        if strides[1] == 4:
            assert kernels[1] in {4, 5}, "stride_w and kernel_w do not match"

        group = int(attrs.get("group", 1))

        if isinstance(attrs.get("layout", "NCHW"), Enum):
            layout = attrs.get("layout", "NCHW")
        else:
            layout = Layout.from_str(attrs.get("layout", "NCHW"))
        assert layout in {Layout.NCHW, Layout.NHWC}

    def serialize(self) -> bytes:
        attrs = tffi.new("Conv2dIntAttrs *")

        attrs.dilation = self.attrs["dilations"]
        attrs.kernel = self.attrs["kernel_shape"]
        attrs.pad = self.attrs["pads"]
        attrs.stride = self.attrs["strides"]
        attrs.group = self.attrs["group"]
        attrs.layout = self.attrs["layout"].value
        attrs.quant_type = self.attrs["quant_type"].value
        attrs.act_type = self.attrs["act_type"]

        return bytes(tffi.buffer(attrs))


@register_op
class ConvTranspose2dInt(Operator):
    def __init__(self, attrs={}):
        self.attrs = ConvTranspose2dIntAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) in {2, 3}
        X = inputs[0]
        W = inputs[1]
        assert len(X.shape) == 4 and len(W.shape) == 4

        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001
        assert X.scale == int(
            temp
        ), "scale of tensor must be same with scale_x in attribute"

        scale_w = self.attrs.get("scale_w")
        temp = math.log(scale_w, 2)
        assert abs(temp - int(temp)) < 0.000001
        self.inputs[1].scale = int(temp)

        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001

        shape = calc_deconv2d_output_shape(
            input_shape=X.shape,
            weight_shape=W.shape,
            kernels=self.attrs["kernel_shape"],
            strides=self.attrs["strides"],
            dilations=self.attrs["dilations"],
            pads=self.attrs["pads"],
            groups=self.attrs["group"],
            layout=Layout.NCHW,
        )
        Y = X.clone(shape=tuple(shape), scale=int(temp))
        self.outputs = [Y]
        return self.outputs

    def get_workspace(self, dev_type):
        workspace_size = get_deconv_workspace_desc(
            None, self.inputs[0], self.inputs[1], self.attrs, self.outputs
        )
        max_workspace = Tensor.from_shape([workspace_size], np.int8, dev_type)
        return [max_workspace]

    def pack_params(self, dev_type: DevType):
        if dev_type == DevType.LUNA or dev_type == DevType.HIFI:
            kernels = self.attrs["kernel_shape"]
            kernel_h = kernels[0]
            kernel_w = kernels[1]
            assert 1 <= kernel_w <= 5
            assert 1 <= kernel_h <= 5

            dilations = self.attrs["dilations"]
            dilation_w = dilations[0]
            dilation_h = dilations[1]
            strides = self.attrs["strides"]
            stride_w = strides[0]
            stride_h = strides[1]
            assert stride_w == 1 or stride_w == 2 or stride_w == 4
            assert stride_h == 1 or stride_h == 2 or stride_h == 4
            group = self.attrs["group"]
            pads = self.attrs["pads"]
            assert 0 <= pads[0] <= 4
            assert 0 <= pads[1] <= 4
            assert 0 <= pads[2] <= 4
            assert 0 <= pads[3] <= 4

            if len(self.inputs) >= 2:
                bias = self.inputs[2]
                if bias.dtype != np.dtype(np.int32):
                    new_bias = np.zeros(bias.shape, np.dtype(np.int32))
                    for i in range(bias.shape[0]):
                        new_bias[i] = bias.data[i]
                    self.inputs[2].update(data=new_bias, dtype=np.dtype(np.int32))

            weight_data = self.inputs[1]
            if weight_data.layout == Layout.NHWC8:
                return
            else:
                kernel_num = weight_data.shape[1]
                kernel_c = weight_data.shape[0]
                assert kernel_c * group == self.inputs[0].shape[1]
                assert kernel_h == weight_data.shape[2]
                assert kernel_w == weight_data.shape[3]

                h = self.inputs[0].shape[2]
                w = self.inputs[0].shape[3]
                data_size = (
                    ((w + 8 * stride_w - 1) // (8 * stride_w))
                    * (8 * stride_w)
                    * ((kernel_c + 7) // 8)
                    * 8
                    * h
                )
                assert data_size * self.inputs[0].dtype.itemsize <= 65536

                if group != 1:
                    assert kernel_num % group == 0
                    num_input_align = ((kernel_c // group + 7) // 8) * 8
                    num_output_align = ((kernel_num + 1) // 2) * 2
                    kernel_size = (
                        num_input_align * num_output_align * kernel_h * kernel_w
                    )
                    assert kernel_size * weight_data.dtype.itemsize <= 32768
                    new_weight_data = np.zeros(
                        (group, num_output_align, num_input_align, kernel_h, kernel_w),
                        weight_data.dtype,
                    )
                    shape = (
                        group * num_output_align,
                        kernel_h,
                        kernel_w,
                        num_input_align,
                    )
                    weight_data_split = weight_data.data.reshape(
                        group, -1, kernel_num, kernel_h, kernel_w
                    )
                    weight_data_transpose = weight_data_split.transpose(0, 2, 1, 3, 4)
                    # weight_data.data = weight_data.data.reshape(group, -1, kernel_c, kernel_h, kernel_w)

                    for g in range(group):
                        for p in range(kernel_num):
                            for q in range(kernel_c // group):
                                new_weight_data[g, p, q, :, :] = weight_data_transpose[
                                    g, p, q, :, :
                                ]
                    new_weight_data = np.flip(new_weight_data, (3, 4))

                    new_weight_data = new_weight_data.transpose(0, 1, 3, 4, 2)
                    # shape = new_weight_data.shape
                    temp = new_weight_data.reshape(
                        group, -1, 2, kernel_h, kernel_w, num_input_align
                    )
                    temp1 = temp.transpose(0, 1, 3, 4, 2, 5)
                    shape1 = temp1.shape
                    temp2 = temp1.reshape(
                        shape1[0], shape1[1], shape1[2], shape1[3], shape1[4], -1, 8
                    )
                    new_weight_data = temp2.transpose(0, 1, 2, 3, 5, 4, 6)
                    self.inputs[1].update(
                        shape=shape, data=new_weight_data, layout=Layout.NHWC8
                    )
                else:
                    num_input_align = ALIGN8(kernel_c)
                    num_output_align = ((kernel_num + 1) // 2) * 2
                    kernel_size = (
                        num_input_align * num_output_align * kernel_h * kernel_w
                    )
                    assert kernel_size * weight_data.dtype.itemsize <= 32768

                    new_weight_data = np.zeros(
                        (num_output_align, num_input_align, kernel_h, kernel_w),
                        weight_data.dtype,
                    )
                    weight_data_tranpose = weight_data.data.transpose(1, 0, 2, 3)

                    for p in range(kernel_num):
                        for q in range(kernel_c):
                            new_weight_data[p, q, :, :] = weight_data_tranpose[
                                p, q, :, :
                            ]
                    new_weight_data = np.flip(new_weight_data, (2, 3))
                    # new_weight_data = np.flip(new_weight_data, 3)

                    new_weight_data = new_weight_data.transpose(0, 2, 3, 1)
                    shape = new_weight_data.shape
                    temp = new_weight_data.reshape(
                        -1, 2, kernel_h, kernel_w, num_input_align
                    )
                    new_weight_data = temp.transpose(0, 2, 3, 1, 4)
                    shape1 = new_weight_data.shape
                    new_weight_data = new_weight_data.reshape(
                        shape1[0], shape1[1], shape1[2], shape1[3], -1, 8
                    )
                    new_weight_data = new_weight_data.transpose(0, 1, 2, 4, 3, 5)
                    self.inputs[1].update(
                        shape=shape, data=new_weight_data, layout=Layout.NHWC8
                    )
                    return


__all__ = ["ConvTranspose2dInt"]
