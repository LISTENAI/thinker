import math
import numpy as np
from typing import Any, Dict, Optional

from ...graph import Tensor

from .._type._ctype import tffi
from .utils import QuantType
from ...enum_defines import DevType, MemType, ALIGN4, ALIGN16, ALIGN32
from .base import Operator, OperatorAttrs, register_op


class LinearIntAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        assert "scale_x" in self.attrs
        assert "scale_w" in self.attrs
        assert "scale_o" in self.attrs
        assert "data_bits" in self.attrs
        assert "o_bits" in self.attrs
        assert "parameter_bits" in self.attrs
        assert "platform_quant" in self.attrs

        self.attrs["transA"] = self.attrs.get("transA", 0)
        self.attrs["transB"] = self.attrs.get("transB", 1)
        self.attrs["alpha"] = self.attrs.get("alpha", 1.0)
        self.attrs["beta"] = self.attrs.get("beta", 0.0)

    def serialize(self) -> bytes:
        attrs = tffi.new("LinearIntAttrs *")

        attrs.transA = self.attrs["transA"]
        attrs.transB = self.attrs["transB"]
        quant_type = QuantType.from_str(self.attrs.get("platform_quant", "qmax_quant"))

        attrs.quant_type = quant_type.value

        return bytes(tffi.buffer(attrs))


@register_op
class LinearInt(Operator):
    def __init__(self, attrs={}):
        self.attrs = LinearIntAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) in {2, 3}
        X = inputs[0]
        W = inputs[1]
        x_shape = list(X.shape)
        w_shape = list(W.shape)
        assert X.dtype == W.dtype

        x_h = x_shape[-2]
        x_w = x_shape[-1]

        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001
        assert self.inputs[0].scale == int(temp)

        scale_w = self.attrs.get("scale_w")
        temp = math.log(scale_w, 2)
        assert abs(temp - int(temp)) < 0.000001
        self.inputs[1].scale = temp

        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001
        self.outputs[0].scale = temp

        if not self.attrs["transA"] and self.attrs["transB"]:
            assert x_w == w_shape[-1], "x_w:{} and w_shape[-1]:{}".format(
                x_w, w_shape[-1]
            )
        elif self.attrs["transA"] and self.attrs["transB"]:
            assert x_h == w_shape[-1], "x_h:{} and w_shape[-1]:{}".format(
                x_h, w_shape[-1]
            )
        elif not self.attrs["transA"] and not self.attrs["transB"]:
            assert x_w == w_shape[0], "x_w:{} and w_shape[0]:{}".format(x_w, w_shape[0])
        else:
            assert x_h == w_shape[0]

        M = ALIGN16(x_h)
        N = ALIGN32(x_w)
        if X.dtype == np.int8:
            assert M * N < 65536, "left matmul of linearint must less 64KB"
        elif X.dtype == np.int16:
            assert M * N < 32768, "left matmul of linearint must less 64KB"
        elif X.dtype == np.int32:
            assert M * N < 16384, "left matmul of linearint must less 64KB"
        else:
            raise (f"[LinearInt] Not supported type of X.dtype:{X.dtype}.")

        # 1D
        if len(X.shape) == 1:
            shape = [w_shape[0]]
        elif len(X.shape) == 2:
            shape = [x_shape[0], w_shape[0]]
        else:
            shape = [x_shape[0], x_shape[1], w_shape[0]]

        Y = X.clone(shape=shape, dtype=X.dtype, scale=int(temp))

        self.outputs = [Y]

    def get_workspace(self, dev_type: DevType):
        if len(self.inputs) > 2:
            workspace_bytes = self.outputs[0].nbytes * self.inputs[2].dtype.itemsize
            max_workspace = Tensor.from_shape([workspace_bytes], np.int8, dev_type)
            return [max_workspace]
        elif self.inputs[0].mem_type != self.outputs[0].mem_type:
            workspace_bytes = self.outputs[0].nbytes * self.outputs[0].dtype.itemsize
            max_workspace = Tensor.from_shape([workspace_bytes], np.int8, dev_type)
            return [max_workspace]
        else:
            return []

    def pack_params(self, dev_type: DevType):
        transA = self.attrs["transA"]
        transB = self.attrs["transB"]
        if transB:
            weight = self.inputs[1]
            data = weight.data.transpose(1, 0)
            self.inputs[1].update(data=data, shape=data.shape)
            # self.attrs["transB"] = 0


__all__ = ["LinearInt"]
