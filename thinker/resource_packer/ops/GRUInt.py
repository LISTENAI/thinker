import math
import numpy as np

from .._type._ctype import tffi
from ...graph import Tensor
from ...enum_defines import DevType
from .base import Operator, OperatorAttrs, register_op


class GRUIntAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        assert "scale_i" in self.attrs
        assert "scale_iw" in self.attrs
        assert "scale_hw" in self.attrs
        assert "scale_h" in self.attrs
        assert "scale_o" in self.attrs
        assert "batch_first" in self.attrs
        assert "hidden_size" in self.attrs

    def serialize(self) -> bytes:
        attrs = tffi.new("GRUIntAttrs *")

        attrs.hidden_size = self.attrs["hidden_size"]
        attrs.input_size = self.attrs["input_size"]
        attrs.layout = self.attrs.get("batch_first", 0)

        return bytes(tffi.buffer(attrs))


@register_op
class GRUInt(Operator):
    def __init__(self, attrs={}):
        self.attrs = GRUIntAttrs(attrs)
        self.weight_index = 1

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) > 3
        X = inputs[0]  # input scale
        i2h_w = inputs[self.weight_index]  ##i2h_w scale
        h2h_w = inputs[self.weight_index + 1]
        assert X.dtype == i2h_w.dtype and X.dtype == h2h_w.dtype
        assert len(X.shape) == 3
        assert len(i2h_w.shape) == 3 or len(i2h_w.shape) == 2
        assert len(h2h_w.shape) == 3 or len(h2h_w.shape) == 2

        scale_i = self.attrs.get("scale_i")
        temp = math.log(scale_i, 2)
        abs(temp - int(temp)) < 0.000001
        assert self.inputs[0].scale == int(temp)

        scale_iw = self.attrs.get("scale_iw")
        temp = math.log(scale_iw, 2)
        abs(temp - int(temp)) < 0.000001
        self.inputs[1].scale = temp

        scale_hw = self.attrs.get("scale_hw")
        temp = math.log(scale_hw, 2)
        abs(temp - int(temp)) < 0.000001
        self.inputs[2].scale = temp

        scale_h = self.attrs.get("scale_h")
        temp4 = math.log(scale_h, 2)
        assert temp4 == int(temp4)

        layout = self.attrs.get("batch_first")
        if layout == 0:
            T = X.shape[0]
            B = X.shape[1]
        else:
            B = X.shape[0]
            T = X.shape[1]
        hidden_size = self.attrs["hidden_size"]
        if layout == 0:
            yshape = list([T, B, hidden_size])
        else:
            yshape = list([B, T, hidden_size])

        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o, 2)
        abs(temp - int(temp)) < 0.000001
        Y = Tensor(shape=yshape, dtype=X.dtype, scale=temp)
        self.outputs = [Y]

        hshape = list([1, B, hidden_size])
        hidden_o = Tensor(shape=hshape, dtype=np.float32, scale=temp4)
        self.outputs.append(hidden_o)

    def get_workspace(self, dev_type):
        h2h_weight = self.inputs[self.weight_index + 1]
        hidden_size = h2h_weight.shape[1]
        max_workspace = Tensor.from_shape(
            [hidden_size * 4 * 3 ],
            np.int8,
            dev_type,
        )
        return [max_workspace]

    def pack_params(self, dev_type: DevType):
        weight_i = self.inputs[1]
        weight_h = self.inputs[2]
        data_i = weight_i.data.transpose(1, 0)
        data_h = weight_h.data.transpose(1, 0)
        self.inputs[1].update(data=data_i, shape=data_i.shape)
        self.inputs[2].update(data=data_h, shape=data_h.shape)


__all__ = ["GRUInt"]
