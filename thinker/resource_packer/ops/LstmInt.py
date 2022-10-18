import math
import numpy as np

from ...graph import Tensor
from .utils import QuantType
from .._type._ctype import tffi
from ...enum_defines import DevType
from .base import Operator, OperatorAttrs, register_op


class LstmIntAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        assert "hidden_size" in self.attrs
        assert "input_size" in self.attrs
        assert "batch_first" in self.attrs
        batch_first = self.attrs.get("batch_first", 0)
        if batch_first == 0:
            self.attrs["layout"] = 0
        else:
            self.attrs["layout"] = 1
        go_forward = self.attrs.get("go_forward", 1)
        if go_forward == 0:
            self.attrs["direction"] = 1
        elif go_forward == 1:
            self.attrs["direction"] = 0
        else:
            self.attrs["direction"] = 2

        assert "platform_quant" in self.attrs
        assert self.attrs.get("platform_quant") == "luna_quant"
        quant_type = QuantType.from_str(self.attrs.get("platform_quant"))
        act_type = int(self.attrs.get("act_type", 0))
        self.attrs["quant_type"] = quant_type

        assert "scale_i" in self.attrs
        assert "scale_iw" in self.attrs
        assert "scale_hw" in self.attrs
        assert "scale_o" in self.attrs
        assert "scale_h" in self.attrs

    def serialize(self) -> bytes:
        attrs = tffi.new("LstmIntAttrs *")

        attrs.direction = self.attrs["direction"]
        attrs.hidden_size = self.attrs["hidden_size"]
        attrs.input_size = self.attrs["input_size"]
        attrs.layout = self.attrs["layout"]
        attrs.quant_type = self.attrs["quant_type"].value

        return bytes(tffi.buffer(attrs))


@register_op
class LSTMInt(Operator):
    def __init__(self, attrs={}):
        self.attrs = LstmIntAttrs(attrs)
        self.weight_index = 1

    def infer_tensor(self):
        inputs = self.inputs
        assert len(inputs) > 3
        X = inputs[0]  # input scale
        i2h_w = inputs[self.weight_index]  ##i2h_w scale
        h2h_w = inputs[self.weight_index + 1]

        scale_i = self.attrs.get("scale_i", 1.0)
        temp = math.log(scale_i, 2)
        assert abs(temp - int(temp)) < 0.000001
        assert self.inputs[0].scale == temp

        scale_iw = self.attrs.get("scale_iw", 1.0)
        temp = math.log(scale_iw, 2)
        assert abs(temp - int(temp)) < 0.000001
        self.inputs[1].scale = temp

        scale_hw = self.attrs.get("scale_hw", 1.0)
        temp = math.log(scale_hw, 2)
        assert abs(temp - int(temp)) < 0.000001
        self.inputs[2].scale = temp

        scale_o = self.attrs.get("scale_o", 1.0)
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001

        scale_h = self.attrs.get("scale_h", 1.0)
        temp1 = math.log(scale_h, 2)
        assert temp1 == int(temp1)

        assert X.dtype == i2h_w.dtype and X.dtype == h2h_w.dtype
        assert len(X.shape) == 3
        assert len(i2h_w.shape) == 3 or len(i2h_w.shape) == 2
        assert len(h2h_w.shape) == 3 or len(h2h_w.shape) == 2
        if self.attrs["layout"] == 0:
            T = X.shape[0]
            B = X.shape[1]
        else:
            B = X.shape[0]
            T = X.shape[1]
        hidden_size = self.attrs["hidden_size"]
        if self.attrs["layout"] == 0:
            yshape = list([T, B, hidden_size])
        else:
            yshape = list([B, T, hidden_size])
        Y = X.clone(shape=tuple(yshape), scale=int(temp))
        self.outputs = [Y]

        hshape = list([1, B, hidden_size])
        hidden_o = X.clone(shape=hshape, dtype=np.float32, scale=int(temp))
        self.outputs.append(hidden_o)

        cshape = list([1, B, hidden_size])
        hidden_c = X.clone(shape=cshape, dtype=np.float32, scale=int(temp1))
        self.outputs.append(hidden_c)

    def get_workspace(self, dev_type):
        h2h_weight = self.inputs[self.weight_index + 1]
        hidden_size = h2h_weight.shape[1]
        workspace_size = hidden_size * 4 * 4
        max_workspace = Tensor.from_shape([workspace_size], np.int8, dev_type)
        return [max_workspace]

    def pack_params(self, dev_type: DevType):
        weight_i = self.inputs[1]
        weight_h = self.inputs[2]
        data_i = weight_i.data.transpose(1, 0)
        data_h = weight_h.data.transpose(1, 0)
        self.inputs[1].update(data=data_i, shape=data_i.shape)
        self.inputs[2].update(data=data_h, shape=data_h.shape)


__all__ = ["LSTMInt"]
