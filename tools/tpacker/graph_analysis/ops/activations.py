import math
import numpy as np
from typing import List, Dict

from ...graph import Tensor
from ...enum_defines import DevType, MemType
from .base import UnaryOperator, OperatorAttrs, BaseLayout, register_op


@register_op
class Relu(UnaryOperator, BaseLayout):
    """Rectified Linear Unit (Relu) activation function."""
    def infer_tensor(self, dynamic_shape: Dict[str, int]):
        """Infer output tensor shape and data."""
        assert len(self.inputs) == 1, "Unary operator must have exactly one input"
        X = self.inputs[0]
        platform = self.attrs.get("platform", "venus")
        if platform == "arcs":
            assert X.dtype != np.int16, "input data type of Relu do not support int16 for arcs"
        else:
            assert X.dtype in (np.int8, np.int16, np.int32), "input data type of Relu can be int8/int16/int32"

        scale_x = self.attrs.get('scale_x')
        if scale_x is not None:
            input_scale = math.log(scale_x, 2)
            assert abs(input_scale - int(input_scale)) < 1e-6, "scale_x must be a power of 2"
            if X.scale != -1:
                assert X.scale == int(input_scale), "Input scale mismatch"
            else:
                X.scale = int(input_scale)

        scale_o = self.attrs.get("scale_o")
        output_scale = X.scale
        if scale_o is not None:
            output_scale = math.log(scale_o, 2)
            assert abs(output_scale - int(output_scale)) < 1e-6, "scale_o must be a power of 2"

        out_bits = self.attrs.get('o_bits', X.dtype.itemsize * 8)
        if out_bits == 8:
            out_dtype = np.int8
        elif out_bits == 16:
            out_dtype = np.int16
        elif out_bits == 32:
            out_dtype = np.int32
        else:
            assert False, "output type of relu must be int8 or int16 or int32"

        assert X.dtype in (np.int8, np.int16, np.int32), "Relu input must be an integer tensor"
        if platform in ("venus", "venusA", "arcs"):
            shift = int(output_scale) - int(X.scale)
            assert 0 <= shift <= 63, f"Relu on {platform} requires scale shift in [0, 63]"
        if platform == "arcs":
            assert out_dtype in (np.int8, np.int32), "Relu on arcs only supports int8/int32 output"
        Y = X.clone(dtype=out_dtype, bits=out_bits // 8, scale=int(output_scale))
        self.outputs = [Y]

    def get_workspace(self):
        """Calculate the required workspace size."""
        data = self.inputs[0]
        output = self.outputs[0]
        data_size = np.prod(data.shape)
        workspace_size = 0
        platform = self.attrs.get("platform", "venus")

        if data.mem_type != MemType.SHARE_MEM or output.mem_type != MemType.SHARE_MEM:
            assert data.dtype == np.int8 and output.dtype == np.int8, \
            "PSRAM Relu requires int8 input/output"
            workspace_size = data_size
        else:
            workspace_size = 0

        workspace_size = min(65536, workspace_size)
        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []


class PreluAttrs(OperatorAttrs):
    def __init__(self, attrs={}):
        super().__init__(attrs, "PreluAttrs")

    def checkparams(self):
        for name in ("slope", "post_shift"):
            assert name in self.attrs, f"Missing required attribute: {name}"
            assert isinstance(self.attrs[name], (int, np.integer)), f"PRelu {name} must be an integer"
            assert 0 <= self.attrs[name] <= 63, f"PRelu {name} must be in [0, 63]"
        assert self.attrs["slope"] + self.attrs["post_shift"] <= 63, \
            "PRelu slope + post_shift must not exceed 63"


class ReluxAttrs(OperatorAttrs):
    def __init__(self, attrs={}):
        super().__init__(attrs, "ReluxAttrs")

    def checkparams(self):
        for name in ("threshold", "shift"):
            assert name in self.attrs, f"Missing required attribute: {name}"
            assert isinstance(self.attrs[name], (int, np.integer)), \
                f"Relux {name} must be an integer"
        assert -128 <= self.attrs["threshold"] <= 127, \
            "Relux threshold must fit in int8"
        assert 0 <= self.attrs["shift"] <= 63, \
            "Relux shift must be in [0, 63]"


@register_op
class PRelu(UnaryOperator, BaseLayout):
    """Parametric Rectified Linear Unit (PRelu) activation function."""
    def __init__(self, attrs={}):
        super().__init__()
        self.attrs = PreluAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        super().infer_tensor(dynamic_shape)
        assert self.inputs[0].dtype in (np.int8, np.int16, np.int32), \
            "PRelu only supports integer input tensors"
        if self.attrs.get("platform", "venus") == "arcs":
            assert self.inputs[0].dtype in (np.int8, np.int32), \
                "PRelu on arcs only supports int8/int32 input"
        assert self.outputs[0].dtype == self.inputs[0].dtype, \
            "PRelu output dtype must match input dtype"

    def get_workspace(self):
        data = self.inputs[0]
        output = self.outputs[0]
        assert data.mem_type == MemType.SHARE_MEM and output.mem_type == MemType.SHARE_MEM,\
        "mem type of PRelu input/output must be share memory"
        return []

@register_op("Relux")
class ReluX(UnaryOperator, BaseLayout):
    """Parametric Rectified Linear Unit (ReluX) activation function."""
    def __init__(self, attrs={}):
        super().__init__()
        self.attrs = ReluxAttrs(attrs)

    def get_workspace(self):
        data = self.inputs[0]
        output = self.outputs[0]
        assert data.mem_type == MemType.SHARE_MEM and output.mem_type == MemType.SHARE_MEM,\
        "mem type of ReluX input/output must be share memory"
        return []

    def infer_tensor(self, dynamic_shape):
        assert len(self.inputs) == 1, "Relux operator must have exactly one input"
        X = self.inputs[0]
        platform = self.attrs.get("platform", "venus")
        assert platform in ("arcs", "venusA"), \
            "Relux is only supported on arcs and venusA"
        if platform == "arcs":
            assert X.dtype in (np.int8, np.int32), \
                "Relux on arcs only supports int8/int32 input"
        else:
            assert X.dtype == np.int8, \
                "Relux on venusA only supports int8 input"
        shift = self.attrs["shift"]
        self.outputs = [X.clone(dtype=np.int8, bits=1, scale=X.scale + shift)]

@register_op
class Sigmoid(UnaryOperator, BaseLayout):
    pass       

@register_op
class Tanh(UnaryOperator, BaseLayout):
    pass 

__all__ = ["Relu", "PRelu", "ReluX", "Sigmoid", "Tanh"]
