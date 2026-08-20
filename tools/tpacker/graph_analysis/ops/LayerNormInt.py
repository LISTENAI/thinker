import math
import numpy as np
from typing import List

from ...graph import Tensor
from ...xsympy import is_sympy
from .utils import calc_expr, combine4bit_8bit
from ...enum_defines import DevType, Layout, MemType, ALIGN4
from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op

class LayerNormIntAttrs(OperatorAttrs):
    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the LayerNormInt operation."""
        attrs = tffi.new("LayerNormIntAttrs *")
        attrs.axis = 1
        attrs.keepdims = 1
        attrs.eps = 0.00001
        return bytes(tffi.buffer(attrs))

@register_op
class LayerNormInt(Operator):
    def __init__(self, attrs={}):
        """Initialize the LayerNormInt operator with given attributes."""
        self.attrs = LayerNormIntAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        X = self.inputs[0]
        W = self.inputs[1]

        platform = self.attrs.get("platform", "venus")
        assert platform in ("venus", "arcs", "venusA"), "Unsupported LayerNormInt platform"
        assert len(self.inputs) == 3, "LayerNormInt requires weight and bias"
        min_rank = 2 if platform == "arcs" else 3
        assert len(X.shape) >= min_rank, f"LayerNormInt on {platform} requires rank >= {min_rank}"
        if len(X.shape) > 3:
            assert int(np.prod(X.shape[:-3])) == 1, "LayerNormInt cannot flatten leading dimensions"
        assert X.zero == 0 and W.zero == 0 and self.inputs[2].zero == 0, \
            "LayerNormInt only supports zero point 0"
        # Check weight compatibility
        assert W.size in (X.shape[-1] * X.shape[-2], X.shape[-1]), "Layer norm not supported for this weight shape"
        if platform in ("arcs", "venusA"):
            assert X.dtype == np.int8, "LayerNormInt on arcs/venusA only supports int8 input"
            assert W.dtype == np.int8, "LayerNormInt on arcs/venusA only supports int8 weight storage"
            assert self.attrs.get("parameter_bits", 8) == 8, \
                "LayerNormInt on arcs/venusA only supports int8 weights"
            assert self.inputs[2].dtype == np.int32, "LayerNormInt on arcs/venusA requires int32 bias"
            assert self.inputs[2].size == W.size, "LayerNormInt bias size must match weight size"
            assert W.size <= 32767, "LayerNormInt normalization width exceeds safe integer range"
        elif platform == "venus":
            assert X.dtype == np.int8, "LayerNormInt on venus only supports int8 input"
            assert W.dtype in (np.int8, np.int16), "LayerNormInt on venus requires int8 or int16 weight"
            assert self.attrs.get("parameter_bits", W.dtype.itemsize * 8) == W.dtype.itemsize * 8, \
                "LayerNormInt weight bits must match its storage type on venus"
            assert self.inputs[2].dtype == np.int32, "LayerNormInt on venus requires int32 bias"
            assert self.inputs[2].size == W.size, "LayerNormInt bias size must match weight size"
            assert W.size <= 133144, "LayerNormInt normalization width exceeds safe integer range"

        # Process input scale
        scale_x = self.attrs.get("scale_x")
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "Input scale must be a power of 2"
        if X.scale != -1:
            assert X.scale == int(temp), "Input scale must match attribute scale_x"
        else:
            X.scale = int(temp)

        # Process weight scale
        scale_w = self.attrs.get("scale_w")
        temp = math.log(scale_w, 2)
        assert abs(temp - int(temp)) < 0.000001, "Weight scale must be a power of 2"
        W.scale = int(temp)

        # Process bias scale if present
        if len(self.inputs) == 3:
            self.inputs[2].scale = X.scale + W.scale

        # Process output scale
        scale_o = self.attrs.get("scale_o")
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "Output scale must be a power of 2"

        shift = 15 + W.scale - int(temp)
        assert 0 <= shift <= 63, "LayerNormInt output shift exceeds Luna limit"
        q_x = int(math.log(scale_x, 2))
        assert 0 <= q_x <= 15, "LayerNormInt input scale is outside safe eps range"

        # Create output tensor
        output_dtype = {8: np.int8, 16: np.int16, 32: np.int32}.get(
            int(self.attrs.get("o_bits", 8))
        )
        assert output_dtype is not None, "LayerNormInt output bits must be 8, 16 or 32"
        Y = X.clone(dtype=output_dtype, bits=int(self.attrs.get("o_bits", 8)) // 8,
                    scale=int(temp))
        assert Y.zero == 0, "LayerNormInt only supports zero point 0"
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the LayerNormInt operation."""
        x = self.inputs[0]
        w = self.inputs[1]
        w_size = int(np.prod(w.shape))
        platform = self.attrs.get("platform", "venus")
        if platform == "venusA":
            assert x.mem_type == MemType.SHARE_MEM, "LayerNormInt on venusA requires SHARE_MEM input"
            assert self.outputs[0].mem_type == MemType.SHARE_MEM, "LayerNormInt on venusA requires SHARE_MEM output"
            t = w_size
            workspace_size = ((t + 1) & ~1) * 2 + 4 + t * 4 * 2
        elif platform == "arcs":
            workspace_size = 8 + w_size * 14 + ALIGN4(w_size)
        else:
            workspace_size = 8 + w_size * 8
            if x.mem_type != MemType.SHARE_MEM:
                workspace_size += ALIGN4(w_size * x.dtype.itemsize)
            if self.outputs[0].mem_type != MemType.SHARE_MEM:
                workspace_size += ALIGN4(w_size * self.outputs[0].dtype.itemsize)
            if w.mem_type != MemType.SHARE_MEM:
                workspace_size += ALIGN4(w.nbytes)
            if self.inputs[2].mem_type != MemType.SHARE_MEM:
                workspace_size += ALIGN4(self.inputs[2].nbytes)
        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

    def pack_params(self):
        """Pack the parameters for the LayerNormInt operation, handling weight quantization."""
        weight_bits = self.attrs.get("parameter_bits", 8)
        data = self.inputs[1].data
        shape = self.inputs[1].shape
        if weight_bits == 4:
            data = combine4bit_8bit(data)
        self.inputs[1].update(data=data, shape=shape, bits=np.float32(weight_bits / 8))

    def flops_counter(self, dynamic_shape) -> int:
        """Calculate the number of floating-point operations (FLOPs) for the LayerNormInt operation."""
        X = self.inputs[0]
        Y = self.outputs[0]
        xshape = list(X.shape)
        yshape = list(Y.shape)

        # Resolve symbolic expressions in shapes
        for i, s in enumerate(xshape):
            if is_sympy(s):
                xshape[i] = calc_expr(str(s), dynamic_shape)
        for i, s in enumerate(yshape):
            if is_sympy(s):
                yshape[i] = calc_expr(str(s), dynamic_shape)

        # Calculate FLOPs
        flops = int(np.prod(xshape)) + int(np.prod(yshape)) * 6
        return flops

__all__ = ["LayerNormInt"]
