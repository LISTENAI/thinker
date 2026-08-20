import math
import numpy as np
from typing import List

from ...graph import Tensor
from ...enum_defines import DevType, MemType
from ...resource_packer._type._ctype import tffi
from .base import iqBinaryOperator, OperatorAttrs, register_op

class iqSumAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        """Check if required parameters are present."""
        assert "dims" in self.attrs, "Missing required attribute: dims"
        assert isinstance(self.attrs["dims"], int) and not isinstance(self.attrs["dims"], bool), \
            "iqSum dims must be an integer axis"

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the iqSum operation."""
        attrs = tffi.new("iqSumAttrs *")
        attrs.axis = self.attrs["dims"]
        return bytes(tffi.buffer(attrs))

@register_op
class iqSum(iqBinaryOperator):
    def __init__(self, attrs={}):
        """Initialize the iqSum operator with given attributes."""
        self.attrs = iqSumAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        assert len(inputs) == 1, "iqSum operator must have exactly one input"

        X = inputs[0]
        axis = self.attrs["dims"]
        assert isinstance(axis, int) and not isinstance(axis, bool), \
            "iqSum dims must be an integer axis"

        # Ensure axis is within valid range
        assert -len(X.shape) <= axis < len(X.shape), "Axis out of bounds"
        real_axis = axis + len(X.shape) if axis < 0 else axis
        platform = self.attrs.get("platform")
        assert platform is None or platform in {"venus", "arcs", "venusA"}, \
            "iqSum platform must be venus, arcs, or venusA"
        assert real_axis == len(X.shape) - 1, "iqSum only supports the last axis"
        assert X.dtype in (np.int8, np.int16, np.int32), \
            "iqSum only supports int8, int16, or int32 input"
        assert X.zero == 0, "iqSum only supports zero point 0"
        assert X.shape[real_axis] > 0, "iqSum reduction axis must be non-empty"

        scale_x = self.attrs.get("scale_x")
        if scale_x is not None:
            assert scale_x > 0, "iqSum scale_x must be positive"
            q_x = math.log(scale_x, 2)
            assert abs(q_x - int(q_x)) < 1e-6 and X.scale == int(q_x), \
                "iqSum scale_x must be a power of 2 matching input scale"
        scale_o = self.attrs.get("scale_o")
        assert scale_o is not None and scale_o > 0, "iqSum scale_o must be positive"
        q_o = math.log(scale_o, 2)
        assert abs(q_o - int(q_o)) < 1e-6, "iqSum scale_o must be a power of 2"
        q_o = int(q_o)
        assert 0 <= X.scale - q_o <= 63, "iqSum requires Luna shift in [0, 63]"

        # Calculate output shape
        output_shape = list(X.shape)
        output_shape[real_axis] = 1

        # Create output tensor
        Y = X.clone(shape=tuple(output_shape), scale=q_o)
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the iqSum operation."""
        shape = list(self.inputs[0].shape)
        platform = self.attrs.get("platform", "venus")
        assert platform in {"venus", "arcs", "venusA"}, \
            "iqSum platform must be venus, arcs, or venusA"
        X = self.inputs[0]
        Y = self.outputs[0]
        if platform in {"venus", "arcs"}:
            assert X.dtype == np.int8 and Y.dtype == np.int8, \
                f"iqSum on {platform} only supports int8 input and output"
        else:
            assert X.dtype in (np.int8, np.int16, np.int32) and Y.dtype == X.dtype, \
                "iqSum on venusA requires matching int8, int16, or int32 input and output"
        assert X.mem_type == MemType.SHARE_MEM, \
            f"iqSum on {platform} requires SHARE_MEM input"
        assert Y.mem_type == MemType.SHARE_MEM, \
            f"iqSum on {platform} requires SHARE_MEM output"
        if platform == "venusA":
            return []
        axis = self.attrs["dims"]
        shape[axis] = 1

        workspace_size = 1
        for s in shape:
            workspace_size *= s
        workspace_size *= 4  # Assuming 4 bytes per element

        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

__all__ = ["iqSum"]
