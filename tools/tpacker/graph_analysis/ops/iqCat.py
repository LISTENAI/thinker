import math
import numpy as np
from typing import Any, Dict, Optional, List
from ...graph import Tensor
from ...resource_packer._type._ctype import tffi
from ...xsympy import is_sympy
from .utils import QuantType, calc_expr
from ...enum_defines import DevType, MemType, Layout
from .base import Operator, OperatorAttrs, register_op, BaseLayout

class iqCatAttrs(OperatorAttrs):
    def __init__(self, attrs: Optional[Dict[str, Any]] = {}) -> None:
        """Initialize the iqCatAttrs with given attributes."""
        super().__init__(attrs, "iqCatAttrs")

    def checkparams(self) -> None:
        """Check if all required parameters are present."""
        assert "scale_o" in self.attrs, "Missing required attribute: scale_o"
        assert "axis" in self.attrs or "dim" in self.attrs, "Missing axis or dim attribute"
        if "dim" in self.attrs:
            self.attrs['axis'] = self.attrs['dim']

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the iqCat operation."""
        attrs = tffi.new("iqCatAttrs *")
        axis = self.attrs.get("axis", self.attrs.get("dim"))
        attrs.axis = axis
        return bytes(tffi.buffer(attrs))

@register_op
class iqCat(Operator, BaseLayout):
    def __init__(self, attrs={}):
        """Initialize the iqCat operator with given attributes."""
        self.attrs = iqCatAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        num_input = len(inputs)
        assert num_input > 1, f'iqCat requires at least two inputs, got {num_input}'

        # Process scales and data types for each input
        for i, entry in enumerate(inputs):
            scale_name = f'scale_x_{i}'
            scale_x = self.attrs.get(scale_name)
            temp = math.log(scale_x, 2)
            assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
            if entry.scale != -1:
                assert entry.scale == int(temp), "Input scale must match attribute scale_x"
            else:
                entry.scale = int(temp)
            assert entry.dtype in (np.int8, np.int16, np.int32), "Unsupported data type"

        # Determine output shape
        shape = list(inputs[0].shape)
        ndims = len(shape)
        axis = int(self.attrs["axis"])
        assert -ndims <= axis < ndims, "Axis out of bounds"
        axis = axis + ndims if axis < 0 else axis

        for num in range(1, num_input):
            assert len(inputs[0].shape) == len(inputs[num].shape), "Input shapes must have the same dimensions"
            for i in range(ndims):
                if i != axis:
                    if is_sympy(inputs[0].shape[i]) and is_sympy(inputs[num].shape[i]):
                        assert calc_expr(str(inputs[0].shape[i]), dynamic_shape) == calc_expr(str(inputs[num].shape[i]), dynamic_shape), "Incompatible dimensions"
                    else:
                        assert inputs[0].shape[i] == inputs[num].shape[i], "Incompatible dimensions"
                    shape[i] = inputs[0].shape[i]
                else:
                    shape[i] += inputs[num].shape[i]

        # Process output scale
        scale_o = self.attrs.get("scale_o", 1.0)
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"

        # Create output tensor
        Y = inputs[0].clone(shape=tuple(shape), scale=int(temp))

        # Concatenate data if all inputs have data
        if all(x.has_data() for x in inputs):
            Y.data = np.concatenate([x.data for x in inputs], axis=axis)

        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the iqCat operation."""
        inputs = self.inputs
        output = self.outputs[0]
        scale_y = self.attrs.get('scale_o', 1.0)
        platform = self.attrs.get("platform", "venus")
        axis = int(self.attrs["axis"])
        shape = list(inputs[0].shape)
        ndims = len(shape)
        trailing = 1
        for i in range(axis + 1, ndims):
            trailing *= shape[i]
        out_size = np.prod(output.shape)

        workspace_size = 0
        flag = 0
        data_type = None
        mem_flag = 0

        if platform != "venusA":
            for i, entry in enumerate(inputs):
                scale_name = f'scale_x_{i}'
                scale_x = self.attrs.get(scale_name)
                if data_type is not None:
                    assert data_type == entry.dtype, "All inputs must have the same data type"
                else:
                    data_type = entry.dtype

                if entry.mem_type != MemType.SHARE_MEM:
                    mem_flag = 1
                if scale_x != scale_y:
                    flag = 1

            if flag:
                if data_type == np.int8:
                    if mem_flag or output.mem_type != MemType.SHARE_MEM:
                        workspace_size += trailing * output.shape[axis]
                else:
                    assert False, "iqCat does not support this data type"
        else:
            if output.mem_type != MemType.SHARE_MEM:
                workspace_size = out_size
            workspace_size = min(workspace_size, 65536)

        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

    def sub_layout_convert(self):
        """Convert the layout of input and output tensors if necessary."""
        inputs = self.inputs
        if inputs[0].layout == Layout.NHWC:
            axis = int(self.attrs["axis"])
            if axis == 1:
                axis = 3
            elif axis == 2:
                axis = 1
            elif axis == 3:
                axis = 2
            self.attrs["axis"] = axis

__all__ = ["iqCat"]