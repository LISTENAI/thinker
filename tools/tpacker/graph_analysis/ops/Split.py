from ...graph import *
from ...enum_defines import Layout
from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op

class SplitAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        """Check and validate Split attributes."""
        assert "axis" in self.attrs and "dims" in self.attrs, "axis and dims must be specified"

    def serialize(self) -> bytes:
        """Serialize Split attributes to bytes."""
        attrs = tffi.new("SplitAttrs *")
        attrs.axis = self.attrs["axis"]
        attrs.dims = self.attrs["dims"]
        attrs.split = self.attrs["split"]
        return bytes(tffi.buffer(attrs))

@register_op
class Split(Operator):
    def __init__(self, attrs={}):
        self.attrs = SplitAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensors based on input tensor and attributes."""
        inputs = self.inputs
        assert len(inputs) == 1, "Split expects exactly one input"
        X = inputs[0]
        axis = self.attrs["axis"]
        assert axis < len(X.shape), "axis out of bounds"

        # Set default split based on dims
        if self.attrs.get("split", None) == None:
            dims = self.attrs["dims"]
            if isinstance(dims, int):
                split = [X.shape[axis] // dims] * dims
            elif isinstance(dims, list):
                split = dims
            self.attrs["split"] = split

        split = self.attrs["split"]
        outputs = []

        for idx in split:
            shape = list(X.shape)
            shape[axis] = int(idx)
            yi = X.clone(shape=shape, scale=X.scale)
            outputs.append(yi)

        self.outputs = outputs

    def sub_layout_convert(self):
        """Adjust axis for NHWC layout."""
        if self.inputs[0].layout == Layout.NHWC:
            axis = self.attrs["axis"]
            axis_mapping = {1: 3, 2: 1, 3: 2}
            self.attrs["axis"] = axis_mapping.get(axis, axis)

__all__ = ["Split"]