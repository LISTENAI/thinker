import numpy as np

from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, BaseLayout, register_op


class ClipAttrs(OperatorAttrs):
    """Attributes handler for Clip operator."""

    def normalize(self) -> None:
        self.attrs["min"] = float(self.attrs.get("min", -np.inf))
        self.attrs["max"] = float(self.attrs.get("max", np.inf))

    def checkparams(self) -> None:
        assert not np.isnan(self.attrs["min"]) and not np.isnan(self.attrs["max"]), \
            "Clip bounds cannot be NaN"
        assert self.attrs["min"] <= self.attrs["max"], "Clip min must be <= max"
    
    def serialize(self) -> bytes:
        """Serialize attributes to bytes."""
        attrs = tffi.new("ClipAttrs *")
        attrs.min = self.attrs["min"]
        attrs.max = self.attrs["max"]
        return bytes(tffi.buffer(attrs))


@register_op
class Clip(Operator, BaseLayout):
    """Clip operator to limit tensor values within specified bounds."""
    
    def __init__(self, attrs=None):
        super().__init__()
        self.attrs = ClipAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor with clipped values."""
        assert len(self.inputs) in (1, 3), "Clip operator expects 1 or 3 input tensors"
        X = self.inputs[0]
        assert X.dtype in (np.int8, np.int16, np.int32, np.float32), \
            "Clip input must be int8, int16, int32, or float32"

        if len(self.inputs) == 3:
            XMin, XMax = self.inputs[1:]
            assert XMin is not None and XMax is not None, \
                "Clip requires both min and max inputs"
            assert XMin.dtype == X.dtype and XMax.dtype == X.dtype, \
                "Clip min/max dtype must match the input dtype"
            assert XMin.size == 1 and XMax.size == 1, \
                "Clip min/max inputs must be scalar tensors"

        Y = X.clone()

        if X.has_data() and len(self.inputs) == 1:
            min_val, max_val = self.attrs["min"], self.attrs["max"]
            if np.issubdtype(X.dtype, np.integer):
                limits = np.iinfo(X.dtype)
                min_val = min(max(min_val, limits.min), limits.max)
                max_val = min(max(max_val, limits.min), limits.max)
                self.attrs["min"], self.attrs["max"] = min_val, max_val
            Y.data = np.clip(X.data, min_val, max_val).astype(X.dtype)
        elif len(self.inputs) == 1 and np.issubdtype(X.dtype, np.integer):
            limits = np.iinfo(X.dtype)
            self.attrs["min"] = min(max(self.attrs["min"], limits.min), limits.max)
            self.attrs["max"] = min(max(self.attrs["max"], limits.min), limits.max)
        elif all(tensor.has_data() for tensor in self.inputs):
            min_val = self.inputs[1].data.item()
            max_val = self.inputs[2].data.item()
            assert min_val <= max_val, "Clip min input must be <= max input"
            Y.data = np.clip(X.data, min_val, max_val)

        self.outputs = [Y]


__all__ = ["Clip"]
