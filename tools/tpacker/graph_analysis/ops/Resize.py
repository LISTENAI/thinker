from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op


class ResizeAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        """Check and set default parameters for Resize operation"""
        # Set default values for parameters
        self.attrs["cubic_coeff_a"] = self.attrs.get("cubic_coeff_a", -0.75)
        self.attrs["coordinate_transformation_mode"] = self.attrs.get("coordinate_transformation_mode", "half_pixel")
        self.attrs["mode"] = self.attrs.get("mode", "nearest")
        self.attrs["nearest_mode"] = self.attrs.get("nearest_mode", "round_prefer_floor")

        # Map coordinate transformation modes to integers
        coord_trans_mode = {
            "half_pixel": 0,
            "pytorch_half_pixel": 1,
            "align_corners": 2,
            "asymmetric": 3,
            "tf_crop_and_resize": 4
        }.get(self.attrs["coordinate_transformation_mode"], 4)
        self.attrs["pack_coord_trans_mode"] = coord_trans_mode

        # Map interpolation modes to integers
        mode = {
            "nearest": 0,
            "linear": 1,
            "cubic": 2
        }.get(self.attrs["mode"], 2)
        self.attrs["pack_mode"] = mode

        # Map nearest neighbor modes to integers
        nearest_mode = {
            "round_prefer_floor": 0,
            "round_prefer_ceil": 1,
            "floor": 2,
            "ceil": 3
        }.get(self.attrs["nearest_mode"], 0)
        self.attrs["pack_nearest_mode"] = nearest_mode

    def serialize(self) -> bytes:
        """Serialize Resize attributes to bytes"""
        attrs = tffi.new("ResizeAttrs *")
        attrs.coord_trans_mode = self.attrs["pack_coord_trans_mode"]
        attrs.cubic_coeff_a = self.attrs["cubic_coeff_a"]
        attrs.mode = self.attrs["pack_mode"]
        attrs.nearest_mode = self.attrs["pack_nearest_mode"]
        return bytes(tffi.buffer(attrs))


@register_op
class Resize(Operator):
    def __init__(self, attrs={}):
        self.attrs = ResizeAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor shape based on input and resize parameters"""
        X = self.inputs[0]
        Y = None

        # Determine new shape based on input tensors
        if self.inputs[2].nbytes != 0:
            shape = X.shape * self.inputs[2].data
            yshape = [int(s) for s in shape]
        elif self.inputs[3].nbytes != 0:
            yshape = self.inputs[3].data

        if yshape is not None:
            Y = X.clone(shape=tuple(yshape))
        self.outputs = [Y]


__all__ = ["Resize"]