from .._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op


class ResizeAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        self.attrs["cubic_coeff_a"] = self.attrs.get("cubic_coeff_a", -0.75)
        self.attrs["coordinate_transformation_mode"] = self.attrs.get(
            "coordinate_transformation_mode", "half_pixel"
        )
        self.attrs["mode"] = self.attrs.get("mode", "nearest")
        self.attrs["nearest_mode"] = self.attrs.get(
            "nearest_mode", "round_prefer_floor"
        )

        coord_trans_mode = self.attrs["coordinate_transformation_mode"]
        if coord_trans_mode == "half_pixel":
            coord_trans_mode = 0
        elif coord_trans_mode == "pytorch_half_pixel":
            coord_trans_mode = 1
        elif coord_trans_mode == "align_corners":
            coord_trans_mode = 2
        elif coord_trans_mode == "asymmetric":
            coord_trans_mode = 3
        else:  # tf_crop_and_resize
            coord_trans_mode = 4
        self.attrs["pack_coord_trans_mode"] = coord_trans_mode

        self.attrs["cubic_coeff_a"] = self.attrs["cubic_coeff_a"]

        mode = self.attrs["mode"]
        if mode == "nearest":
            mode = 0
        elif mode == "linear":
            mode = 1
        else:  # cubic
            mode = 2
        self.attrs["pack_mode"] = mode

        nearest_mode = self.attrs["nearest_mode"]
        if nearest_mode == "round_prefer_floor":
            nearest_mode = 0
        elif nearest_mode == "round_prefer_ceil":
            nearest_mode = 1
        elif nearest_mode == "floor":
            nearest_mode = 2
        else:  # ceil
            nearest_mode = 3
        self.attrs["pack_nearest_mode"] = nearest_mode

    def serialize(self) -> bytes:
        attrs = tffi.new("ResizeAttrs *")

        attrs.coord_trans_mode = self.attrs["coordinate_transformation_mode"]
        attrs.cubic_coeff_a = self.attrs["cubic_coeff_a"]
        attrs.mode = self.attrs["mode"]
        attrs.nearest_mode = self.attrs["nearest_mode"]

        return bytes(tffi.buffer(attrs))


@register_op
class Resize(Operator):
    def __init__(self, attrs={}):
        self.attrs = ResizeAttrs(attrs)

    def infer_tensor(self):
        inputs = self.inputs
        X = inputs[0]
        Y = None
        if inputs[2].nbytes != 0:
            shape = X.shape * inputs[2].data
            yshape = []
            for i, s in enumerate(shape):
                yshape.append(int(shape[i]))

            Y = X.clone(shape=tuple(yshape))
        elif inputs[3].nbytes != 0:
            shape = inputs[3].data
            Y = X.clone(shape=tuple(shape))
        self.outputs = [Y]


__all__ = ["Resize"]
