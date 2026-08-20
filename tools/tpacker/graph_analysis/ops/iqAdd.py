import math
import numpy as np
from typing import List
from ...graph import Tensor
from .utils import calc_expr
from ...enum_defines import DevType, MemType
from ...xsympy import is_sympy
from .base import iqBinaryOperator, register_op, BaseLayout

@register_op
class iqAdd(iqBinaryOperator, BaseLayout):
    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape and properties based on inputs."""
        inputs = self.inputs
        assert len(inputs) == 2, "iqAdd operator must have exactly two inputs"

        X1 = inputs[0]
        X2 = inputs[1]
        platform = self.attrs.get("platform", "venus")
        for name in ("scale_x", "scale_y", "scale_o"):
            value = self.attrs.get(name, 1.0)
            assert np.isfinite(value) and value > 0, \
                f"iqAdd {name} must be finite and positive"
            exponent = math.log(value, 2)
            assert abs(exponent - round(exponent)) < 0.000001, \
                f"iqAdd {name} must be a power of 2"
        assert X1.zero == X2.zero == 0, "iqAdd only supports zero point 0"

        # Expand shapes to the same dimension
        shape1 = list(X1.shape)
        shape2 = list(X2.shape)
        if len(shape1) > len(shape2):
            shape2 = [1] * (len(shape1) - len(shape2)) + shape2
        else:
            shape1 = [1] * (len(shape2) - len(shape1)) + shape1

        assert len(shape1) == len(shape2), "Shapes must have the same dimensions after expansion"

        # Check shape compatibility
        diff_count = 0
        for i in range(len(shape1)):
            if is_sympy(shape1[i]) and is_sympy(shape2[i]):
                temp_shape1 = calc_expr(str(shape1[i]), dynamic_shape)
                temp_shape2 = calc_expr(str(shape2[i]), dynamic_shape)
                if temp_shape1 != temp_shape2:
                    diff_count += 1
            elif shape1[i] != shape2[i]:
                assert shape1[i] == 1 or shape2[i] == 1, "Incompatible dimensions"
                diff_count += 1
        venus_scalar_rhs = (
            platform == "venus" and
            X2.dtype == np.float32 and len(X2.shape) == 0
        )
        if platform != "venus" or not venus_scalar_rhs:
            assert diff_count == 0, "iqAdd runtime does not support broadcasting"

        # Process scales
        scale_x = self.attrs.get('scale_x', 1.0)
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        if X1.scale != -1:
            assert X1.scale == int(temp), "Input scale must match attribute scale_x"
        else:
            X1.scale = int(temp)

        scale_y = self.attrs.get('scale_y', 1.0)
        temp = math.log(scale_y, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"
        X2.scale = int(temp)

        scale_o = self.attrs.get("scale_o", 1.0)
        temp = math.log(scale_o, 2)
        assert abs(temp - int(temp)) < 0.000001, "Scale must be a power of 2"

        if platform == "venus":
            rhs_is_scalar = X2.dtype == np.float32 and len(X2.shape) == 0
            assert rhs_is_scalar or (
                tuple(X1.shape) == tuple(X2.shape) and
                X1.dtype == X2.dtype == np.int8
            ), "iqAdd on venus only supports same-shape int8 tensors or a float scalar rhs"
            if rhs_is_scalar:
                shape1 = list(X1.shape)
                assert 0 <= int(X1.scale) <= 30, \
                    "iqAdd scalar rhs on venus requires input scale in [0, 30]"
                shifts = (int(X1.scale) - int(temp),)
            else:
                shifts = (int(X1.scale) - int(temp), int(X2.scale) - int(temp))
            for shift in shifts:
                assert 0 <= shift <= 63, "iqAdd on venus requires q7 right shift in [0, 63]"
        elif platform == "venusA":
            assert tuple(X1.shape) == tuple(X2.shape), "iqAdd on venusA runtime does not support broadcasting"
            assert X1.dtype == X2.dtype, "iqAdd on venusA requires inputs with the same dtype"
            assert X1.dtype in (np.int8, np.int16, np.int32), "iqAdd on venusA only supports int8/int16/int32"
            qx = int(math.log(scale_x, 2))
            qy = int(math.log(scale_y, 2))
            qo = int(temp)
            max_lshift = {np.dtype("i1"): 6, np.dtype("i2"): 14, np.dtype("i4"): 30}[np.dtype(X1.dtype)]
            for shift in (qx - qo, qy - qo):
                assert shift <= 63, "iqAdd on venusA shift exceeds Luna limit"
                assert shift >= -max_lshift, "iqAdd on venusA left shift exceeds scalar range"
        elif platform == "arcs":
            assert X1.dtype == X2.dtype == np.int8, \
                "iqAdd on arcs only supports int8 inputs"
            resolved_shape1 = tuple(
                calc_expr(str(dim), dynamic_shape) if is_sympy(dim) else dim
                for dim in X1.shape
            )
            resolved_shape2 = tuple(
                calc_expr(str(dim), dynamic_shape) if is_sympy(dim) else dim
                for dim in X2.shape
            )
            assert resolved_shape1 == resolved_shape2, \
                f"iqAdd on arcs does not support broadcasting: {resolved_shape1} vs {resolved_shape2}"
            qx = int(math.log(scale_x, 2))
            qy = int(math.log(scale_y, 2))
            qo = int(temp)
            for shift in (qx - qo, qy - qo):
                assert -6 <= shift <= 63, "iqAdd on arcs requires scale shift in [-6, 63]"

        Y = X1.clone(shape=tuple(shape1), scale=int(temp), zero=0)
        assert Y.dtype == X1.dtype, "iqAdd output dtype must match input dtype"
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the iqAdd operation."""
        x1 = self.inputs[0]
        x2 = self.inputs[1]
        size = x1.nbytes
        Y = self.outputs[0]

        scale_x = self.attrs["scale_x"]
        scale_y = self.attrs["scale_y"]
        scale_o = self.attrs["scale_o"]
        platform = self.attrs.get("platform", "venus")

        if platform == "venus" and x2.dtype == np.float32 and len(x2.shape) == 0:
            assert x1.mem_type == MemType.SHARE_MEM and Y.mem_type == MemType.SHARE_MEM, \
                "iqAdd scalar path on venus requires SHARE_MEM input and output"
            return []

        workspace_size = 0
        if platform == "venusA":
            # VenusA stages PSRAM data in typed buffers.  Reserve two chunks
            # for the two scaled inputs when needed, and one for PSRAM output.
            input_staging = 0
            if x1.mem_type != MemType.SHARE_MEM or scale_x != scale_o:
                input_staging += x1.nbytes
            if x2.mem_type != MemType.SHARE_MEM or scale_y != scale_o:
                input_staging += x2.nbytes
            output_staging = Y.nbytes if Y.mem_type != MemType.SHARE_MEM else 0
            workspace_size = max(input_staging, output_staging)
            if output_staging and input_staging:
                workspace_size = max(workspace_size, 2 * max(x1.nbytes, x2.nbytes))
            workspace_size = min(workspace_size, 65536)
        elif platform == "venus":
            # Venus platform workspace calculation based on iqadd.h logic
            # Check if inputs need PSRAM to SHARE_MEM copy or scale conversion
            x1_need_workspace = (scale_x != scale_o) or (x1.mem_type != MemType.SHARE_MEM)
            x2_need_workspace = (scale_y != scale_o) or (x2.mem_type != MemType.SHARE_MEM)
            y_in_psram = (Y.mem_type != MemType.SHARE_MEM)

            if y_in_psram:
                # Output in PSRAM needs workspace for computation
                if x1_need_workspace and x2_need_workspace:
                    # Need space for Y + processed X1 + processed X2 = 2*size
                    # (Y at offset 0, X2 at offset size since X1 already processed in-place)
                    workspace_size = size * 2
                else:
                    # Only need space for Y result before copying to PSRAM
                    workspace_size = size
            else:
                # Output in SHARE_MEM, only need workspace for input processing
                if x1_need_workspace and x2_need_workspace:
                    # Need space for both processed inputs
                    workspace_size = size
        elif platform == "arcs":
            # Arcs platform workspace calculation based on arcs/iqadd.h implementation
            # Arcs uses chunked processing and requires workspace for PSRAM inputs/scale conversion

            x1_in_psram = (x1.mem_type != MemType.SHARE_MEM)
            x2_in_psram = (x2.mem_type != MemType.SHARE_MEM)
            y_in_psram = (Y.mem_type != MemType.SHARE_MEM)

            scale_x_eq = (scale_x == scale_o)
            scale_y_eq = (scale_y == scale_o)

            workspace_size = 0

            if y_in_psram:
                # Y in PSRAM: need workspace for temporary output
                workspace_size = size
                # If need to process either input, need additional space
                if ((not scale_x_eq) or x1_in_psram) and ((not scale_y_eq) or x2_in_psram):
                    workspace_size = size * 2
            elif ((not scale_x_eq) or x1_in_psram) and ((not scale_y_eq) or x2_in_psram):
                workspace_size = size

            workspace_size = min(workspace_size, 65536)

        if workspace_size != 0:
            return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]
        return []

__all__ = ["iqAdd"]
