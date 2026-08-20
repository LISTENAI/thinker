import math
import numpy as np
from typing import Any, Dict, List, Optional

from ...graph import Tensor
from ...xsympy import is_sympy
from ...enum_defines import DevType, MemType, Layout, ALIGN2, ALIGN4, ALIGN8
from ...resource_packer._type._ctype import tffi
from .base import Operator, OperatorAttrs, register_op, BaseLayout

class TransposeAttrs(OperatorAttrs):
    def __init__(self, attrs: Optional[Dict[str, Any]] = {}) -> None:
        """Initialize Transpose attributes with given parameters."""
        super().__init__(attrs, "TransposeAttrs")

    def checkparams(self) -> None:
        """Validate and set default parameters for Transpose operation."""
        self.attrs["perm"] = self.attrs.get("perm", [])
        assert len(self.attrs["perm"]) <= 5, "Transpose runtime supports at most 5 dimensions"
        self.attrs["ndim"] = len(self.attrs["perm"])

    def serialize(self) -> bytes:
        """Serialize Transpose attributes to bytes."""
        attrs = tffi.new("TransposeAttrs *")
        attrs.axes_ = self.attrs["perm"]
        attrs.ndim_ = self.attrs["ndim"]
        return bytes(tffi.buffer(attrs))

@register_op
class Transpose(Operator):
    def __init__(self, attrs={}):
        """Initialize Transpose operator with given attributes."""
        self.attrs = TransposeAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer output tensor shape by transposing input tensor according to specified axes."""
        inputs = self.inputs
        assert len(inputs) == 1, "Transpose operation requires exactly one input"
        X = inputs[0]
        shape = list(X.shape)
        axes = self.attrs["perm"]

        if not axes:
            axes = list(range(len(shape) - 1, -1, -1))

        assert 2 <= len(shape) <= 5, "Transpose runtime supports rank 2 to 5"
        assert all(is_sympy(dim) or int(dim) > 0 for dim in shape), \
            "Transpose dimensions must be positive"
        assert len(axes) == len(shape), "Transpose perm must match the input rank"
        axes = [int(axis) for axis in axes]
        assert sorted(axes) == list(range(len(shape))), \
            "Transpose perm must be a complete permutation"
        self.attrs["perm"] = axes
        self.attrs["ndim"] = len(axes)

        def _reduce_4d_transpose(sh, perm):
            for axis in range(3):
                pos = perm.index(axis)
                if pos + 1 < 4 and perm[pos + 1] == axis + 1:
                    reduced_shape = list(sh)
                    reduced_shape[axis] *= reduced_shape[axis + 1]
                    del reduced_shape[axis + 1]
                    reduced_perm = [x for x in perm if x != axis + 1]
                    reduced_perm = tuple(x if x <= axis else x - 1 for x in reduced_perm)
                    return tuple(reduced_shape), reduced_perm
            return None

        def _venus_plain_ok(rows, cols):
            if is_sympy(rows) or is_sympy(cols):
                return True
            if X.dtype == np.int8:
                return ALIGN4(rows) * ALIGN8(cols) <= 65536
            if X.dtype == np.int16:
                return ALIGN4(rows) * ALIGN2(cols) <= 32768
            return ALIGN2(rows) * ALIGN2(cols) <= 16384

        def _venus_split_ok(rows, cols):
            if is_sympy(rows) or is_sympy(cols):
                return True
            if _venus_plain_ok(rows, cols):
                return True
            return any(rows % split == 0 and _venus_plain_ok(rows // split, cols)
                       for split in range(2, rows + 1))

        def _venus_axis_ok(axis_shape, axis_perm):
            if tuple(axis_perm) in {(0, 2, 1), (2, 0, 1), (1, 0, 2)}:
                return _venus_plain_ok(axis_shape[1], axis_shape[2])
            return _venus_plain_ok(axis_shape[0], axis_shape[2])

        platform = self.attrs.get("platform", "venus")
        if platform == "venus":
            axes_tuple = tuple(axes)
            valid_axis3 = {(0, 2, 1), (2, 0, 1), (2, 1, 0), (1, 2, 0), (1, 0, 2)}
            assert X.dtype in (np.int8, np.int16, np.int32, np.float32), \
                "Transpose on venus only supports int8/int16/int32/float32 tensors"
            if len(shape) == 2:
                assert axes_tuple == (1, 0), "Transpose on venus only supports (1, 0) for rank 2"
                assert _venus_split_ok(shape[0], shape[1]), \
                    "Transpose on venus has no valid row split within matrix capacity"
            elif len(shape) == 3:
                assert axes_tuple in valid_axis3, "Unsupported venus rank-3 transpose permutation"
                assert _venus_axis_ok(shape, axes_tuple), \
                    "Transpose on venus exceeds trans_axis matrix capacity"
            elif len(shape) == 4:
                reduced = _reduce_4d_transpose(shape, axes_tuple)
                batch_axis = tuple(a - 1 for a in axes_tuple[1:]) if axes_tuple[0] == 0 else None
                assert (reduced is not None and reduced[1] in valid_axis3) or batch_axis in valid_axis3, \
                    "Unsupported venus rank-4 transpose permutation"
                if reduced is not None and reduced[1] in valid_axis3:
                    check_shape, check_axis = reduced
                else:
                    check_shape, check_axis = tuple(shape[1:]), batch_axis
                assert _venus_axis_ok(check_shape, check_axis), \
                    "Transpose on venus exceeds trans_axis matrix capacity"
            elif len(shape) == 5:
                assert axes_tuple == (0, 2, 1, 3, 4), \
                    "Venus rank-5 transpose only supports (0, 2, 1, 3, 4)"
                assert _venus_axis_ok((shape[1], shape[2], shape[3] * shape[4]), (1, 0, 2)), \
                    "Transpose on venus exceeds trans_axis matrix capacity"
            else:
                raise AssertionError("Transpose on venus only supports rank 2, 3, 4, or the reducible rank-5 permutation")
        if platform == "arcs":
            axes_tuple = tuple(axes)
            valid_axis3 = {(0, 2, 1), (2, 0, 1), (2, 1, 0), (1, 2, 0), (1, 0, 2)}
            assert X.dtype in (np.int8, np.int32, np.float32), \
                f"Transpose on arcs only supports int8 or 4-byte tensors, got {X.dtype} for shape {X.shape}"
            if len(shape) == 2:
                assert axes_tuple == (1, 0), "Transpose on arcs only supports 2D matrix transpose for 2D tensors"
            elif len(shape) == 3:
                assert axes_tuple in valid_axis3, "Unsupported arcs 3D transpose axis"
            elif len(shape) == 4:
                reduced = _reduce_4d_transpose(shape, axes_tuple)
                batch_axis = tuple(a - 1 for a in axes_tuple[1:]) if axes_tuple[0] == 0 else None
                assert (reduced is not None and reduced[1] in valid_axis3) or batch_axis in valid_axis3, \
                    "Unsupported arcs 4D transpose axis"
            else:
                raise AssertionError("Transpose on arcs only supports 2D/3D/4D tensors")
        if platform == "venusA":
            axes_tuple = tuple(axes)
            valid_axis3 = {(0, 2, 1), (2, 0, 1), (2, 1, 0), (1, 2, 0), (1, 0, 2)}
            assert X.dtype in (np.int8, np.int16, np.int32, np.float32), \
                "Transpose on venusA only supports int8/int16 or 4-byte tensors"
            if len(shape) == 2:
                assert axes_tuple == (1, 0), "Transpose on venusA only supports 2D matrix transpose for 2D tensors"
            elif len(shape) == 3:
                assert axes_tuple in valid_axis3, "Unsupported venusA 3D transpose axis"
            elif len(shape) == 4:
                assert axes_tuple[0] == 0, "venusA rank-4 transpose requires axes[0] == 0"
                batch_axis = tuple(a - 1 for a in axes_tuple[1:])
                assert batch_axis in valid_axis3, "Unsupported venusA 4D transpose axis"
            else:
                raise AssertionError("Transpose on venusA only supports 2D/3D/4D tensors")
            def _plain_ok(m, n):
                if X.dtype == np.int8:
                    return ALIGN4(m) * ALIGN8(n) <= 65536
                if X.dtype == np.int16:
                    return ALIGN4(m) * ALIGN4(n) <= 65536
                if X.dtype in (np.int32, np.float32):
                    return ALIGN2(m) * ALIGN4(n) <= 32768
                return False
            def _axis_ok(sh, ax):
                if tuple(ax) in {(0, 2, 1), (2, 0, 1), (1, 0, 2)}:
                    return _plain_ok(sh[1], sh[2])
                return _plain_ok(sh[0], sh[2])
            if X.dtype in (np.int8, np.int16, np.int32):
                if len(shape) == 3:
                    assert _axis_ok(shape, axes_tuple), "Transpose on venusA exceeds trans_axis input limit"
                elif len(shape) == 4:
                    check_shape = tuple(shape[1:])
                    check_axis = tuple(a - 1 for a in axes_tuple[1:])
                    assert _axis_ok(check_shape, check_axis), "Transpose on venusA exceeds trans_axis input limit"
        new_shape = [shape[x] for x in axes]
        new_shape += shape[len(axes):]

        Y = X.clone(shape=tuple(new_shape), scale=X.scale)

        # Handle layout conversion
        if X.layout == Layout.NCHW and tuple(axes) == (0, 1, 3, 2):
            Y.layout = Layout.NCWH
        elif X.layout == Layout.NCWH and tuple(axes) == (0, 1, 3, 2):
            Y.layout = Layout.NCHW

        if X.has_data():
            Y.data = X.data.transpose(axes)

        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace size for the operation."""
        X = self.inputs[0]
        Y = self.outputs[0]
        platform = self.attrs.get("platform", "venus")

        workspace_size = 0
        if X.mem_type != MemType.SHARE_MEM:
            workspace_size += X.nbytes
        if Y.mem_type != MemType.SHARE_MEM:
            workspace_size += Y.nbytes

        if platform == "venus":
            if not is_sympy(workspace_size):
                assert workspace_size <= 65536, \
                    "Transpose on venus requires more SHARE_MEM workspace than the 64KB matrix capacity"

        if workspace_size != 0:
            max_workspace = Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)
            return [max_workspace]

__all__ = ["Transpose"]
