import math
import numpy as np
from typing import List
from ...graph import Tensor
from ...enum_defines import DevType, MemType
from ...resource_packer._type._ctype import tffi
from .base import iqBinaryOperator, OperatorAttrs, register_op

class iqPadAttrs(OperatorAttrs):
    def checkparams(self) -> None:
        """Check if all required parameters are present and valid."""
        assert "mode" in self.attrs, "Missing required attribute: mode"
        assert self.attrs["mode"] in {'constant', 'reflect', 'replicate'}, "Invalid mode"

    def serialize(self) -> bytes:
        """Serialize the attributes into bytes for the iqPad operation."""
        attrs = tffi.new("iqPadAttrs *")
        mode = self.attrs["mode"]
        attrs.mode = 0 if mode == 'constant' else 1 if mode == 'replicate' else 2
        return bytes(tffi.buffer(attrs))

@register_op
class iqPad(iqBinaryOperator):
    def __init__(self, attrs={}):
        """Initialize the iqPad operator with given attributes."""
        self.attrs = iqPadAttrs(attrs)

    def infer_tensor(self, dynamic_shape):
        """Infer the output tensor shape based on inputs and padding."""
        inputs = self.inputs
        assert len(inputs) == 3, "iqPad operator must have exactly three inputs"

        X = inputs[0]
        pads = inputs[1]
        fill_tensor = inputs[2]
        shape = list(X.shape)

        assert 0 < len(X.shape) <= 4, "iqPad input rank must be in [1, 4]"
        assert len(pads.shape) == 1, "Padding must be a 1D tensor"
        platform = self.attrs.get("platform")
        assert platform is None or platform in {"venus", "arcs", "venusA"}, \
            "iqPad platform must be venus, arcs, or venusA"
        assert X.dtype == np.int8, "iqPad only supports int8 tensors"
        assert X.zero == 0, "iqPad only supports zero point 0"
        assert pads.dtype == np.int64 and pads.has_data(), \
            "iqPad requires constant int64 pads"
        assert fill_tensor.size == 1 and fill_tensor.has_data() and \
               np.issubdtype(fill_tensor.dtype, np.number), \
            "iqPad requires a constant scalar numeric fill value"

        pad_size = pads.shape[0]
        assert pad_size in (2, 4, 6, 8) and pad_size <= 2 * len(X.shape), \
            "iqPad pads must describe trailing input dimensions"
        if platform == "venus":
            assert len(X.shape) == 4, "iqPad on venus only supports 4D tensors"
            assert X.shape[0] == 1, "iqPad on venus only supports batch size 1"
        elif platform == "arcs":
            assert len(X.shape) in (3, 4), "iqPad on arcs only supports 3D or 4D tensors"
            if len(X.shape) == 4:
                assert X.shape[0] == 1, "iqPad on arcs only supports batch size 1"
        elif platform == "venusA":
            assert len(X.shape) < 4 or X.shape[0] == 1, \
                "iqPad on venusA requires batch size 1 for 4D tensors"
        else:
            assert 0 < len(X.shape) <= 4, "iqPad only supports rank 1 to 4"

        values = [int(v) for v in pads.data.reshape(-1)]
        assert all(v >= 0 for v in values), "iqPad does not support negative pads"
        pad_dims = pad_size // 2
        base = len(shape) - pad_dims
        for i in range(pad_dims):
            shape[base + i] += values[i] + values[i + pad_dims]

        if platform in ("venus", "arcs") or (platform == "venusA" and len(X.shape) == 4):
            assert pad_size in (4, 6, 8), \
                f"iqPad on {platform} only supports pad size 4/6/8"
            assert all(v == 0 for v in values[:pad_dims - 2] +
                       values[pad_dims:2 * pad_dims - 2]), \
                f"iqPad on {platform} cannot pad batch or channels"
        if self.attrs["mode"] == "reflect":
            for i in range(pad_dims):
                assert values[i] < X.shape[base + i] and \
                       values[i + pad_dims] < X.shape[base + i], \
                    "iqPad reflect pads must be smaller than the padded dimension"
        fill = int(fill_tensor.data.reshape(-1)[0])
        if platform == "venusA":
            assert fill == 0, "iqPad on venusA only supports zero fill"
        else:
            assert self.attrs["mode"] == "constant" or fill == 0, \
                "iqPad replicate/reflect requires an unused zero fill value"

        Y = X.clone(shape=tuple(shape))
        self.outputs = [Y]

    def get_workspace(self) -> List[Tensor]:
        """Calculate the required workspace for the iqPad operation."""
        platform = self.attrs.get("platform", "venus")
        input_data = self.inputs[0]
        fill_tensor = self.inputs[2]
        fill = int(fill_tensor.data.reshape(-1)[0])
        assert -128 <= fill <= 127, "iqPad fill value must fit in int8"
        fill_tensor.dtype = np.dtype(np.int8)
        fill_tensor.bits = np.float32(fill_tensor.dtype.itemsize)
        fill_tensor.data = np.asarray(fill, dtype=np.int8)
        pads = [int(v) for v in self.inputs[1].data.reshape(-1)]
        pad_dims = len(pads) // 2
        if platform == "venusA":
            assert fill == 0, "iqPad on venusA only supports zero fill"
        if platform == "arcs":
            assert len(input_data.shape) in (3, 4), \
                "iqPad on arcs only supports 3D or 4D tensors"
            assert len(pads) in (4, 6, 8), \
                "iqPad on arcs only supports pad size 4/6/8"
            if len(input_data.shape) == 4:
                assert input_data.shape[0] == 1, \
                    "iqPad on arcs only supports batch size 1"
            assert all(v == 0 for v in pads[:pad_dims - 2] +
                       pads[pad_dims:2 * pad_dims - 2]), \
                "iqPad on arcs cannot pad batch or channels"
            return []

        output_data = self.outputs[0]
        assert input_data.mem_type == MemType.SHARE_MEM, \
            f"iqPad on {platform} requires SHARE_MEM input"
        if platform == "venus":
            assert output_data.mem_type == MemType.SHARE_MEM, \
                "iqPad on venus requires SHARE_MEM output"
            # Venus always transposes through a workspace, including when both
            # input and output tensors are in shared memory.
            return [Tensor.from_shape([
                max(input_data.nbytes, output_data.nbytes)
            ], np.int8, MemType.SHARE_MEM)]
        else:
            assert output_data.mem_type in (MemType.SHARE_MEM, MemType.PSRAM), \
                "iqPad on venusA output must be in SHARE_MEM or PSRAM"
        if platform == "venusA" and len(input_data.shape) != 4:
            if output_data.mem_type == MemType.SHARE_MEM:
                return []
            return [Tensor.from_shape([output_data.nbytes], np.int8, MemType.SHARE_MEM)]

        assert len(input_data.shape) == 4 and input_data.shape[0] == 1, \
            f"iqPad on {platform} only supports 4D tensors with batch size 1"
        assert len(pads) in (4, 6, 8), \
            f"iqPad on {platform} only supports pad size 4/6/8"
        assert all(v == 0 for v in pads[:pad_dims - 2] +
                   pads[pad_dims:2 * pad_dims - 2]), \
            f"iqPad on {platform} cannot pad batch or channels"
        c, h, w = input_data.shape[1:]
        _, co, ho, wo = output_data.shape

        def has_valid_transpose_split(row, col):
            if ((row + 3) // 4 * 4) * ((col + 7) // 8 * 8) <= 65536:
                return True
            return any(row % split == 0 and
                       (((row // split + 3) // 4 * 4) * ((col + 7) // 8 * 8) <= 65536)
                       for split in range(2, row + 1))

        assert has_valid_transpose_split(c, h * w), \
            "iqPad on venus input transpose has no valid split"
        assert has_valid_transpose_split(ho * wo, co), \
            "iqPad on venus output transpose has no valid split"
        workspace_size = output_data.nbytes + (
            output_data.nbytes if output_data.mem_type != MemType.SHARE_MEM
            else input_data.nbytes
        )

        return [Tensor.from_shape([workspace_size], np.int8, MemType.SHARE_MEM)]

__all__ = ["iqPad"]
