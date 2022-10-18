import math
import numpy as np
from typing import Any, Dict, Optional

from ...graph import Tensor
from ...enum_defines import DevType
from .base import UnaryOperator, OperatorAttrs, register_op


@register_op
class Dequant(UnaryOperator):
    def checkparams(self) -> None:
        assert "scale_o" in self.attrs

    def infer_tensor(self):
        assert len(self.inputs) in {1, 2}
        X = self.inputs[0]

        scale_x = self.attrs.get("scale_o")
        temp = math.log(scale_x, 2)
        assert abs(temp - int(temp)) < 0.000001
        assert X.scale == int(temp)

        Y = X.clone(dtype=np.dtype("f4"), scale=1.0)
        self.outputs = [Y]
        if all([x.has_data() for x in self.inputs]):
            self.forward()


__all__ = ["Dequant"]
