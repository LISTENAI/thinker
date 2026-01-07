from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np

from ....graph import Tensor
from ....xsympy import is_sympy
from .OperatorLayout import BaseLayout
from ..utils import QuantType, calc_expr
from ....enum_defines import Layout, DevType, MemType
from ....resource_packer._type._ctype import tffi


class OperatorAttrs:
    """Base class for operator attributes handling."""
    
    def __init__(self, attrs: Optional[Dict[str, Any]] = None, attr_name: str = "Attrs"):
        self.attrs = attrs if attrs is not None else {}
        self.attr_struct_name = attr_name
        self.normalize()
        self.checkparams()

    def normalize(self) -> None:
        """Normalize attributes (to be implemented in subclasses)."""
        pass

    def checkparams(self) -> None:
        """Check attribute validity (to be implemented in subclasses)."""
        pass

    def _auto_assign(self, src_attrs, dst_attrs):
        """Automatically assign attributes from source to destination."""
        attr_dict = dir(dst_attrs)
        for key in attr_dict:
            if key not in src_attrs:
                raise ValueError(f"Missing attribute: {key}")
            value = src_attrs[key]
            if isinstance(value, Enum):
                value = value.value
            setattr(dst_attrs, key, value)

    def serialize(self) -> bytes:
        """Serialize attributes to bytes."""
        try:
            attrs = tffi.new(self.attr_struct_name + " *")
            self._auto_assign(self.attrs, attrs)
            return bytes(tffi.buffer(attrs))
        except:
            return b""

    def __getitem__(self, x: str):
        """Get attribute value by key."""
        return self.attrs[x]

    def __setitem__(self, x: str, val):
        """Set attribute value by key."""
        self.attrs[x] = val

    def get(self, key, default=None):
        """Get attribute value with default if key not found."""
        return self.attrs.get(key, default)


class Operator(BaseLayout):
    """Base class for all operators."""
    
    def __init__(self, attrs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.init_ops()
        self.attrs = OperatorAttrs(attrs)

    def init_ops(self, inputs: List[Tensor] = [], outputs: List[Tensor] = []):
        """Initialize operator with input and output tensors."""
        self.inputs = [input for input in inputs]
        self.outputs = [output for output in outputs]
        self.workspace = []

    def set_input(self, inputs: List[Tensor]):
        """Set input tensors."""
        self.inputs = inputs

    def get_output(self) -> List[Tensor]:
        """Get output tensors."""
        return self.outputs

    def get_workspace(self) -> List[Tensor]:
        """Get workspace tensors."""
        return self.workspace

    def pack_attrs(self) -> bytes:
        """Pack attributes to bytes."""
        return self.attrs.serialize()

    def infer_tensor(self, dynamic_shape):
        """Infer tensor shapes (to be implemented in subclasses)."""
        raise NotImplementedError

    def flops_counter(self, dynamic_shape: List[int]) -> int:
        """Count floating-point operations (to be implemented in subclasses)."""
        return 0

    def is_inplace(self) -> bool:
        """Check if operator is inplace."""
        return False

    def sub_layout_convert(self):
        """Subclass-specific layout conversion (to be implemented in subclasses)."""
        pass

    def layout_convert(self, op_type: str):
        """Convert layout based on operator type."""
        for output in self.outputs:
            layout = output.layout
            shape = output.shape
            if layout == Layout.NHWC:
                output.shape = (shape[0], shape[2], shape[3], shape[1]) if len(shape) == 4 else (shape[0], shape[2], shape[1])
                output.layout = Layout.NHWC
            elif layout == Layout.NCWH:
                output.shape = (shape[0], shape[1], shape[3], shape[2]) if len(shape) == 4 else (shape[0], shape[2], shape[1])
                output.layout = Layout.NCWH

        if op_type in {'Conv2dInt', 'ConvTranspose2dInt'}:
            pads = tuple(self.attrs['pads'])
            strides = tuple(self.attrs['strides'])
            dilations = tuple(self.attrs['dilations'])
            kernel_shape = tuple(self.attrs['kernel_shape'])
            if layout == Layout.NCWH:
                self.attrs['kernel_shape'] = (kernel_shape[1], kernel_shape[0])
                self.attrs['strides'] = (strides[1], strides[0])
                self.attrs['dilations'] = (dilations[1], dilations[0])
                if len(pads) == 2:
                    self.attrs['pads'] = (pads[1], pads[0])
                elif len(pads) == 4:
                    self.attrs['pads'] = (pads[1], pads[0], pads[3], pads[2])
        elif op_type in {"MaxPool", "MeanPool"}:
            pads = tuple(self.attrs['pads'])
            strides = tuple(self.attrs['strides'])
            kernel_shape = tuple(self.attrs['kernel_shape'])
            if layout == Layout.NCWH:
                self.attrs['kernel_shape'] = (kernel_shape[1], kernel_shape[0])
                self.attrs['strides'] = (strides[1], strides[0])
                if len(pads) == 2:
                    self.attrs['pads'] = (pads[1], pads[0])
                elif len(pads) == 4:
                    self.attrs['pads'] = (pads[1], pads[0], pads[3], pads[2])

        self.sub_layout_convert()

    def pack_params(self):
        """Pack operator parameters (to be implemented in subclasses)."""
        pass


_OPERATORS = {}
def register_op(name=None):
    """Decorator to register operator classes."""
    def decorator(cls, _name):
        if _name in _OPERATORS:
            raise LookupError(f"Operator {_name} already registered!")
        _OPERATORS[_name] = cls
        return cls

    if isinstance(name, str):
        return lambda cls: decorator(cls, name)
    else:
        cls = name
        decorator(cls, cls.__name__)
        return cls


def create_operator(op_type: str, attrs: Dict = {}, inputs: List[Tensor] = [], outputs: List[Tensor] = []):
    """Create an operator instance."""
    op_class = _OPERATORS.get(op_type, None)
    if op_class is not None:
        op = op_class(attrs)
        op.init_ops(inputs, outputs)
        return op
    else:
        return None


__all__ = ["Operator", "OperatorAttrs", "register_op", "create_operator"]