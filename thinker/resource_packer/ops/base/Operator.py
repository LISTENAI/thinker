from enum import Enum
from typing import Any, Dict, List, Optional

from ....graph import Tensor
from ..._type._ctype import tffi
from .OperatorLayout import BaseLayout
from ....enum_defines import Layout, DevType


class OperatorAttrs(object):
    def __init__(self, attrs: Optional[Dict[str, Any]] = {}, attr_name: str = "Attrs"):
        self.attrs: Dict = attrs
        # attrs c struct
        self.attr_struct_name: str = attr_name
        self.normalize()
        self.checkparams()

    def normalize(self) -> None:
        pass

    def checkparams(self) -> None:
        pass

    def _auto_assign(self, src_attrs, dst_attrs):
        attr_dict = dir(dst_attrs)
        for key in attr_dict:
            if key not in src_attrs.keys():
                raise ("missing attrs", key)
            value = src_attrs[key]
            if isinstance(value, Enum):
                value = value.value
            setattr(dst_attrs, key, value)

    def serialize(self) -> bytes:
        try:
            attrs = tffi.new(self.attr_struct_name + " *")
            attrs = tffi.new(self.attr_struct_name + " *")
            self._auto_assign(self.attrs, attrs)
        except:
            return b""
        return bytes(tffi.buffer(attrs))

    def __getitem__(self, x: str):
        return self.attrs[x]

    def __setitem__(self, x: str, val):
        self.attrs[x] = val

    def get(self, key, default=None):
        if key not in self.attrs:
            return default
        return self.attrs[key]


class Operator(BaseLayout):
    def __init__(self, attrs: Optional[Dict[str, Any]] = {}):
        self.init_ops()
        self.attrs = OperatorAttrs(attrs)

    def init_ops(self, inputs: List[Tensor] = [], outputs: List[Tensor] = []):
        self.inputs: List[Tensor] = [input for input in inputs]
        self.outputs: List[Tensor] = [output for output in outputs]
        self.workspace: List[Tensor] = []

    def set_input(self, inputs: List[Tensor]):
        self.inputs = inputs

    def get_output(self) -> List[Tensor]:
        return self.outputs

    def get_workspace(self, dev_type: DevType) -> List[Tensor]:
        return self.workspace

    def pack_attrs(self) -> bytes:
        return self.attrs.serialize()

    # need implement function
    def infer_tensor(self):
        raise NotImplementedError

    def is_inplace(self):
        return False

    # sub class convert, as, concat split slice
    def sub_layout_convert(self):
        pass

    def layout_convert(self):
        for output in self.outputs:
            layout = output.layout
            shape = output.shape
            if layout == Layout.NHWC:
                output.shape = shape[0], shape[2], shape[3], shape[1]
            if layout == Layout.NCWH:
                output.shape = shape[0], shape[1], shape[3], shape[2]
                output.layout = Layout.NCWH
        self.sub_layout_convert()

    def pack_params(self, dev_type: DevType):
        pass


_OPERATORS = {}


def register_op(name=None):
    def decorator(cls, _name):
        if _name in _OPERATORS:
            raise LookupError("Operator %s already registered!" % _name)
        _OPERATORS[_name] = cls
        return cls

    if type(name) != str:
        cls = name
        decorator(cls, cls.__name__)
        return cls
    return lambda cls: decorator(cls, name)


def create_operator(op_type, attrs={}, inputs=[], outputs=[]):
    convert_op = _OPERATORS.get(op_type, None)
    if convert_op is not None:
        op = convert_op(attrs)
        op.init_ops(inputs, outputs)
        return op
    else:
        return None


__all__ = ["Operator", "OperatorAttrs", "register_op", "create_operator"]
