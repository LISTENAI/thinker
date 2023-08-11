import math
from ...enum_defines import DevType
from .base import iqBinaryOperator, register_op


@register_op
class iqSub(iqBinaryOperator):
    pass


__all__ = ["iqSub"]
