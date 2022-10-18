import os
from cffi import FFI

tffi = FFI()

_ctype_file1 = os.path.dirname(__file__) + "/../../executor/core/operator_attrs.h"
_ctype_file2 = (
    os.path.dirname(__file__) + "/../../executor/include/thinker/thinker_type.h"
)

with open(_ctype_file1) as f:
    _ctype_cdef = f.readlines()
    _ctype_cdef = [
        x
        for x in _ctype_cdef
        if not x.startswith("#ifndef")
        and not x.startswith("#endif")
        and not x.startswith("#include")
    ]

tffi.cdef("\n".join(_ctype_cdef))

with open(_ctype_file2) as f:
    _ctype_cdef = f.readlines()
    _ctype_cdef = [
        x
        for x in _ctype_cdef
        if not x.startswith("#ifndef")
        and not x.startswith("#endif")
        and not x.startswith("#include")
    ]

tffi.cdef("\n".join(_ctype_cdef))

__all__ = ["tffi"]
