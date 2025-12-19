from ._ctype import tffi
from ...enum_defines import ALIGN16


class tParameter(object):
    def __init__(self, mem_type, mem_id, param_buff):

        self.obj = tffi.new("tParameter *")
        self.obj.memory_.type_ = mem_type
        # self.obj.device_.id_    = 0
        self.obj.mem_id_ = mem_id
        self.obj.offset_ = tffi.sizeof("tParameter")

        param_size = ALIGN16(len(param_buff))
        param_buff += b"\0" * (param_size - len(param_buff))

        self.obj.size_ = param_size
        self.bytes = bytes(tffi.buffer(self.obj)) + param_buff

    def to_bytes(self):
        return self.bytes


class tParameterList(object):
    def __init__(self, list):
        self._list = list
        self.obj = tffi.new("tParameterList*")
        self.obj.count_ = len(self._list)
        self.obj.elem_size_ = tffi.sizeof("tParameter")
        self.obj.header_size_ = tffi.sizeof("tParameterList")
        self.obj.offset_ = tffi.sizeof("tParameterList")

        self.bytes = bytes(tffi.buffer(self.obj))
        for x in self._list:
            self.bytes += x.to_bytes()

    def to_bytes(self):
        return self.bytes


__all__ = ["tParameter", "tParameterList"]
