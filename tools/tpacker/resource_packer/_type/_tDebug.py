import numpy as np
from ._ctype import tffi


class tTensorName(object):
    def __init__(self, name):

        name = name if len(name) <= 60 else name.rsplit("/", 1)[-1][-60:]
        self.obj = tffi.new("tTensorName *")
        self.obj.name_ = np.array([ord(x) for x in name], dtype=np.uint8).tobytes()

    def to_bytes(self):
        return bytes(tffi.buffer(self.obj))


class tDebugList(object):
    def __init__(self, tensor_name_list):
        self.obj = tffi.new("tDebugList *")
        self.obj.tensor_name_count_ = len(tensor_name_list)
        self.obj.offset_ = tffi.sizeof("tDebugList")

        self.bytes = bytes(tffi.buffer(self.obj))
        for name in tensor_name_list:
            self.bytes += name.to_bytes()

    def to_bytes(self):
        return self.bytes


__all__ = ["tTensorName", "tDebugList"]
