from ._ctype import tffi


def nptype_to_thinker(nptype):
    import numpy as np

    nptype = np.dtype(nptype)
    dtype = (ord(nptype.str[-2]) << 8) + int(nptype.str[-1])
    return dtype


class tTensor(object):
    def __init__(self, tensor, mem_id, mem_offset, mem_type):
        dtype = nptype_to_thinker(tensor.dtype)

        self.obj = tffi.new("tTensor *")
        self.obj.mem_.type_ = mem_type
        self.obj.dtype_ = dtype
        self.obj.mem_id_ = mem_id
        self.obj.scale_ = tensor.scale
        self.obj.zero_ = tensor.zero
        self.obj.shape_.ndim_ = len(tensor.shape)
        self.obj.shape_.dims_ = tensor.shape
        self.obj.offset_ = mem_offset
        self.obj.layout_ = tensor.layout.value

    def to_bytes(self):
        return bytes(tffi.buffer(self.obj))


class tTensorList(object):
    def __init__(self, tensor_list):
        self._list = tensor_list
        self.obj = tffi.new("tTensorList *")
        self.obj.count_ = len(tensor_list)
        self.obj.elem_size_ = tffi.sizeof("tTensor")
        self.obj.header_size_ = tffi.sizeof("tTensorList")
        self.obj.offset_ = tffi.sizeof("tTensorList")

        self.bytes = bytes(tffi.buffer(self.obj))
        for x in self._list:
            self.bytes += x.to_bytes()

    def to_bytes(self):
        return self.bytes


__all__ = ["tTensor", "tTensorList"]
