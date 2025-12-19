import numpy as np
from ._ctype import tffi
from ...enum_defines import ALIGN16


class tDMA(object):  # little endian
    def __init__(self, dev_type1, dev_type2, src_tensor_ids, dst_tensor_ids, size):
        self.src_tensor = src_tensor_ids
        self.dst_tensor = dst_tensor_ids

        self.obj = tffi.new("tDMA *")
        self.obj.src_device_.type_ = dev_type1.value
        self.obj.dst_device_.type_ = dev_type2.value
        self.obj.src_tensor_id_ = src_tensor_ids
        self.obj.dst_tensor_id_ = dst_tensor_ids
        self.obj.size_ = size

    def to_bytes(self):
        # self.pack()
        return bytes(tffi.buffer(self.obj))


class tDMAList(object):
    def __init__(self, dma_list):
        # self._list                  = dma_list
        self.obj = tffi.new("tDMAList *")

        if dma_list == None:
            self.obj.count_ = 0
        else:
            self.obj.count_ = len(dma_list)
        self.obj.elem_size_ = tffi.sizeof("tDMA")
        self.obj.header_size_ = tffi.sizeof("tDMAList")
        self.obj.offset_ = self.obj.header_size_

        self.bytes = bytes(tffi.buffer(self.obj))
        if dma_list != None:
            for x in dma_list:
                self.bytes += x[0].to_bytes()

    def to_bytes(self):
        return self.bytes


__all__ = ["tDMA", "tDMAList"]
