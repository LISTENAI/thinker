from ._ctype import tffi
from ._crc24 import crc24_calc

from ._tIO import *
from ._tDebug import *
from ._tDMA import *
from ._tModel import *
from ._tMemory import *
from ._tTensor import *
from ._tOperator import *
from ._tParamter import *
from ._tShapeInfer import *
from ...enum_defines import ALIGN16, Colors


class tModel(object):
    def __init__(
        self,
        platform: str,
        memory: tMemoryList,
        tensor: tTensorList,
        op: tOperatorList,
        io: tIOInfo,
        param: tParameterList,
        shape:tShapeInferHdr,
        debug: tDebugList,
        dma: tDMAList,
    ):

        self.hdr = tffi.new("tModelHeader *")

        self.hdr.label_ = b"thinker10"
        self.hdr.crc32_ = 0

        pack_buff = bytes(tffi.buffer(self.hdr))
        offset = ALIGN16(len(pack_buff))
        head_size = offset
        pack_buff += b"\0" * (offset - len(pack_buff))

        self.hdr.memory_offset_ = offset
        pack_buff += memory.to_bytes()
        offset = ALIGN16(len(pack_buff))
        pack_buff += b"\0" * (offset - len(pack_buff))

        self.hdr.tensor_offset_ = offset
        pack_buff += tensor.to_bytes()
        offset = ALIGN16(len(pack_buff))
        pack_buff += b"\0" * (offset - len(pack_buff))

        self.hdr.op_offset_ = offset
        pack_buff += op.to_bytes()
        offset = ALIGN16(len(pack_buff))
        pack_buff += b"\0" * (offset - len(pack_buff))

        self.hdr.io_offset_ = offset
        pack_buff += io.to_bytes()
        offset = ALIGN16(len(pack_buff))
        pack_buff += b"\0" * (offset - len(pack_buff))

        self.hdr.param_offset_ = offset
        pack_buff += param.to_bytes()
        offset = ALIGN16(len(pack_buff))
        pack_buff += b"\0" * (offset - len(pack_buff))

        self.hdr.shape_infer_offset_ = offset
        pack_buff += shape.to_bytes()
        offset = ALIGN16(len(pack_buff))
        pack_buff += b'\0'*(offset-len(pack_buff))

        self.hdr.debug_offset_ = offset
        pack_buff += debug.to_bytes()
        offset = ALIGN16(len(pack_buff))
        pack_buff += b"\0" * (offset - len(pack_buff))

        self.hdr.dma_offset_ = offset
        pack_buff += dma.to_bytes()
        offset = ALIGN16(len(pack_buff))
        pack_buff += b"\0" * (offset - len(pack_buff))

        crc = crc24_calc(pack_buff[head_size:])

        self.hdr.total_size_ = offset
        self.hdr.crc32_ = crc

        if platform.upper() == "VENUS":
            self.hdr.reserved = 0
        elif platform.upper() == "MARS":
            self.hdr.reserved = 1
        elif platform.upper() == "ARCS":
            self.hdr.reserved = 2
        elif platform.upper() == "VENUSA":
            self.hdr.reserved = 3

        hdr_buff = bytes(tffi.buffer(self.hdr))

        self.bytes = hdr_buff + pack_buff[len(hdr_buff) :]

        print(f"{Colors.CYAN} resource_total_size:{len(self.bytes)}{Colors.RESET}")

    def to_bytes(self):
        return self.bytes


__all__ = ["tModel"]
