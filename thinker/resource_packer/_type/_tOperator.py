import numpy as np
from ._ctype import tffi
from ...enum_defines import ALIGN16


class tOperator(object):
    def __init__(
        self,
        op_attrs,
        op_type,
        dev_type,
        num_input,
        num_output,
        tensor_ids=[],
        scalar_ids=[],
    ):

        self.op_type = op_type
        self.tensor_ids = tensor_ids
        self.scalar_ids = scalar_ids
        self.attrs_buff = bytes(op_attrs)

        self.obj = tffi.new("tOperator *")
        self.obj.op_id_ = 0

        self.obj.num_input_ = num_input
        self.obj.num_output_ = num_output
        self.obj.num_temp_ = len(tensor_ids) - num_input - num_output
        # self.obj.num_scalar_ = len(scalar_ids)
        self.obj.num_scalar_ = dev_type.value

    def pack(self):
        pack_buff = bytes(tffi.buffer(self.obj))
        offset = len(pack_buff)

        self.obj.attr_offset_ = offset
        pack_buff += self.attrs_buff
        offset = ALIGN16(len(pack_buff))
        pack_buff += b"\0" * (offset - len(pack_buff))

        self.obj.tensor_offset_ = offset
        pack_buff += np.array(self.tensor_ids, dtype=np.int32).tobytes()
        offset = ALIGN16(len(pack_buff))
        pack_buff += b"\0" * (offset - len(pack_buff))

        self.obj.scalar_offset_ = offset
        self.obj.total_size_ = offset

        hdr_buff = bytes(tffi.buffer(self.obj))
        self.bytes = hdr_buff + pack_buff[len(hdr_buff) :]

    def to_bytes(self):
        self.pack()
        return self.bytes


class tOperatorList(object):
    def __init__(self, op_list):
        self._list = op_list
        self.obj = tffi.new("tOperatorList *")

        op_types = {}
        op_type_list = []
        op_buff = b""
        for x in op_list:
            if x.op_type not in op_types:
                op_types[x.op_type] = len(op_type_list)
                op_type_list.append(x.op_type)
            x.obj.op_id_ = op_types[x.op_type]  # update op_id
            op_buff += x.to_bytes()
        if len(op_type_list) != 0:
            max_len = max([len(x) for x in op_type_list])
            max_len = ALIGN16(max_len + 1)
        else:
            max_len = 0

        self.obj.op_count_ = len(op_list)
        self.obj.type_count_ = len(op_types)

        self.obj.type_length_ = max_len
        self.obj.header_size_ = tffi.sizeof("tOperatorList")
        self.obj.type_offset_ = ALIGN16(self.obj.header_size_)
        self.obj.op_offset_ = self.obj.type_offset_ + max_len * len(op_types)
        self.obj.op_size_ = len(op_buff)

        self.bytes = bytes(tffi.buffer(self.obj))
        for name in op_type_list:
            self.bytes += np.array([ord(x) for x in name], dtype=np.uint8).tobytes()
            self.bytes += b"\0" * (max_len - len(name))

        self.bytes += op_buff

    def to_bytes(self):
        return self.bytes


_all__ = ["tOperator", "tOperatorList"]
