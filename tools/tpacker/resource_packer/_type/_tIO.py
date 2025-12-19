import numpy as np
from ctypes import Structure, sizeof
from ctypes import c_uint16, c_uint32

from ...enum_defines import ALIGN16


class tIO(object):
    def __init__(self, tensor_id, name):
        self.tensor_id = tensor_id
        self.name = name


class tState(object):
    def __init__(self, in_tensor_id, out_tensor_id, name):
        self.in_tensor_id = in_tensor_id
        self.out_tensor_id = out_tensor_id
        self.name = name


class tIOInfo(Structure):
    _fields_ = [
        ("num_input", c_uint16),
        ("num_output", c_uint16),
        ("num_state", c_uint16),
        ("name_length", c_uint16),
        ("tensor_offset", c_uint32),
        ("name_offset", c_uint32),
    ]

    def __init__(self, inputs, outputs, states=[]):
        self.num_input = c_uint16(len(inputs))
        self.num_output = c_uint16(len(outputs))
        self.num_state = c_uint16(len(states))

        self.tensor_ids = (
            [x.tensor_id for x in inputs]
            + [x.tensor_id for x in outputs]
            + [x.in_tensor_id for x in states]
            + [x.out_tensor_id for x in states]
        )

        self.name_list = (
            [x.name for x in inputs]
            + [x.name for x in outputs]
            + [x.name for x in states]
        )

        max_len = max([len(x) for x in self.name_list])
        max_len = ALIGN16(max_len + 1)
        self.name_length = c_uint16(max_len)

        self.tensor_offset = sizeof(self)
        tensor_bytes = np.array(self.tensor_ids, dtype=np.int32).tobytes()
        self.name_offset = self.tensor_offset + len(tensor_bytes)

        self.bytes = bytes(self)
        self.bytes += tensor_bytes
        for name in self.name_list:
            self.bytes += np.array([ord(x) for x in name], dtype=np.uint8).tobytes()
            self.bytes += b"\0" * (max_len - len(name))

    def to_bytes(self):
        return self.bytes


__all__ = ["tIO", "tIOInfo"]
