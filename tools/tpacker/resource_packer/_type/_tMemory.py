from typing import List

from ._ctype import tffi
from ...enum_defines import MemType, ALIGN16, Colors


class tMemory(object):  # little endian
    def __init__(self, mem_type, size):
        self.obj = tffi.new("tMemory *")
        self.obj.dev_type_ = mem_type
        self.obj.mem_type_ = 0

        size = ALIGN16(size)
        self.obj.size_ = int(size)
        self.obj.dptr_ = 0

    def to_bytes(self):
        return bytes(tffi.buffer(self.obj))


class tMemoryList(object):
    def __init__(
        self, shared_memory_list: List[tMemory], runtime_memory_list: List[tMemory], ramsize: int
    ):
        memory_list = []
        for i in range(len(runtime_memory_list)):
            memory_list += runtime_memory_list[i]
        if shared_memory_list == None:
            self._list = memory_list
        else:
            self._list = shared_memory_list + memory_list

        _all_dev = []
        for i in range(len(self._list)):
            if self._list[i].obj.dev_type_ not in _all_dev:
                _all_dev.append(self._list[i].obj.dev_type_)

        for i in _all_dev:
            _total_size = 0
            for j in range(len(self._list)):
                if i == self._list[j].obj.dev_type_:
                    _total_size += self._list[j].obj.size_
            print(f"{Colors.CYAN}  {MemType(i)} need capacity: {_total_size} Bytes{Colors.RESET}")
            if i == 2:
                assert _total_size <= ramsize, f"{Colors.RED}  SHARE-MEM to be allocated was {_total_size}, exceed {ramsize} Byte{Colors.RESET}"
            elif i == 1:
                assert _total_size <= 8 * 1024 * 1024, f"{Colors.RED}  SPSRAM to be allocated was {_total_size}, exceed 8MB{Colors.RESET}"

        self.obj = tffi.new("tMemoryList *")

        self.obj.shared_count_ = len(shared_memory_list)
        self.obj.total_count_ = len(self._list)
        self.obj.elem_size_ = tffi.sizeof("tMemory")
        self.obj.header_size_ = tffi.sizeof("tMemoryList")
        self.obj.offset_ = ALIGN16(self.obj.header_size_)

        self.bytes = bytes(tffi.buffer(self.obj))
        for x in self._list:
            self.bytes += x.to_bytes()

    def to_bytes(self):
        return self.bytes


__all__ = ["tMemory", "tMemoryList"]
