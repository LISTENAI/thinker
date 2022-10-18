from ._ctype import tffi
from ...enum_defines import ALIGN16


class tTenDimPair(object):
    def __init__(self, tensor_id, dim_id):
        self.obj = tffi.new("tTenDimPair *")
        self.obj.tensor_id_ = tensor_id
        self.obj.dim_id_ = dim_id

    def to_bytes(self):
        return bytes(tffi.buffer(self.obj))


class tDyAxisInfo(object):
    def __init__(self, tensor_id, dim_id, scalar_id):
        self.obj = tffi.new("tDyAxisInfo *")
        self.obj.tensor_id_ = tensor_id
        self.obj.dy_dim_id_ = dim_id
        self.obj.scalar_input_id_ = scalar_id

    def to_bytes(self):
        return bytes(tffi.buffer(self.obj))


class tShapeInferHdr(object):
    def __init__(self, dy_axis_list, id_pair_list, scalar_graph):
        assert len(id_pair_list) == len(scalar_graph.outputs)
        self.hdr = tffi.new("tShapeInferHdr *")
        pack_buffer = bytes(tffi.buffer(self.hdr))
        offset = ALIGN16(len(pack_buffer))
        pack_buffer += b"\0" * (offset - len(pack_buffer))
        # serialize dynamic axis
        self.hdr.dy_axis_offset_ = offset
        for dy_axis in dy_axis_list:
            pack_buffer += dy_axis.to_bytes()
        offset = ALIGN16(len(pack_buffer))
        pack_buffer += b"\0" * (offset - len(pack_buffer))
        self.hdr.num_dy_axis_ = len(dy_axis_list)

        # serialize scalar graph
        self.hdr.graph_offset_ = offset
        graph_buffer = scalar_graph.serialize()
        pack_buffer += graph_buffer
        offset = ALIGN16(len(pack_buffer))
        pack_buffer += b"\0" * (offset - len(pack_buffer))
        self.hdr.graph_size_ = len(graph_buffer)

        # serialize id pairs
        self.hdr.id_pair_offset_ = offset
        for id_pair in id_pair_list:
            pack_buffer += id_pair.to_bytes()
        offset = ALIGN16(len(pack_buffer))
        pack_buffer += b"\0" * (offset - len(pack_buffer))
        self.hdr.num_id_pair_ = len(id_pair_list)

        hdr_buff = bytes(tffi.buffer(self.hdr))
        self.bytes = hdr_buff + pack_buffer[len(hdr_buff) :]

    def to_bytes(self):
        return self.bytes


__all__ = ["tTenDimPair", "tDyAxisInfo", "tShapeInferHdr"]
