import numpy as np
from ..enum_defines import Layout, ALIGN16


class Tensor:
    def __init__(
        self,
        shape,
        dtype,
        scale=-1.0,
        zero=0,
        layout=Layout.NCHW,
        data=None,
        mem_type=None,
    ):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.scale = scale
        self.zero = zero
        self.layout = layout
        self.data = data
        self.is_dynamic_data = False
        self.mem_type = mem_type

    def __str__(self):
        return "Tensor(shape={}, dtype='{}')".format(self.shape, self.dtype.str)

    def __repr__(self):
        return self.__str__()

    @property
    def size(self):
        if len(self.shape) == 0 and self.data is not None:
            return self.data.size
        size = 1
        for x in self.shape:
            size *= x
        return size

    @property
    def nbytes(self):
        tmp = ALIGN16((int)(self.size * self.dtype.itemsize))
        return tmp

    @property
    def get_cstep(self):
        align_size = self.shape[2] * self.shape[3] * self.elemsize
        return np.int((align_size + 15) // 16 * 16 / self.elemsize)

    def has_data(self):
        return self.data is not None

    def clone(self, **kwargs):
        """
        Clone Tensor attribute and construct a new Tensor
        Notice: do not clone data
        """
        t = self.copy()
        for k, v in kwargs.items():
            setattr(t, k, v)
        t.dtype = np.dtype(t.dtype)
        return t

    def copy(self):
        tensor = Tensor(self.shape, self.dtype)
        tensor.scale = self.scale
        tensor.zero = self.zero
        tensor.data = None
        tensor.layout = self.layout
        tensor.mem_type = self.mem_type
        tensor.is_dynamic_data = self.is_dynamic_data
        return tensor

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def from_numpy(x):
        assert isinstance(x, np.ndarray)
        tensor = Tensor(x.shape, x.dtype)
        tensor.data = x
        return tensor

    @staticmethod
    def from_shape(shape, dtype, mem_type):
        tensor = Tensor(shape, dtype, mem_type=mem_type)
        if isinstance(dtype, np.dtype):
            dtype = dtype.type
        return tensor


__all__ = ["Tensor"]
