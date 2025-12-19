import ctypes as ct
import numpy as np

DTypeUndefined = 0
Float16 = 0x6602
Float32 = 0x6604
Float64 = 0x6608
Int4    = 0x6900
Int8    = 0x6901
Int16   = 0x6902
Int32   = 0x6904
Int64   = 0x6908
Uint8   = 0x7501
Uint16  = 0x7502
Uint32  = 0x7504
Uint64  = 0x7508
Bool    = 0x6201

DTYPE_TO_NP = {
    Float16: np.float16,
    Float32: np.float32,
    Float64: np.float64,
    Int4:    np.int8,     # 无直接 int4 类型，用 int8 占位
    Int8:    np.int8,
    Int16:   np.int16,
    Int32:   np.int32,
    Int64:   np.int64,
    Uint8:   np.uint8,
    Uint16:  np.uint16,
    Uint32:  np.uint32,
    Uint64:  np.uint64,
    Bool:    np.bool_,
}

def dtype_bytes(dtype_val: int) -> int:
    return dtype_val & 0xF

class tShape(ct.Structure):
    _fields_ = [
        ("ndim_", ct.c_int32),
        ("dims_", ct.c_int32 * 8),
    ]


class tData(ct.Structure):
    _fields_ = [
        ("dptr_", ct.c_void_p),
        ("dev_type_", ct.c_uint16),
        ("dtype_", ct.c_uint16),
        ("zero_", ct.c_uint16),
        ("scale_", ct.c_float),
        ("shape_", tShape),
    ]


class tMemory(ct.Structure):
    _fields_ = [
        ("size_", ct.c_uint32),
        ("dev_type_", ct.c_uint8),
        ("mem_type_", ct.c_uint8),
        ("_pad_", ct.c_uint8 * 2),
        ("dptr_", ct.c_uint64),
    ]


tStatus = ct.c_int32
tModelHandle = ct.c_uint64
tExecHandle = ct.c_uint64


def bind_functions(lib):
    """
    为 libthinker.so 绑定所有函数的参数类型和返回值类型
    """

    lib.tGetVersion.argtypes = [ct.c_int8]
    lib.tGetVersion.restype = ct.c_char_p

    lib.tInitialize.argtypes = []
    lib.tInitialize.restype = tStatus

    lib.tUninitialize.argtypes = []
    lib.tUninitialize.restype = tStatus

    lib.tGetMemoryPlan.argtypes = [
        ct.POINTER(tMemory),
        ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_int8),
        ct.c_uint64,
    ]
    lib.tGetMemoryPlan.restype = tStatus


    lib.tModelInit.argtypes = [
        ct.POINTER(tModelHandle),
        ct.POINTER(ct.c_int8),
        ct.c_uint64,
        ct.POINTER(tMemory),
        ct.c_int32,
    ]
    lib.tModelInit.restype = tStatus

    lib.tModelFini.argtypes = [tModelHandle]
    lib.tModelFini.restype = tStatus

    lib.tGetInputCount.argtypes = [tModelHandle]
    lib.tGetInputCount.restype = ct.c_int32

    lib.tGetInputName.argtypes = [tModelHandle, ct.c_int32]
    lib.tGetInputName.restype = ct.c_char_p

    lib.tGetInputShape.argtypes = [tModelHandle, ct.c_int32]
    lib.tGetInputShape.restype = tShape

    lib.tGetInputDataType.argtypes = [tModelHandle, ct.c_int32]
    lib.tGetInputDataType.restype = ct.c_int32  # tDtype

    lib.tGetInputInfo.argtypes = [tExecHandle, ct.c_int32, ct.POINTER(tData)]
    lib.tGetInputInfo.restype = tStatus

    lib.tSetInput.argtypes = [tExecHandle, ct.c_int32, ct.POINTER(tData)]
    lib.tSetInput.restype = tStatus

    lib.tSetInputByName.argtypes = [tExecHandle, ct.c_char_p, ct.POINTER(tData)]
    lib.tSetInputByName.restype = tStatus

    lib.tGetOutputCount.argtypes = [tModelHandle]
    lib.tGetOutputCount.restype = ct.c_int32

    lib.tGetOutputName.argtypes = [tModelHandle, ct.c_int32]
    lib.tGetOutputName.restype = ct.c_char_p

    lib.tGetOutputShape.argtypes = [tModelHandle, ct.c_int32]
    lib.tGetOutputShape.restype = tShape

    lib.tGetOutputDataType.argtypes = [tModelHandle, ct.c_int32]
    lib.tGetOutputDataType.restype = ct.c_int32  # tDtype

    lib.tGetOutput.argtypes = [tExecHandle, ct.c_int32, ct.POINTER(tData)]
    lib.tGetOutput.restype = tStatus

    lib.tGetOutputByName.argtypes = [tExecHandle, ct.c_char_p, ct.POINTER(tData)]
    lib.tGetOutputByName.restype = tStatus

    lib.tCreateExecutor.argtypes = [
        tModelHandle,
        ct.POINTER(tExecHandle),
        ct.POINTER(tMemory),
        ct.c_int32,
    ]
    lib.tCreateExecutor.restype = tStatus

    lib.tReleaseExecutor.argtypes = [tExecHandle]
    lib.tReleaseExecutor.restype = tStatus

    lib.tForward.argtypes = [tExecHandle]
    lib.tForward.restype = tStatus

    lib.tExecutorStart.argtypes = [tExecHandle]
    lib.tExecutorStart.restype = tStatus

    lib.tExecutorStop.argtypes = [tExecHandle]
    lib.tExecutorStop.restype = tStatus

    lib.tUpdateShape.argtypes = [
        tExecHandle,
        ct.POINTER(ct.c_char_p),
        ct.POINTER(ct.c_uint32),
        ct.c_int32,
    ]
    lib.tUpdateShape.restype = tStatus

    print("    -> [Thinker] All function bindings registered successfully.")
    return lib
