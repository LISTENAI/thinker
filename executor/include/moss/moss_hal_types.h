#ifndef MOSS_HAL_TYPES_H
#define MOSS_HAL_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MOSS_OK                  = 0,
    MOSS_ERR_INVALID_ARG     = -1,
    MOSS_ERR_NULL_PTR        = -2,
    MOSS_ERR_OUT_OF_MEMORY   = -3,
    MOSS_ERR_DEVICE_ERROR    = -4,
    MOSS_ERR_NOT_SUPPORTED   = -5,
    MOSS_ERR_TIMEOUT         = -6,
    MOSS_ERR_BUSY            = -7,
    MOSS_ERR_INVALID_STATE   = -8,
    MOSS_ERR_ABI_MISMATCH    = -9,
} MossStatus;

typedef enum {
    MOSS_MEM_FLASH     = 0,
    MOSS_MEM_PSRAM     = 1,
    MOSS_MEM_SHARED    = 2,
    MOSS_MEM_DEVICE    = 3,
    MOSS_MEM_HOST      = 4,
} MossMemSpace;

typedef enum {
    MOSS_DTYPE_I4   = 0,
    MOSS_DTYPE_I8   = 1,
    MOSS_DTYPE_I16  = 2,
    MOSS_DTYPE_I32  = 3,
    MOSS_DTYPE_I64  = 4,
    MOSS_DTYPE_F16  = 5,
    MOSS_DTYPE_F32  = 6,
    MOSS_DTYPE_BF16 = 7,
} MossDtype;

typedef void (*MossKernelFunc)(int32_t argc, void** argv);

#ifdef __cplusplus
}
#endif

#endif /* MOSS_HAL_TYPES_H */
