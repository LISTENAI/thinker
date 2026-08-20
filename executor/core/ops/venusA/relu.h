#ifndef __RELU_H__
#define __RELU_H__

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

#include "thinker_status.h"

/**
 * @brief Calculate ReLU activation for different data types
 * @param X_dtype Input data type
 * @param Y_dtype Output data type
 * @param src Input data pointer
 * @param dst Output data pointer
 * @param size Size of data
 * @param shift Shift value
 * @return int32_t Operation status
 */
static int32_t calc_relu_luna(const void *src, void *dst, int32_t X_dtype, int32_t Y_dtype, uint32_t size, int32_t shift) {
    switch (X_dtype) {
        case Int8: {
            switch (Y_dtype) {
                case Int8:  return API_LIB(relu_i8o8)((const int8_t *)src, (int8_t *)dst, size, shift);
                case Int16:  return API_LIB(relu_i8o16)((const int8_t *)src, (int16_t *)dst, size, shift);
                case Int32: return API_LIB(relu_i8o32)((const int8_t *)src, (int32_t *)dst, size, shift);
            }
        }
        case Int16: {
            switch (Y_dtype) {
                case Int8:  return API_LIB(relu_i16o8)((const int16_t *)src, (int8_t *)dst, size, shift);
                case Int16:  return API_LIB(relu_i16o16)((const int16_t *)src, (int16_t *)dst, size, shift);
                case Int32: return API_LIB(relu_i16o32)((const int16_t *)src, (int32_t *)dst, size, shift);
            }
        }
        case Int32: {
            switch (Y_dtype) {
                case Int8:  return API_LIB(relu_i32o8)((const int32_t *)src, (int8_t *)dst, size, shift);
                case Int16:  return API_LIB(relu_i32o16)((const int32_t *)src, (int16_t *)dst, size, shift);
                case Int32: return API_LIB(relu_i32o32)((const int32_t *)src, (int32_t *)dst, size, shift);
            }
        }
    }
    return T_ERR_INVALID_DATATYPE;
}

/**
 * @brief Main ReLU function
 * @param X Input tensor
 * @param Y Output tensor
 * @param Workspace Temporary workspace tensor (optional)
 * @return tStatus Operation status
 */
tStatus relu_luna(tTensor *X, tTensor *Y, tTensor *Workspace) {
    int32_t shift = Y->scale_ - X->scale_;
    void *src = (void *)X->dptr_;
    void *dst = (void *)Y->dptr_;
    void *tmp_buf = NULL;
    uint32_t tmp_size = 0;

    if (Workspace != NULL) {
        tmp_buf = (void *)Workspace->dptr_;
        tmp_size = getTensorSize(Workspace);
    }

#if THINKER_PARAM_CHECK
if (shift < 0 || shift > 63) {
    return (T_ERR_INVALID_PARA);
}
#endif

    uint32_t size = getTensorSize(X);
    // If input is in PSRAM, process in chunks
    if ((X->mem_.type_ != 2) || (Y->mem_.type_ != 2)) {
#if THINKER_PARAM_CHECK
if (X->dtype_ != Int8 || Y->dtype_ != Int8) {
    return (T_ERR_INVALID_DATATYPE);
}

        if (tmp_buf == NULL || tmp_size == 0) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
        int32_t split_num = 1;
        int32_t split_size = size;
        while (split_size > tmp_size) {
            split_num++;
            split_size = (size + split_num - 1) / split_num;
        }

        int32_t final_split_size = size - split_size * (split_num - 1);
        for (int i = 0; i < split_num; i++) {
            int32_t offset = i * split_size;
            int8_t *p_in = (int8_t *)src + offset;
            int8_t *p_out = (int8_t *)dst + offset;

            if (i == split_num - 1) {
                split_size = final_split_size;
            }
            if (X->mem_.type_ != 2) {
                 THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(tmp_buf, p_in, split_size), "luna_memcpy_i8o8");
                p_in = tmp_buf;
            }
            if (Y->mem_.type_ != 2) {
                p_out = tmp_buf;
            }
            THINKER_RET_CHECK(calc_relu_luna(p_in, p_out, Int8, Int8, split_size, shift), "calc_relu_luna");
            if (Y->mem_.type_ != 2) {
                opi_psram_cpy_out((int8_t *)dst + offset, p_out, split_size);
            }
        }
    }
    else {
        return calc_relu_luna(src, dst, X->dtype_, Y->dtype_, size, shift);
    }

    return T_SUCCESS;
}

#endif
