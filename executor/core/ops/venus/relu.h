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
                case Int8:  return API_LIB(relu_q7_int8)((const q7_t *)src, (q7_t *)dst, size, shift);
                case Int16: return API_LIB(relu_q7_int16)((const q7_t *)src, (q15_t *)dst, size, shift);
                case Int32: return API_LIB(relu_q7_int32)((const q7_t *)src, (q31_t *)dst, size, shift);
            }
        }
        case Int16: {
            switch (Y_dtype) {
                case Int8:  return API_LIB(relu_q15_int8)((const q15_t *)src, (q7_t *)dst, size, shift);
                case Int16: return API_LIB(relu_q15_int16)((const q15_t *)src, (q15_t *)dst, size, shift);
                case Int32: return API_LIB(relu_q15_int32)((const q15_t *)src, (q31_t *)dst, size, shift);
            }
        }
        case Int32: {
            switch (Y_dtype) {
                case Int8:  return API_LIB(relu_q31_int8)((const q31_t *)src, (q7_t *)dst, size, shift);
                case Int16: return API_LIB(relu_q31_int16)((const q31_t *)src, (q15_t *)dst, size, shift);
                case Int32: return API_LIB(relu_q31_int32)((const q31_t *)src, (q31_t *)dst, size, shift);
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
    #if THINKER_PARAM_CHECK
    if (X == NULL || Y == NULL || X->dptr_ == 0 || Y->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
    #endif
    int32_t shift = Y->scale_ - X->scale_;
    void *src = (void *)X->dptr_;
    void *dst = (void *)Y->dptr_;
    void *tmp_buf = NULL;
    uint32_t tmp_size = 0;

    if (Workspace != NULL) {
        tmp_buf = (void *)Workspace->dptr_;
        tmp_size = getTensorSize(Workspace);
    }

    uint32_t size = getTensorSize(X);
    #if THINKER_PARAM_CHECK
    if ((X->dtype_ != Int8 && X->dtype_ != Int16 && X->dtype_ != Int32) ||
                        (Y->dtype_ != Int8 && Y->dtype_ != Int16 && Y->dtype_ != Int32)) {
        return (T_ERR_INVALID_DATATYPE);
    }
    if (shift < 0 || shift > 63) {
        return (T_ERR_INVALID_PARA);
    }
    #endif
    // If input is in PSRAM, process in chunks
    if ((X->mem_.type_ != 2) || (Y->mem_.type_ != 2)) {
#if THINKER_PARAM_CHECK
if (X->dtype_ != Int8 || Y->dtype_ != Int8) {
    return (T_ERR_INVALID_DATATYPE);
}
#endif
        #if THINKER_RUNTIME_CHECK
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
            uint32_t offset = (uint32_t)i * split_size;
            int8_t *p_in = (int8_t *)src + offset;
            int8_t *p_out = (int8_t *)dst + offset;

            if (i == split_num - 1) {
                split_size = final_split_size;
            }
            if (X->mem_.type_ != 2) {
                memcpy(tmp_buf, p_in, split_size);
                p_in = tmp_buf;
            }
            if (Y->mem_.type_ != 2) {
                p_out = tmp_buf;
            }
            THINKER_RET_CHECK(calc_relu_luna(p_in, p_out, Int8, Int8, split_size, shift), "calc_relu_luna");
            if (Y->mem_.type_ != 2) {
                memcpy((int8_t *)dst + offset, p_out, split_size);
            }
        }
    }
    else {
        return calc_relu_luna(src, dst, X->dtype_, Y->dtype_, size, shift);
    }

    return T_SUCCESS;
}

#endif
