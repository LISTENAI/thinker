#ifndef _GELU_LUNA_H_
#define _GELU_LUNA_H_

#include <math.h>
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

static int32_t gelu_scale_i32_to_q27(const int32_t *src, int32_t *dst, uint32_t size, int32_t shift) {
    int32_t scalar = 1;
    uint32_t rshift = 0;
    if (shift >= 0) {
#if THINKER_PARAM_CHECK
if (shift > 30) {
    return (T_ERR_INVALID_PARA);
}
#endif
        scalar = (int32_t)(1UL << (uint32_t)shift);
    } else {
#if THINKER_PARAM_CHECK
if (shift < -63) {
    return (T_ERR_INVALID_PARA);
}
#endif
        rshift = (uint32_t)(-shift);
    }
    return API_LIB(scale_i32i32o32)(src, scalar, dst, size, rshift);
}

static int32_t gelu_scale_i32_to_i8(const int32_t *src, int8_t *dst, uint32_t size, int32_t shift) {
    int32_t scalar = 1;
    uint32_t rshift = 0;
    if (shift >= 0) {
#if THINKER_PARAM_CHECK
if (shift > 63) {
    return (T_ERR_INVALID_PARA);
}
#endif
        rshift = (uint32_t)shift;
    } else {
#if THINKER_PARAM_CHECK
if (shift < -30) {
    return (T_ERR_INVALID_PARA);
}
#endif
        scalar = (int32_t)(1UL << (uint32_t)(-shift));
    }
    return API_LIB(scale_i32i32o8)(src, scalar, dst, size, rshift);
}

/**
 * @brief Quantized GELU activation function implementation
 * @param X Input tensor
 * @param Y Output tensor
 * @param Temp Temporary workspace tensor
 * @return int32_t Operation status
 */
int32_t gelu_luna(tTensor *X, tTensor *Y, tTensor *Temp) {
#if THINKER_PARAM_CHECK
    if (X == NULL || Y == NULL || Temp == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if (Y->dtype_ != Int8 || Temp->dtype_ != Int8 ||
                        Temp->byte_ != 1) {
        return (T_ERR_INVALID_DATATYPE);
    }
#endif

#if THINKER_RUNTIME_CHECK
    if (X->mem_.type_ != 2 || Y->mem_.type_ != 2 ||
                        Temp->mem_.type_ != 2) {
        return (T_ERR_INVALID_PLATFROM);
    }

    if (X->dptr_ == 0 || Y->dptr_ == 0 || Temp->dptr_ == 0 ||
                        ((uintptr_t)Temp->dptr_ & 3U) != 0 ||
                        Temp->dptr_ == X->dptr_ || Temp->dptr_ == Y->dptr_) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    size_t input_elements = getTensorSize(X);
    #if THINKER_PARAM_CHECK
    if (input_elements > UINT32_MAX) {
        return (T_ERR_INVALID_DATA);
    }
#endif
    uint32_t input_size = (uint32_t)input_elements;
    size_t workspace_size = getTensorDataSize(Temp);
    // Quantization parameters (GELU uses Q27 input)
    const int32_t Q_INPUT = 27;
    const int32_t Q_OUTPUT = 27;
    #if THINKER_PARAM_CHECK
    if (!isfinite(X->scale_) || !isfinite(Y->scale_) ||
                        floorf(X->scale_) != X->scale_ ||
                        floorf(Y->scale_) != Y->scale_ ||
                        X->scale_ < -3.0f || X->scale_ > 90.0f ||
                        Y->scale_ < -36.0f || Y->scale_ > 57.0f) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    int32_t x_q = (int32_t)X->scale_;
    int32_t y_q = (int32_t)Y->scale_;

    int8_t *dst = (int8_t *)Y->dptr_;
    int32_t shift_i = Q_INPUT - x_q;
    int32_t shift_o = Q_OUTPUT - y_q;

    if (X->dtype_ == Int8) {
        size_t required = ALIGN4((size_t)input_size * sizeof(int16_t)) +
                          (size_t)input_size * sizeof(int32_t);
        if (workspace_size < required) {
            return T_ERR_NO_WORKSPACE;
        }
        int8_t *src = (int8_t *)X->dptr_;
        int16_t *tmp = (int16_t *)Temp->dptr_;
        int32_t *tmp1 = (int32_t *)((int8_t *)tmp + ALIGN4(input_size * sizeof(int16_t)));

        THINKER_RET_CHECK(API_LIB(scale_i8i8o16)(src, 1, tmp, input_size, 0), "luna_scale_i8i8o16");
        THINKER_RET_CHECK(API_LIB(scale_i16i16o32)(tmp, 1, tmp1, input_size, 0), "luna_scale_i16i16o32");
        THINKER_RET_CHECK(gelu_scale_i32_to_q27(tmp1, tmp1, input_size, shift_i), "gelu_scale_i32_to_q27");
        THINKER_RET_CHECK(API_LIB(gelu_i32o32)(tmp1, tmp1, input_size), "luna_gelu_i32o32");
        THINKER_RET_CHECK(gelu_scale_i32_to_i8(tmp1, dst, input_size, shift_o), "gelu_scale_i32_to_i8");
    }
    else if (X->dtype_ == Int16) {
        if (workspace_size < (size_t)input_size * sizeof(int32_t)) {
            return T_ERR_NO_WORKSPACE;
        }
        int16_t *src = (int16_t *)X->dptr_;
        int32_t *tmp = (int32_t *)Temp->dptr_;

        THINKER_RET_CHECK(API_LIB(scale_i16i16o32)(src, 1, tmp, input_size, 0), "luna_scale_i16i16o32");
        THINKER_RET_CHECK(gelu_scale_i32_to_q27(tmp, tmp, input_size, shift_i), "gelu_scale_i32_to_q27");

        THINKER_RET_CHECK(API_LIB(gelu_i32o32)(tmp, tmp, input_size), "luna_gelu_i32o32");
        THINKER_RET_CHECK(gelu_scale_i32_to_i8(tmp, dst, input_size, shift_o), "gelu_scale_i32_to_i8");
    }
    else if (X->dtype_ == Int32) {
        if (workspace_size < (size_t)input_size * sizeof(int32_t)) {
            return T_ERR_NO_WORKSPACE;
        }
        int32_t *src = (int32_t *)X->dptr_;
        int32_t *tmp = (int32_t *)Temp->dptr_;

        if (shift_i != 0) {
            THINKER_RET_CHECK(gelu_scale_i32_to_q27(src, tmp, input_size, shift_i), "gelu_scale_i32_to_q27");
            src = tmp;
        }
        THINKER_RET_CHECK(API_LIB(gelu_i32o32)(src, tmp, input_size), "luna_gelu_i32o32");
        THINKER_RET_CHECK(gelu_scale_i32_to_i8(tmp, dst, input_size, shift_o), "gelu_scale_i32_to_i8");
    }
    else {
        return T_ERR_INVALID_DATATYPE;
    }
    return T_SUCCESS;
}

#endif  // _GELU_LUNA_H_
