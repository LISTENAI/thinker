#ifndef _LOGSOFTMAXINT_LUNA_H_
#define _LOGSOFTMAXINT_LUNA_H_

#include <math.h>
#include <stdio.h>
#include <string.h>
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/* Logarithm table for base 2 and natural logarithm */
static const int32_t log2_table[] = {
    /* Table data remains unchanged */
};

/**
 * @brief Calculate the number of leading zeros in a 32-bit integer
 * @param x Input integer
 * @return Number of leading zeros
 */
static int32_t nsa(int32_t x) {
    uint32_t ux = x < 0 ? -x : x;
    if (ux == 0x80000000) return 0;
    ux &= 0x7FFFFFFF;
    int32_t ix = 0;
    while (!(ux & 0x40000000) && ix < 31) {
        ux <<= 1;
        ix++;
    }
    return ix;
}

/**
 * @brief Determine the sign of a 64-bit integer
 * @param x Input integer
 * @return -1 for negative, 0 for zero, 1 for positive
 */
static int32_t sign_int64(int64_t x) {
    int32_t s = x < 0 ? -1 : 1;
    return x == 0 ? 0 : s;
}

/**
 * @brief Saturated multiplication for 32-bit fixed-point numbers
 * @param z First operand (Q1.63)
 * @param x Second operand (Q1.31)
 * @param y Third operand (Q1.31)
 * @return Result (Q1.63)
 */
static int64_t mula_32_f63(int64_t z, int32_t x, int32_t y) {
    int64_t s = (int64_t)x * y;
    int64_t s0[2] = {s, s < 0 ? -1 : 0};
    int64_t s1[2] = {z, z < 0 ? -1 : 0};
    
    s0[0] <<= 1;
    s0[1] = (s0[1] << 1) | ((s0[0] >> 63) & 1);
    
    int64_t s2[2] = {s0[0] + s1[0], s1[1] + (s0[0] < 0 && s1[0] < 0 ? 1 : 0)};
    int32_t overflow = 0;
    if (sign_int64(s0[0]) * sign_int64(s1[0]) > 0 && sign_int64(s0[0]) * sign_int64(s2[0]) < 0) {
        overflow = 1;
    }
    s2[1] += overflow;
    
    int64_t s3 = s2[0];
    if ((s3 > 0 && s2[1] > 0) || (s3 > 0 && s2[1] == 0 && s3 > 0x7FFFFFFFFFFFFFFF)) {
        s3 = 0x7FFFFFFFFFFFFFFF;
    }
    if ((s3 < 0 && s2[1] < -1) || (s3 < 0 && s2[1] == -1 && s3 < 0x8000000000000000)) {
        s3 = 0x8000000000000000;
    }
    return s3;
}

/**
 * @brief Saturate a 64-bit integer to 32-bit range
 * @param x Input integer
 * @return Saturated 32-bit integer
 */
static int32_t sat32(int64_t x) {
    if (x < 0x80000000) return x;
    if (x > 0x7FFFFFFF) return 0x7FFFFFFF;
    return (int32_t)x;
}

/**
 * @brief Vector logarithm function for 32-bit fixed-point numbers
 * @param Y Output array
 * @param X Input array
 * @param N Length of arrays
 */
static void vec_logn_32x32_sim(int32_t *Y, const int32_t *X, int N) {
    const int32_t min_int32 = 0x80000000;
    const int32_t hx = 1 << 30;  // Q1.30
    const int32_t sx = 1 << 22;  // 0.5 in Q23
    const int32_t mx = (1 << 23) - 1;
    const int32_t ln_2 = 0x58B90BFC;  // Q31
    
    for (int i = 0; i < N; i++) {
        int32_t x = X[i];
        if (x <= 0) {
            Y[i] = min_int32;
            continue;
        }
        
        int32_t x_nsa = nsa(x);
        x = (x * (1 << x_nsa)) - hx;
        int32_t dx = (x & mx) - sx;
        dx <<= 2;
        int32_t offset = (x >> 23) << 3;
        
        int32_t log2_x0 = log2_table[offset];
        int64_t yf = (int64_t)log2_x0 << 32;
        yf = mula_32_f63(yf, log2_table[offset + 1], dx);
        
        int32_t xf = (int32_t)(yf >> 38);
        int32_t nx = (16 - x_nsa) << 25;
        int32_t yx = xf + nx;
        
        int64_t yx_tmp = (int64_t)yx * ln_2;
        yx_tmp = round(yx_tmp * pow(2, -31));
        Y[i] = sat32(yx_tmp);
    }
}

/**
 * @brief LogSoftmax function for integer tensors
 * @param data Input tensor
 * @param out Output tensor
 * @param Workspace Workspace buffer
 * @param attrs Operation attributes
 * @return Execution status
 */
int32_t logsoftmaxint_luna(tTensor *data, tTensor *out, tTensor *Workspace, LogSoftmaxIntAttrs *attrs) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    int32_t SOFTMAX_Q_IN = 25;
    int32_t SOFTMAX_Q_OUT = 15;
    
    int32_t axis = attrs->axis < 0 ? data->shape_.ndim_ + attrs->axis : attrs->axis;
    int32_t leading = 1, stride = 1;
    for (int32_t i = 0; i < axis; i++) leading *= data->shape_.dims_[i];
    for (int32_t i = axis; i < data->shape_.ndim_; i++) stride *= data->shape_.dims_[i];
    int32_t data_size = leading * stride;
    
    if (!(data->dtype_ == Int8 || data->dtype_ == Int16 || data->dtype_ == Int32)) return T_ERR_INVALID_DATATYPE;
    if (!(out->dtype_ == Int8 || out->dtype_ == Int16 || out->dtype_ == Int32)) return T_ERR_INVALID_DATATYPE;
    if (out->mem_.type_ != 2) return T_ERR_INVALID_PLATFROM;
    
    int32_t x_scale = (int32_t)data->scale_;
    int32_t y_scale = (int32_t)out->scale_;
    
    if (data->dtype_ == Int8) {
        int16_t *p_tmp0 = (int16_t *)Workspace->dptr_;
        int32_t *p_tmp1 = (int32_t *)(p_tmp0 + data_size);
        int32_t *dst_tmp = p_tmp1 + 4 * data_size;
        
        ret = API_LIB(scale_i8i8o16)((int8_t *)data->dptr_, 1, p_tmp0, data_size, 0);
        ret |= API_LIB(scale_i16i16o32)(p_tmp0, 1, p_tmp1, data_size, 0);
        ret |= API_LIB(scale_i32i32o32)(p_tmp1, (1 << (SOFTMAX_Q_IN - x_scale)), p_tmp1, data_size, 0);
        
        for (int32_t l = 0; l < leading; l++) {
            int32_t offset = l * stride;
            ret |= API_LIB(logsoftmax_i32o32)(p_tmp1 + offset, dst_tmp + offset, stride);
        }
        
        if (out->dtype_ == Int8) {
            ret |= API_LIB(scale_i32i32o8)(dst_tmp, 1, (int8_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale));
        } else if (out->dtype_ == Int16) {
            ret |= API_LIB(scale_i32i32o16)(dst_tmp, 1, (int16_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale));
        } else {
            ret |= API_LIB(scale_i32i32o32)(dst_tmp, 1, (int32_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale));
        }
    } else if (data->dtype_ == Int16) {
        int32_t *p_tmp = (int32_t *)Workspace->dptr_;
        int32_t *dst_tmp = p_tmp + 4 * data_size;
        
        ret = API_LIB(scale_i16i16o32)((int16_t *)data->dptr_, 1, p_tmp, data_size, 0);
        ret |= API_LIB(scale_i32i32o32)(p_tmp, (1 << (SOFTMAX_Q_IN - x_scale)), p_tmp, data_size, 0);
        
        for (int32_t l = 0; l < leading; l++) {
            int32_t offset = l * stride;
            ret |= API_LIB(logsoftmax_i32o32)(p_tmp + offset, dst_tmp + offset, stride);
        }
        
        if (out->dtype_ == Int8) {
            ret |= API_LIB(scale_i32i32o8)(dst_tmp, 1, (int8_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale));
        } else if (out->dtype_ == Int16) {
            ret |= API_LIB(scale_i32i32o16)(dst_tmp, 1, (int16_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale));
        } else {
            ret |= API_LIB(scale_i32i32o32)(dst_tmp, 1, (int32_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale));
        }
    } else if (data->dtype_ == Int32) {
        int32_t *p_tmp = (int32_t *)Workspace->dptr_;
        int32_t *dst_tmp = p_tmp + 4 * stride;
        
        ret = API_LIB(scale_i32i32o32)((int32_t *)data->dptr_, (1 << (SOFTMAX_Q_IN - x_scale)), p_tmp, data_size, 0);
        
        for (int32_t l = 0; l < leading; l++) {
            int32_t offset = l * stride;
            ret |= API_LIB(logsoftmax_i32o32)(p_tmp + offset, dst_tmp + offset, stride);
        }
        
        if (out->dtype_ == Int8) {
            ret |= API_LIB(scale_i32i32o8)(dst_tmp, 1, (int8_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale));
        } else if (out->dtype_ == Int16) {
            ret |= API_LIB(scale_i32i32o16)(dst_tmp, 1, (int16_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale));
        } else {
            ret |= API_LIB(scale_i32i32o32)(dst_tmp, 1, (int32_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale));
        }
    }
    
    return ret;
}

#endif