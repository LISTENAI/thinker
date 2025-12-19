#ifndef _MUL_LUNA_H_
#define _MUL_LUNA_H_

#include <math.h>
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

/**
 * @brief Function pointer type for vector multiplication operations
 */
typedef int32_t (*VEC_MUL_LUNA_API)(void *src1, void *src2, void *dst, int32_t size, int32_t shift);

/**
 * @brief Function pointer type for vector scaling operations
 */
typedef int32_t (*VEC_SCALE_LUNA_API)(void *src, int32_t scalar, void *dst, int32_t size, int32_t shift);

/**
 * @brief Mapping of vector multiplication and scaling functions for different data types
 */
struct luna_vec_mul_item {
    void *luna_api;
};

static struct luna_vec_mul_item luna_vec_api_list[][3] = {
    {{API_LIB(mul_i8i8o8)}, {API_LIB(mul_i8i8o16)}, {API_LIB(mul_i8i8o32)}},
    {{API_LIB(mul_i16i16o8)}, {API_LIB(mul_i16i16o16)}, {API_LIB(mul_i16i16o32)}},
    {{API_LIB(mul_i32i32o8)}, {API_LIB(mul_i32i32o16)}, {API_LIB(mul_i32i32o32)}},
    {{API_LIB(scale_i8i8o8)}, {API_LIB(scale_i8i8o16)}, {API_LIB(scale_i8i8o32)}},
    {{API_LIB(scale_i16i16o8)}, {API_LIB(scale_i16i16o16)}, {API_LIB(scale_i16i16o32)}},
    {{API_LIB(scale_i32i32o8)}, {API_LIB(scale_i32i32o16)}, {API_LIB(scale_i32i32o32)}}
};

/**
 * @brief Performs vector multiplication with quantization adjustment
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param Y Output tensor
 * @param size Number of elements to process
 * @param shift Quantization shift
 * @return int32_t Operation status
 */
int32_t calc_vec_mul_luna(tTensor *lhs, tTensor *rhs, tTensor *Y, int32_t size, int32_t shift) {
    int32_t ret = T_ERR_FAIL;
    int32_t in_idx = (lhs->dtype_ & 0xF) >> 1;
    int32_t ou_idx = (Y->dtype_ & 0xF) >> 1;
    VEC_MUL_LUNA_API luna_vec_api = (VEC_MUL_LUNA_API)luna_vec_api_list[in_idx][ou_idx].luna_api;
    void *src1  = (void *)lhs->dptr_;
    void *src2  = (void *)rhs->dptr_;
    void *dst   = (void *)Y->dptr_;
    ret = luna_vec_api(src1, src2, dst, size, shift);
    return ret;
}

/**
 * @brief Performs vector scaling with quantization adjustment
 * @param lhs Input tensor
 * @param scalar Scaling factor
 * @param Y Output tensor
 * @param size Number of elements to process
 * @param shift Quantization shift
 * @return int32_t Operation status
 */
int32_t calc_vec_scale_luna(tTensor *lhs, int32_t scalar, tTensor *Y, int32_t size, int32_t shift) {
    int32_t ret = T_ERR_FAIL;
    int32_t in_idx = ((lhs->dtype_ & 0xF) >> 1) + 3;
    int32_t ou_idx = (Y->dtype_ & 0xF) >> 1;
    VEC_SCALE_LUNA_API luna_vec_api = (VEC_SCALE_LUNA_API)luna_vec_api_list[in_idx][ou_idx].luna_api;
    void *src = (void *)lhs->dptr_;
    void *dst = (void *)Y->dptr_;
    ret = luna_vec_api(src, scalar, dst, size, shift);
    return ret;
}

/**
 * @brief Performs vector multiplication with broadcast for specific tensor shapes
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param Y Output tensor
 * @param Temp Temporary workspace tensor
 * @param shift Quantization shift
 * @return int32_t Operation status
 */
int32_t calc_vec_mul_luna_b2b2_broadcast_h1w1(tTensor *lhs, tTensor *rhs, tTensor *Y, tTensor *Temp, int32_t shift) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    int32_t c = lhs->shape_.dims_[1];
    int32_t h = lhs->shape_.dims_[2];
    int32_t w = lhs->shape_.dims_[3];
    int8_t *p_tmp1 = (int8_t *)Temp->dptr_;
    int8_t *p_tmp2 = p_tmp1 + c;

    ret = API_LIB(memset_i8o8)(p_tmp1, 1, h * w);
    ret |= API_LIB(mat_mul_i8i8o8)((int8_t *)rhs->dptr_, p_tmp1, p_tmp2, c, 1, h * w, 0);
    ret |= API_LIB(mul_i8i8o8)((int8_t *)lhs->dptr_, p_tmp2, (int8_t *)Y->dptr_, c * h * w, shift);

    return ret;
}

/**
 * @brief Quantized multiplication operation implementation
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param Y Output tensor
 * @param Temp Temporary workspace tensor
 * @param attrs Operation attributes
 * @return int32_t Operation status
 */
int32_t iqmul_luna(tTensor *lhs, tTensor *rhs, tTensor *Y, tTensor *Temp, iqBinaryAttrs *attrs) {
    int32_t ret = T_ERR_FAIL;

    if (lhs->dtype_ == Int8 && rhs->dtype_ == Int32) {
        return T_ERR_INVALID_DATATYPE;
    }

    int32_t x1_q = (int32_t)lhs->scale_;
    int32_t x2_q = (int32_t)rhs->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    int32_t shift = x1_q + x2_q - y_q;
    size_t size = getTensorSize(lhs);

    if (shift < 0) {
        return ret;
    }

    if (lhs->shape_.ndim_ == 4 && rhs->shape_.ndim_ == 4 &&
        lhs->shape_.dims_[1] == rhs->shape_.dims_[1] &&
        rhs->shape_.dims_[2] == 1 && rhs->shape_.dims_[3] == 1) {
        ret = calc_vec_mul_luna_b2b2_broadcast_h1w1(lhs, rhs, Y, Temp, shift);
    } else if (rhs->shape_.ndim_ == 0) {
        int32_t scalar = *(int32_t *)rhs->dptr_;
        if (rhs->dtype_ == Int8) {
            scalar = *(int8_t *)rhs->dptr_;
        } else if (rhs->dtype_ == Int16) {
            scalar = *(int16_t *)rhs->dptr_;
        } else if (rhs->dtype_ == Float32) {
            scalar = (int32_t)*(float *)rhs->dptr_;
        } else {
            return T_ERR_INVALID_DATATYPE;
        }
        ret = calc_vec_scale_luna(lhs, scalar, Y, size, shift);
    } else {
        ret = calc_vec_mul_luna(lhs, rhs, Y, size, shift);
    }

    return ret;
}

#endif  // _MUL_LUNA_H_