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
 * @brief Vector multiplication function type definition
 */
typedef int32_t (*VEC_MUL_LUNA_API)(void *src1, void *src2, void *dst, int32_t size, int32_t shift);

/**
 * @brief Vector scaling function type definition
 */
typedef int32_t (*VEC_SCALE_LUNA_API)(void *src, int32_t scalar, void *dst, int32_t size, int32_t shift);

/**
 * @brief Vector operation function item structure
 */
struct luna_vec_mul_item {
    void *luna_api;
};

/**
 * @brief Vector operation function table
 */
struct luna_vec_mul_item luna_vec_api_list[][3] = {
    {{API_LIB(mul_q7_int8)}, {API_LIB(mul_q7_int16)}, {API_LIB(mul_q7_int32)}},
    {{API_LIB(mul_q15_int8)}, {API_LIB(mul_q15_int16)}, {API_LIB(mul_q15_int32)}},
    {{API_LIB(mul_q31_int8)}, {API_LIB(mul_q31_int16)}, {API_LIB(mul_q31_int32)}},
    {{API_LIB(scale_q7_int8)}, {API_LIB(scale_q7_int16)}, {API_LIB(scale_q7_int32)}},
    {{API_LIB(scale_q15_int8)}, {API_LIB(scale_q15_int16)}, {API_LIB(scale_q15_int32)}},
    {{API_LIB(scale_q31_int8)}, {API_LIB(scale_q31_int16)}, {API_LIB(scale_q31_int32)}},
};

/**
 * @brief Vector multiplication function
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param Y Output tensor
 * @param size Tensor size
 * @param shift Shift amount
 * @return int32_t Operation status
 */
static int32_t calc_vec_mul_luna(tTensor *lhs, tTensor *rhs, tTensor *Y,
                                 tTensor *Temp, int32_t size, int32_t shift) {
    int32_t in_idx = (lhs->dtype_ & 0xF) >> 1;
    int32_t ou_idx = (Y->dtype_ & 0xF) >> 1;
    VEC_MUL_LUNA_API luna_vec_api = (VEC_MUL_LUNA_API)luna_vec_api_list[in_idx][ou_idx].luna_api;
    int32_t lhs_size = size * lhs->byte_;
    int32_t rhs_size = size * rhs->byte_;
    int32_t output_size = size * Y->byte_;
    int32_t required = (lhs->mem_.type_ != 2 ? ALIGN4(lhs_size) : 0) +
                       (rhs->mem_.type_ != 2 ? ALIGN4(rhs_size) : 0) +
                       (Y->mem_.type_ != 2 ? output_size : 0);
    int8_t *p_tmp = Temp ? (int8_t *)Temp->dptr_ : NULL;
    void *src1 = (void *)lhs->dptr_;
    void *src2 = (void *)rhs->dptr_;
    void *dst = (void *)Y->dptr_;

    #if THINKER_RUNTIME_CHECK
    if (required &&
                          (Temp == NULL || Temp->dptr_ == 0 ||
                            getTensorDataSize(Temp) < (size_t)required)) {
        return (T_ERR_NO_WORKSPACE);
    }
    #endif
    if (lhs->mem_.type_ != 2) {
        src1 = p_tmp;
        memcpy(src1, (void *)lhs->dptr_, lhs_size);
        p_tmp += ALIGN4(lhs_size);
    }
    if (rhs->mem_.type_ != 2) {
        src2 = p_tmp;
        memcpy(src2, (void *)rhs->dptr_, rhs_size);
        p_tmp += ALIGN4(rhs_size);
    }
    if (Y->mem_.type_ != 2) {
        dst = p_tmp;
    }
    THINKER_RET_CHECK(luna_vec_api(src1, src2, dst, size, shift), "luna_vec_api");
    if (Y->mem_.type_ != 2) {
        memcpy((void *)Y->dptr_, dst, output_size);
    }
    return T_SUCCESS;
}

/**
 * @brief Vector scaling function
 * @param lhs Input tensor
 * @param scalar Scaling factor
 * @param Y Output tensor
 * @param size Tensor size
 * @param shift Shift amount
 * @return int32_t Operation status
 */
static int32_t calc_vec_scale_luna(tTensor *lhs, int32_t scalar, tTensor *Y,
                                   tTensor *Temp, int32_t size, int32_t shift) {
    int32_t in_idx = ((lhs->dtype_ & 0xF) >> 1) + 3;
    int32_t ou_idx = (Y->dtype_ & 0xF) >> 1;
    VEC_SCALE_LUNA_API luna_vec_api = (VEC_SCALE_LUNA_API)luna_vec_api_list[in_idx][ou_idx].luna_api;
    int32_t input_size = size * lhs->byte_;
    int32_t output_size = size * Y->byte_;
    int32_t required = (lhs->mem_.type_ != 2 ? ALIGN4(input_size) : 0) +
                       (Y->mem_.type_ != 2 ? output_size : 0);
    int8_t *p_tmp = Temp ? (int8_t *)Temp->dptr_ : NULL;
    void *src = (void *)lhs->dptr_;
    void *dst = (void *)Y->dptr_;

    #if THINKER_RUNTIME_CHECK
    if (required &&
                          (Temp == NULL || Temp->dptr_ == 0 ||
                            getTensorDataSize(Temp) < (size_t)required)) {
        return (T_ERR_NO_WORKSPACE);
    }
    #endif
    if (lhs->mem_.type_ != 2) {
        src = p_tmp;
        memcpy(src, (void *)lhs->dptr_, input_size);
        p_tmp += ALIGN4(input_size);
    }
    if (Y->mem_.type_ != 2) {
        dst = p_tmp;
    }
    THINKER_RET_CHECK(luna_vec_api(src, scalar, dst, size, shift), "luna_vec_api");
    if (Y->mem_.type_ != 2) {
        memcpy((void *)Y->dptr_, dst, output_size);
    }
    return T_SUCCESS;
}

/**
 * @brief Vector multiplication with broadcast support
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param Y Output tensor
 * @param Temp Temporary tensor (if needed)
 * @param shift Shift amount
 * @return int32_t Operation status
 */
static int32_t calc_vec_mul_luna_b2b2_broadcast_h1w1(tTensor *lhs, tTensor *rhs, tTensor *Y, tTensor *Temp, int32_t shift) {
    int32_t c = lhs->shape_.dims_[1];
    int32_t h = lhs->shape_.dims_[2];
    int32_t w = lhs->shape_.dims_[3];
    int32_t lhs_size = c * h * w;
    int32_t required = ALIGN4(h * w) + ALIGN4(lhs_size) +
                       (rhs->mem_.type_ != 2 ? ALIGN4(c) : 0) +
                       (lhs->mem_.type_ != 2 ? ALIGN4(lhs_size) : 0) +
                       (Y->mem_.type_ != 2 ? lhs_size : 0);
    int8_t *p_tmp1 = Temp ? (int8_t *)Temp->dptr_ : NULL;
    int8_t *p_tmp2;
    int8_t *p_tmp;
    int8_t *p_rhs = (int8_t *)rhs->dptr_;
    int8_t *p_lhs = (int8_t *)lhs->dptr_;
    int8_t *p_out = (int8_t *)Y->dptr_;

    #if THINKER_RUNTIME_CHECK
    if (Temp == NULL || Temp->dptr_ == 0 ||
                           getTensorDataSize(Temp) < (size_t)required) {
        return (T_ERR_NO_WORKSPACE);
    }
    #endif
    p_tmp2 = p_tmp1 + ALIGN4(h * w);
    p_tmp = p_tmp2 + ALIGN4(lhs_size);
    if (rhs->mem_.type_ != 2) {
        p_rhs = p_tmp;
        memcpy(p_rhs, (void *)rhs->dptr_, c);
        p_tmp += ALIGN4(c);
    }
    if (lhs->mem_.type_ != 2) {
        p_lhs = p_tmp;
        memcpy(p_lhs, (void *)lhs->dptr_, lhs_size);
        p_tmp += ALIGN4(lhs_size);
    }
    if (Y->mem_.type_ != 2) {
        p_out = p_tmp;
    }

    THINKER_RET_CHECK(API_LIB(memset)(p_tmp1, 1, h * w), "luna_memset");
    THINKER_RET_CHECK(API_LIB(mat_mul_q7_int8)(p_rhs, p_tmp1, p_tmp2, c, 1, h * w, 0), "luna_mat_mul_q7_int8");
    THINKER_RET_CHECK(API_LIB(mul_q7_int8)(p_lhs, p_tmp2, p_out, lhs_size, shift), "luna_mul_q7_int8");
    if (Y->mem_.type_ != 2) {
        memcpy((void *)Y->dptr_, p_out, lhs_size);
    }

    return T_SUCCESS;
}

/**
 * @brief Integer quantized multiplication operation
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param Y Output tensor
 * @param Temp Temporary tensor (if needed)
 * @param attrs Operation attributes
 * @return int32_t Operation status
 */
int32_t iqmul_luna(tTensor *lhs, tTensor *rhs, tTensor *Y, tTensor *Temp, iqBinaryAttrs *attrs) {
    #if THINKER_PARAM_CHECK
    if (lhs == NULL || rhs == NULL || Y == NULL || attrs == NULL ||
                        lhs->dptr_ == 0 || rhs->dptr_ == 0 || Y->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
    if ((lhs->dtype_ != Int8 && lhs->dtype_ != Int16 && lhs->dtype_ != Int32) ||
                        (rhs->dtype_ != Int8 && rhs->dtype_ != Int16 && rhs->dtype_ != Int32) ||
                        (Y->dtype_ != Int8 && Y->dtype_ != Int16 && Y->dtype_ != Int32)) {
        return (T_ERR_INVALID_DATATYPE);
    }
    #endif
    int32_t x1_q = (int32_t)lhs->scale_;
    int32_t x2_q = (int32_t)rhs->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    int32_t shift = x1_q + x2_q - y_q;
    size_t size = getTensorSize(lhs);

    #if THINKER_PARAM_CHECK
    if (shift < 0 || shift > 63) {
        return (T_ERR_INVALID_PARA);
    }
    if (lhs->dtype_ != rhs->dtype_ || lhs->dtype_ != Y->dtype_) {
        return (T_ERR_INVALID_DATATYPE);
    }
    if (!equalShape(&lhs->shape_, &Y->shape_)) {
        return (T_ERR_INVALID_DATA);
    }
    #endif

    if (lhs->shape_.ndim_ == 4 && rhs->shape_.ndim_ == 4 &&
        lhs->shape_.dims_[0] == 1 && rhs->shape_.dims_[0] == 1 &&
        lhs->shape_.dims_[1] == rhs->shape_.dims_[1] &&
        rhs->shape_.dims_[2] == 1 && rhs->shape_.dims_[3] == 1) {
        #if THINKER_PARAM_CHECK
        if (lhs->dtype_ != Int8) {
            return (T_ERR_INVALID_DATATYPE);
        }
        #endif
        THINKER_RET_CHECK(calc_vec_mul_luna_b2b2_broadcast_h1w1(lhs, rhs, Y, Temp, shift), "calc_vec_mul_luna_b2b2_broadcast_h1w1");
    } else if (0 == rhs->shape_.ndim_) {  // Scalar case
        int32_t scalar = 0;
        switch (rhs->dtype_) {
            case Int8:
                scalar = (int32_t)(*(int8_t *)rhs->dptr_);
                break;
            case Int16:
                scalar = (int32_t)(*(int16_t *)rhs->dptr_);
                break;
            case Int32:
                scalar = (int32_t)(*(int32_t *)rhs->dptr_);
                break;
            default:
                return T_ERR_INVALID_DATATYPE;
        }
        THINKER_RET_CHECK(calc_vec_scale_luna(lhs, scalar, Y, Temp, size, shift), "calc_vec_scale_luna");
    } else {  // Vector case
        #if THINKER_PARAM_CHECK
        if (!equalShape(&lhs->shape_, &rhs->shape_)) {
            return (T_ERR_INVALID_DATA);
        }
        #endif
        THINKER_RET_CHECK(calc_vec_mul_luna(lhs, rhs, Y, Temp, size, shift), "calc_vec_mul_luna");
    }

    return T_SUCCESS;
}

#endif
