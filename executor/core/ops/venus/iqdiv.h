#ifndef _DIV_LUNA_H_
#define _DIV_LUNA_H_

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
 * @brief Vector scaling function type definition
 */
typedef int32_t (*luna_vec_scale_api)(void *src, int32_t scalar, void *dst, int32_t size, int32_t shift);

/**
 * @brief Vector scaling function item type definition
 */
typedef void *luna_vec_scale_api_item;

/**
 * @brief Vector scaling function table
 */
static luna_vec_scale_api_item luna_vec_scale_api_items[][3] = {
    {
        API_LIB(scale_q7_int8),
        API_LIB(scale_q7_int16),
        API_LIB(scale_q7_int32),
    },
    {
        API_LIB(scale_q15_int8),
        API_LIB(scale_q15_int16),
        API_LIB(scale_q15_int32),
    },
    {
        API_LIB(scale_q31_int8),
        API_LIB(scale_q31_int16),
        API_LIB(scale_q31_int32),
    },
};

/**
 * @brief Vector division function
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param Y Output tensor
 * @param size Tensor size
 * @return int32_t Operation status
 */
static int32_t calc_vec_div_luna(tTensor *lhs, tTensor *rhs, tTensor *Y, int32_t size) {
    int32_t x1_q = (int32_t)lhs->scale_;
    int32_t x2_q = (int32_t)rhs->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    void *src1 = (void *)lhs->dptr_;
    void *src2 = (void *)rhs->dptr_;
    void *dst = (void *)Y->dptr_;

    if (lhs->dtype_ == Int32) {
        THINKER_RET_CHECK(API_LIB(div_q31_int32)((const q31_t *)src1, x1_q, (const q31_t *)src2, x2_q, (q31_t *)dst, y_q, size), "luna_div_q31_int32");
    }
    else {
        THINKER_LOG_FATAL("data type not support!");
        #if THINKER_PARAM_CHECK
        if (1) {
            return (T_ERR_INVALID_DATATYPE);
        }
        #endif
    }

    return T_SUCCESS;
}

static int32_t iqdiv_has_zero_i32(const int32_t *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (data[i] == 0) return 1;
    }
    return 0;
}

/**
 * @brief Fast base-2 logarithm approximation
 * @param x Input integer
 * @return int32_t Approximated log2(x)
 */
static int32_t fastlog2(int32_t x) {
    float fx = (float)x;
    uint32_t ix;
    memcpy(&ix, &fx, sizeof(ix));
    uint32_t exp = (ix >> 23) & 0xFF;
    return exp - 127;
}

/**
 * @brief Vector reverse scaling function
 * @param lhs Input tensor
 * @param scalar Scaling factor
 * @param Y Output tensor
 * @param size Tensor size
 * @param shift Shift amount
 * @return int32_t Operation status
 */
static int32_t calc_vec_rscale_luna(tTensor *lhs, int32_t scalar, tTensor *Y, int32_t size, int32_t shift) {
    void *src = (void *)lhs->dptr_;
    void *dst = (void *)Y->dptr_;
    int32_t rshift = fastlog2(scalar);
    int32_t lshift = shift - rshift;
    int32_t max_lshift = lhs->dtype_ == Int8 ? 6 : (lhs->dtype_ == Int16 ? 14 : 30);
    #if THINKER_PARAM_CHECK
    if (lshift > max_lshift || lshift < -63) {
        return (T_ERR_INVALID_PARA);
    }
    #endif
    int32_t in_idx = ((lhs->dtype_ & 0xF) >> 1);
    int32_t ou_idx = (Y->dtype_ & 0xF) >> 1;
    luna_vec_scale_api luna_vec_api = (luna_vec_scale_api)luna_vec_scale_api_items[in_idx][ou_idx];

    if (lshift < 0) {
        THINKER_RET_CHECK(luna_vec_api(src, 1, dst, size, -lshift), "luna_vec_api");
    } else if (lshift > 0) {
        THINKER_RET_CHECK(luna_vec_api(src, (1 << lshift), dst, size, 0), "luna_vec_api");
    } else if (src != dst) {
        memcpy(dst, src, size * lhs->byte_);
    }

    return T_SUCCESS;
}

/**
 * @brief Integer quantized division operation
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param Y Output tensor
 * @return int32_t Operation status
 */
int32_t iqdiv_luna(tTensor *lhs, tTensor *rhs, tTensor *Y) {
    #if THINKER_PARAM_CHECK
    if (lhs == NULL || rhs == NULL || Y == NULL ||
                        lhs->dptr_ == 0 || rhs->dptr_ == 0 || Y->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
    if (lhs->mem_.type_ != 2 || rhs->mem_.type_ != 2 ||
                        Y->mem_.type_ != 2) {
        return (T_ERR_NO_SUPPORT_OP);
    }
    if (!equalShape(&lhs->shape_, &Y->shape_) ||
                        (rhs->shape_.ndim_ != 0 &&
                         !equalShape(&lhs->shape_, &rhs->shape_))) {
        return (T_ERR_INVALID_DATA);
    }
    #endif
    int32_t x1_q = (int32_t)lhs->scale_;
    int32_t x2_q = (int32_t)rhs->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    int32_t shift = y_q - (x1_q - x2_q);
    #if THINKER_PARAM_CHECK
    if (lhs->zero_ != 0 || rhs->zero_ != 0 || Y->zero_ != 0 ||
                        shift < 0 || shift > 63) {
        return (T_ERR_INVALID_PARA);
    }
    #endif
    if (rhs->shape_.ndim_ == 0) {
        #if THINKER_PARAM_CHECK
        if (lhs->dtype_ != rhs->dtype_ || Y->dtype_ != lhs->dtype_ ||
                            (lhs->dtype_ != Int8 && lhs->dtype_ != Int16 &&
                             lhs->dtype_ != Int32)) {
            return (T_ERR_INVALID_DATATYPE);
        }
        #endif
    } else {
        #if THINKER_PARAM_CHECK
        if (lhs->dtype_ != Int32 || rhs->dtype_ != Int32 ||
                            Y->dtype_ != Int32 ||
                            iqdiv_has_zero_i32((const int32_t *)rhs->dptr_,
                                                getTensorSize(rhs))) {
            return (T_ERR_INVALID_DATATYPE);
        }
        #endif
    }
    size_t size = getTensorSize(lhs);

    if (0 == rhs->shape_.ndim_) {  // Scalar case
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
        #if THINKER_PARAM_CHECK
        if (scalar <= 0 || (scalar & (scalar - 1)) != 0) {
            return (T_ERR_INVALID_PARA);
        }
        #endif
        THINKER_RET_CHECK(calc_vec_rscale_luna(lhs, scalar, Y, size, shift), "calc_vec_rscale_luna");
    } else {  // Vector case
        THINKER_RET_CHECK(calc_vec_div_luna(lhs, rhs, Y, size), "calc_vec_div_luna");
    }

    return T_SUCCESS;
}

#endif
