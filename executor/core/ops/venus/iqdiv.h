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
    int32_t ret = T_ERR_FAIL;
    int32_t x1_q = (int32_t)lhs->scale_;
    int32_t x2_q = (int32_t)rhs->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    void *src1 = (void *)lhs->dptr_;
    void *src2 = (void *)rhs->dptr_;
    void *dst = (void *)Y->dptr_;

    switch (lhs->dtype_) {
        case Int32:
            API_LIB(div_q31_int32)((const q31_t *)src1, x1_q, (const q31_t *)src2, x2_q, (q31_t *)dst, y_q, size);
            break;
        default:
            THINKER_LOG_FATAL("data type not support!");
            break;
    }

    return ret;
}

/**
 * @brief Fast base-2 logarithm approximation
 * @param x Input integer
 * @return int32_t Approximated log2(x)
 */
static int32_t fastlog2(int32_t x) {
    float fx = (float)x;
    int8_t *fx_addr = (int8_t *)&fx;
    uint32_t ix = (uint32_t)(*(uint32_t *)fx_addr);
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
    int32_t ret = T_ERR_FAIL;
    void *src = (void *)lhs->dptr_;
    void *dst = (void *)Y->dptr_;
    int32_t rshift = fastlog2(scalar);
    int32_t lshift = shift - rshift;
    int32_t in_idx = ((lhs->dtype_ & 0xF) >> 1);
    int32_t ou_idx = (Y->dtype_ & 0xF) >> 1;
    luna_vec_scale_api luna_vec_api = (luna_vec_scale_api)luna_vec_scale_api_items[in_idx][ou_idx];

    if (lshift < 0) {
        ret = luna_vec_api(src, 1, dst, size, -lshift);
    } else if (lshift > 0) {
        ret = luna_vec_api(src, (1 << lshift), dst, size, 0);
    }

    return ret;
}

/**
 * @brief Integer quantized division operation
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param Y Output tensor
 * @return int32_t Operation status
 */
int32_t iqdiv_luna(tTensor *lhs, tTensor *rhs, tTensor *Y) {
    int32_t ret = T_ERR_FAIL;
    int32_t x1_q = (int32_t)lhs->scale_;
    int32_t x2_q = (int32_t)rhs->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    int32_t shift = y_q - (x1_q - x2_q);
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
                ret = T_ERR_INVALID_DATATYPE;
                break;
        }
        if (ret == T_ERR_FAIL) {
            ret = calc_vec_rscale_luna(lhs, scalar, Y, size, shift);
        }
    } else {  // Vector case
        ret = calc_vec_div_luna(lhs, rhs, Y, size);
    }

    return ret;
}

#endif