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
 * @brief Function pointer type for vector scaling operations
 */
typedef int32_t (*luna_vec_scale_api)(void *src, int32_t scalar, void *dst, int32_t size, int32_t shift);
typedef void *luna_vec_scale_api_item;
/**
 * @brief Mapping of vector scaling functions for different data types
 */
static luna_vec_scale_api_item luna_vec_scale_api_items[][2] = {
    {API_LIB(scale_i8i8o8),   API_LIB(scale_i8i8o32),},
    {API_LIB(scale_i32i32o8), API_LIB(scale_i32i32o32),},
};

/**
 * @brief Performs vector division with quantization adjustment
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param Y Output tensor
 * @param size Number of elements to process
 * @return int32_t Operation status
 */
static int32_t calc_vec_div_luna(tTensor *lhs, tTensor *rhs, tTensor *Y, int32_t size) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    switch (lhs->dtype_) {
        case Int32:
            ret = API_LIB(div_i32i32o32)((const int32_t *)lhs->dptr_, (const int32_t *)rhs->dptr_, (int32_t *)Y->dptr_, size, 0);
            break;
        default:
            printf("Data type not supported!");
            break;
    }

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
static int32_t calc_vec_rscale_luna(tTensor *lhs, int32_t scalar, tTensor *Y, int32_t size, int32_t shift) {
    int32_t ret = T_ERR_FAIL;

    int32_t rshift = log2f(scalar);
    int32_t lshift = shift - rshift;
    int32_t in_idx = (lhs->dtype_ & 0xF) >> 1;
    int32_t ou_idx = (Y->dtype_ & 0xF) >> 1;
    luna_vec_scale_api luna_vec_api = (luna_vec_scale_api)luna_vec_scale_api_items[in_idx][ou_idx];

    if (lshift < 0) {
        ret = luna_vec_api((void *)lhs->dptr_, 1, (void *)Y->dptr_, size, -lshift);
    } else if (lshift > 0) {
        ret = luna_vec_api((void *)lhs->dptr_, (1 << lshift), (void *)Y->dptr_, size, 0);
    }

    return ret;
}

/**
 * @brief Quantized division operation implementation
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param Y Output tensor
 * @return int32_t Operation status
 */
int32_t iqdiv_luna(tTensor *lhs, tTensor *rhs, tTensor *Y) {
    int32_t ret = T_ERR_FAIL;
    size_t size = getTensorSize(lhs);

    // Calculate quantization shift
    int32_t lhs_scale = (int32_t)lhs->scale_;
    int32_t rhs_scale = (int32_t)rhs->scale_;
    int32_t output_scale = (int32_t)Y->scale_;
    int32_t shift = output_scale - (lhs_scale - rhs_scale);

    // Check if right-hand side is a scalar
    if (rhs->shape_.ndim_ == 0) {
        int32_t scalar = 1;
        if (Int8 == rhs->dtype_) {
          scalar = (int32_t)(*(int8_t *)rhs->dptr_);
        } else if (Int16 == rhs->dtype_) {
          scalar = (int32_t)(*(int16_t *)rhs->dptr_);
        } else if (Int32 == rhs->dtype_) {
          scalar = (int32_t)(*(int32_t *)rhs->dptr_);
        }

        ret = calc_vec_rscale_luna(lhs, scalar, Y, size, shift);
    } else {
        ret = calc_vec_div_luna(lhs, rhs, Y, size);
    }

    return ret;
}

#endif  // _DIV_LUNA_H_