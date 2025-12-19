#ifndef _DIV_LUNA_H_
#define _DIV_LUNA_H_

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "thinker_status.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief Vector division API type definition
 */
typedef int32_t (*luna_vec_scale_api)(void *src, int32_t scalar, void *dst, int32_t size, int32_t shift);
typedef void *luna_vec_scale_api_item;

/**
 * @brief Vector operation API items
 */
static luna_vec_scale_api_item luna_vec_scale_api_items[][2] = {
    {API_LIB(scale_i8i8o8),   API_LIB(scale_i8i8o32),},
    {API_LIB(scale_i32i32o8), API_LIB(scale_i32i32o32),},
};

/**
 * @brief Calculate vector division
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param Y Output tensor
 * @param size Tensor size
 * @return int32_t Operation status
 */
static int32_t calc_vec_div_luna(tTensor *lhs, tTensor *rhs, tTensor *Y, int32_t size) 
{
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    int32_t x1_q = (int32_t)lhs->scale_;
    int32_t x2_q = (int32_t)rhs->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    void *src1 = (void *)lhs->dptr_;
    void *src2 = (void *)rhs->dptr_;
    void *dst = (void *)Y->dptr_;
    uint32_t shift = y_q - x1_q + x2_q;

    switch (lhs->dtype_)
    {
        case Int32:
            ret = API_LIB(div_i32i32o32)((const int32_t *)src1, (const int32_t *)src2, (int32_t *)dst, size, shift);
            break;
        default:
            THINKER_LOG_FATAL("data type not support!");
            break;
    }

    return ret;
}

/**
 * @brief Calculate vector right shift scale
 * @param lhs Input tensor
 * @param scalar Scale factor
 * @param Y Output tensor
 * @param size Tensor size
 * @param shift Shift amount
 * @return int32_t Operation status
 */
static int32_t calc_vec_rscale_luna(tTensor *lhs, int32_t scalar, tTensor *Y, int32_t size, int32_t shift) 
{
  int32_t ret     = T_ERR_FAIL;

  void *src       = (void *)lhs->dptr_;
  void *dst       = (void *)Y->dptr_;
  int32_t rshift  = log2f(scalar);
  int32_t lshift  = shift - rshift;
  int32_t in_idx  = ((lhs->dtype_ & 0xF) >> 1);
  int32_t ou_idx  = (Y->dtype_ & 0xF) >> 1;
  luna_vec_scale_api luna_vec_api = (luna_vec_scale_api)luna_vec_scale_api_items[in_idx][ou_idx];

  if (lshift < 0) {
    ret = luna_vec_api(src, 1, dst, size, -lshift);
  } else if (lshift > 0) {
    ret = luna_vec_api(src, (1 << lshift), dst, size, 0);
  }
  return ret;
}

/**
 * @brief Integer Quantized Division operation
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param Y Output tensor
 * @return int32_t Operation status
 */
int32_t iqdiv_luna(tTensor *lhs, tTensor *rhs, tTensor *Y) 
{
    int32_t ret = T_ERR_FAIL;
    int32_t x1_q = (int32_t)lhs->scale_;
    int32_t x2_q = (int32_t)rhs->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    int32_t shift = y_q - (x1_q - x2_q);
    size_t size = getTensorSize(lhs);

    if (rhs->shape_.ndim_ == 0)  // Scalar division
    {
        int32_t scalar = 1;
        if (rhs->dtype_ == Int8)
        {
            scalar = (int32_t)(*(int8_t *)rhs->dptr_);
        }
        else if (rhs->dtype_ == Int16)
        {
            scalar = (int32_t)(*(int16_t *)rhs->dptr_);
        }
        else if (rhs->dtype_ == Int32)
        {
            scalar = (int32_t)(*(int32_t *)rhs->dptr_);
        }

        ret = calc_vec_rscale_luna(lhs, scalar, Y, size, shift);
    }
    else  // Tensor division
    {
        ret = calc_vec_div_luna(lhs, rhs, Y, size);
    }

    return ret;
}

#endif