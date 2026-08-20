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

static int32_t iqdiv_has_zero_i32(const int32_t *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (data[i] == 0) return 1;
    }
    return 0;
}

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
            THINKER_RET_CHECK(API_LIB(div_i32i32o32)((const int32_t *)src1, (const int32_t *)src2, (int32_t *)dst, size, shift), "luna_div_i32i32o32");
            break;
        default:
            THINKER_LOG_FATAL("data type not support!");
            return T_ERR_INVALID_DATATYPE;
    }

    return T_SUCCESS;
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
static int32_t calc_vec_rscale_luna(tTensor *lhs, int32_t scalar, tTensor *Y, int32_t size, int32_t shift) {
  void *src       = (void *)lhs->dptr_;
  void *dst       = (void *)Y->dptr_;
  int32_t rshift  = log2f(scalar);
  int32_t lshift  = shift - rshift;
  int32_t in_idx  = ((lhs->dtype_ & 0xF) >> 1);
  int32_t ou_idx  = (Y->dtype_ & 0xF) >> 1;
  luna_vec_scale_api luna_vec_api = (luna_vec_scale_api)luna_vec_scale_api_items[in_idx][ou_idx];

  if (lshift < 0) {
    THINKER_RET_CHECK(luna_vec_api(src, 1, dst, size, -lshift), "luna_vec_api");
  } else if (lshift > 0) {
    THINKER_RET_CHECK(luna_vec_api(src, (1 << lshift), dst, size, 0), "luna_vec_api");
  }
  return T_SUCCESS;
}

/**
 * @brief Integer Quantized Division operation
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

    if (lhs->zero_ != 0 || rhs->zero_ != 0 || Y->zero_ != 0) {
        return (T_ERR_INVALID_PARA);
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
    size_t size = getTensorSize(lhs);

    if (rhs->shape_.ndim_ == 0)  // Scalar division
    {
        #if THINKER_PARAM_CHECK
        if (lhs->dtype_ != rhs->dtype_ || Y->dtype_ != lhs->dtype_ ||
                            (lhs->dtype_ != Int8 && lhs->dtype_ != Int32)) {
            return (T_ERR_INVALID_DATATYPE);
        }
#endif
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
        else
        {
            return T_ERR_INVALID_DATATYPE;
        }
        #if THINKER_PARAM_CHECK
        if (scalar <= 0 || (scalar & (scalar - 1)) != 0) {
            return (T_ERR_INVALID_PARA);
        }
#endif
        int32_t lshift = shift - (int32_t)log2f((float)scalar);
        #if THINKER_PARAM_CHECK
        if (lshift < -63 ||
                            lshift > (lhs->dtype_ == Int8 ? 6 :
                                      (lhs->dtype_ == Int16 ? 14 : 30))) {
            return (T_ERR_INVALID_PARA);
        }
#endif

        THINKER_RET_CHECK(calc_vec_rscale_luna(lhs, scalar, Y, size, shift), "calc_vec_rscale_luna");
    }
    else  // Tensor division
    {
        #if THINKER_PARAM_CHECK
        if (lhs->dtype_ != Int32 || rhs->dtype_ != Int32 ||
                            Y->dtype_ != Int32 ||
                            iqdiv_has_zero_i32((const int32_t *)rhs->dptr_,
                                                getTensorSize(rhs))) {
            return (T_ERR_INVALID_DATATYPE);
        }
#endif
        THINKER_RET_CHECK(calc_vec_div_luna(lhs, rhs, Y, size), "calc_vec_div_luna");
    }

    return T_SUCCESS;
}

#endif
