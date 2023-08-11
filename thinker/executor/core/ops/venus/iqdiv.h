#ifndef _DIV_LUNA_H_
#define _DIV_LUNA_H_

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "luna/luna_math.h"
#include "thinker_status.h"

typedef int32_t (*luna_vec_scale_api)(void *src, int32_t scalar, void *dst,
                                      int32_t size, int32_t shift);
typedef void *luna_vec_scale_api_item;

static luna_vec_scale_api_item luna_vec_scale_api_items[][3] = {
    {
        luna_scale_q7_int8,
        luna_scale_q7_int16,
        luna_scale_q7_int32,
    },
    {
        luna_scale_q15_int8,
        luna_scale_q15_int16,
        luna_scale_q15_int32,
    },
    {
        luna_scale_q31_int8,
        luna_scale_q31_int16,
        luna_scale_q31_int32,
    },
};
int32_t calc_vec_div_luna(tTensor *lhs, tTensor *rhs, tTensor *Y,
                          int32_t size) {
  int32_t ret = T_ERR_FAIL;
  int32_t x1_q = (int32_t)lhs->scale_;
  int32_t x2_q = (int32_t)rhs->scale_;
  int32_t y_q = (int32_t)Y->scale_;
  void *src1 = (void *)lhs->dptr_;
  void *src2 = (void *)rhs->dptr_;
  void *dst = (void *)Y->dptr_;
  switch (lhs->dtype_) {
    case Int32:
      luna_div_q31_int32((const q31_t *)src1, x1_q, (const q31_t *)src2, x2_q,
                         (q31_t *)dst, y_q, size);
      break;
    default:
      THINKER_LOG_FATAL("data type not support!");
      break;
  }

  return ret;
}

static int32_t fastlog2(int32_t x) {
  float fx = (float)x;
  int8_t *fx_addr = (int8_t *)&fx;
  uint32_t ix = (uint32_t)(*(uint32_t *)fx_addr);
  uint32_t exp = 0;
  exp = (ix >> 23) && 0xFF;
  return exp - 127;
}

int32_t calc_vec_rscale_luna(tTensor *lhs, int32_t scalar, tTensor *Y,
                             int32_t size, int32_t shift) {
  int32_t ret = T_ERR_FAIL;

  void *src = (void *)lhs->dptr_;
  // void *src2 = (void *)rhs->dptr_;
  void *dst = (void *)Y->dptr_;
  int32_t rshift = log2f(scalar);
  int32_t lshift = shift - rshift;
  int32_t in_idx = ((lhs->dtype_ & 0xF) >> 1);
  int32_t ou_idx = (Y->dtype_ & 0xF) >> 1;
  luna_vec_scale_api luna_vec_api =
      (luna_vec_scale_api)luna_vec_scale_api_items[in_idx][ou_idx];

  if (lshift < 0) {
    ret = luna_vec_api(src, 1, dst, size, -lshift);
  } else if (lshift > 0) {
    ret = luna_vec_api(src, (1 << lshift), dst, size, 0);
  }
  return ret;
}

int32_t iqdiv_luna(tTensor *lhs, tTensor *rhs, tTensor *Y) {
  int32_t ret = T_ERR_FAIL;

  int32_t x1_q = (int32_t)lhs->scale_;
  int32_t x2_q = (int32_t)rhs->scale_;
  int32_t y_q = (int32_t)Y->scale_;
  int32_t shift = y_q - (x1_q - x2_q);
  size_t size = getTensorSize(lhs);

  if (0 == rhs->shape_.ndim_)  // VS
  {
    int32_t scalar = 1;
    if (Int8 == rhs->dtype_) {
      scalar = (int32_t)(*(int8_t *)rhs->dptr_);
    } else if (Int16 == rhs->dtype_) {
      scalar = (int32_t)(*(int16_t *)rhs->dptr_);
    } else if (Int32 == rhs->dtype_) {
      scalar = (int32_t)(*(int32_t *)rhs->dptr_);
    }

    ret = calc_vec_rscale_luna(lhs, scalar, Y, size, shift);
  } else  // VV
  {
    ret = calc_vec_div_luna(lhs, rhs, Y, size);
  }

  return ret;
}

#endif