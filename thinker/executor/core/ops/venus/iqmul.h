#ifndef _MUL_LUNA_H_
#define _MUL_LUNA_H_

#include <math.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "luna/luna_math.h"

typedef int32_t (*VEC_MUL_LUNA_API)(void *src1, void *src2, void *dst,
                                    int32_t size, int32_t shift);
typedef int32_t (*VEC_SCALE_LUNA_API)(void *src, int32_t scalar, void *dst,
                                      int32_t size, int32_t shift);

struct luna_vec_mul_item {
  void *luna_api;
};

struct luna_vec_mul_item luna_vec_api_list[][3] = {
    {{luna_mul_q7_int8}, {luna_mul_q7_int16}, {luna_mul_q7_int32}},
    {{luna_mul_q15_int8}, {luna_mul_q15_int16}, {luna_mul_q15_int32}},
    {{luna_mul_q31_int8}, {luna_mul_q31_int16}, {luna_mul_q31_int32}},
    {{luna_scale_q7_int8}, {luna_scale_q7_int16}, {luna_scale_q7_int32}},
    {{luna_scale_q15_int8}, {luna_scale_q15_int16}, {luna_scale_q15_int32}},
    {{luna_scale_q31_int8}, {luna_scale_q31_int16}, {luna_scale_q31_int32}}};

int32_t calc_vec_mul_luna(tTensor *lhs, tTensor *rhs, tTensor *Y, int32_t size,
                          int32_t shift) {
  int32_t ret = T_ERR_FAIL;
  int32_t in_idx = (lhs->dtype_ & 0xF) >> 1;
  int32_t ou_idx = (Y->dtype_ & 0xF) >> 1;
  VEC_MUL_LUNA_API luna_vec_api =
      (VEC_MUL_LUNA_API)luna_vec_api_list[in_idx][ou_idx].luna_api;
  void *src1 = (void *)lhs->dptr_;
  void *src2 = (void *)rhs->dptr_;
  void *dst = (void *)Y->dptr_;
  ret = luna_vec_api(src1, src2, dst, size, shift);

  return ret;
}

int32_t calc_vec_scale_luna(tTensor *lhs, int32_t scalar, tTensor *Y,
                            int32_t size, int32_t shift) {
  int32_t ret = T_ERR_FAIL;
  int32_t in_idx = ((lhs->dtype_ & 0xF) >> 1) + 3;
  int32_t ou_idx = (Y->dtype_ & 0xF) >> 1;
  VEC_SCALE_LUNA_API luna_vec_api =
      (VEC_SCALE_LUNA_API)luna_vec_api_list[in_idx][ou_idx].luna_api;
  void *src = (void *)lhs->dptr_;
  // void *src2 = (void *)rhs->dptr_;
  void *dst = (void *)Y->dptr_;
  ret = luna_vec_api(src, scalar, dst, size, shift);

  return ret;
}

/*
mul(C*H*W,C*1*1) = C*H*W 
*/
int32_t calc_vec_mul_luna_b2b2_broadcast_h1w1(tTensor *lhs, tTensor *rhs, tTensor *Y, tTensor* Temp, int32_t shift)
{
	int32_t ret = T_ERR_FAIL;
	int32_t c = lhs->shape_.dims_[1];
	int32_t h = lhs->shape_.dims_[2];
	int32_t w = lhs->shape_.dims_[3];
 	int8_t *p_tmp1 = (int8_t *)Temp->dptr_;
 	int8_t *p_tmp2 = p_tmp1 + c;

	ret = luna_memset(p_tmp1, 1, h*w);
	ret |= luna_mat_mul_q7_int8((int8_t *)rhs->dptr_, p_tmp1, p_tmp2, c, 1, h*w, 0);
	ret |= luna_mul_q7_int8((int8_t *)lhs->dptr_, p_tmp2, (int8_t *)Y->dptr_, c*h*w, shift);

	return ret;
}


int32_t iqmul_luna(tTensor *lhs, tTensor *rhs, tTensor *Y, tTensor * Temp,
                   iqBinaryAttrs *attrs) {
  int32_t ret = T_ERR_FAIL;

  int32_t x1_q = (int32_t)lhs->scale_;
  int32_t x2_q = (int32_t)rhs->scale_;
  int32_t y_q = (int32_t)Y->scale_;
  int32_t shift = x1_q + x2_q - y_q;
  size_t size = getTensorSize(lhs);

  if (shift < 0) {
    return ret;
  }

  if (lhs->shape_.ndim_ == 4 && rhs->shape_.ndim_ == 4 && 
	  lhs->shape_.dims_[1] == rhs->shape_.dims_[1] && rhs->shape_.dims_[2] == 1 && rhs->shape_.dims_[2] == rhs->shape_.dims_[3])
  {
	  ret = calc_vec_mul_luna_b2b2_broadcast_h1w1(lhs, rhs, Y, Temp, shift);
  }
  else if (0 == rhs->shape_.ndim_)  // VS
  {
    int32_t scalar = *(int32_t *)rhs->dptr_;
    if (Int8 == rhs->dtype_) {
      scalar = (int32_t)(*(int8_t *)rhs->dptr_);
    } else if (Int16 == rhs->dtype_) {
      scalar = (int32_t)(*(int16_t *)rhs->dptr_);
    }
    ret = calc_vec_scale_luna(lhs, scalar, Y, size, shift);
  } else  // VV
  {
    ret = calc_vec_mul_luna(lhs, rhs, Y, size, shift);
  }

  return ret;
}
#endif
