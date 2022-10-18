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

int32_t iqmul_luna(tTensor *lhs, tTensor *rhs, tTensor *Y,
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

  if (0 == rhs->shape_.ndim_)  // VS
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
