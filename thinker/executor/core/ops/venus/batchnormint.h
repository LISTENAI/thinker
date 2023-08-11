#ifndef _BATCHNORMINT_VENUS_H_
#define _BATCHNORMINT_VENUS_H_

#include <math.h>

#include "c_api/thinker_define.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "luna/luna_math.h"
#include "thinker_status.h"

int32_t batchnormint_luna(const tTensor *X, const tTensor *W,
                          const tTensor *Bias, tTensor *Y, tTensor *workspace) {
  int32_t N = X->shape_.dims_[0];
  int32_t F = X->shape_.dims_[2] * X->shape_.dims_[3];
  int32_t C = X->shape_.dims_[1];
  int32_t one_batch_size = F * C;

  int8_t *p_src = (int8_t *)X->dptr_;
  int8_t *p_dst = (int8_t *)Y->dptr_;
  int8_t *p_weight = (int8_t *)W->dptr_;
  int32_t *p_bias = (int32_t *)Bias->dptr_;
  int32_t *p_tmp = (int32_t *)workspace->dptr_;

  int32_t q_x = (int32_t)X->scale_;
  int32_t q_w = (int32_t)W->scale_;
  int32_t q_o = (int32_t)Y->scale_;

  int32_t shift = q_x + q_w - q_o;

  for (int32_t i = 0; i < N; i++) {
    for (int32_t j = 0; j < C; j++) {
      int8_t w_val = *(p_weight + j);
      int32_t b_val = *(p_bias + j);

      int8_t *p_in = p_src + i * one_batch_size + j * F;
      int8_t *p_ou = p_dst + i * one_batch_size + j * F;

      luna_scale_q7_int32(p_in, w_val, p_tmp, F, 0);
      luna_offset_q31_int8(p_tmp, b_val, p_ou, F, shift);
    }
  }

  return 0;
}
#endif  //_BATCHNORMINT_VENUS_H_
