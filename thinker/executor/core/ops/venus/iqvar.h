#ifndef _VAR_LUNA_H_
#define _VAR_LUNA_H_

#include <math.h>

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "luna/luna_math.h"
#include "thinker_status.h"

int32_t iqvar(tTensor *X, tTensor *Y, tTensor *temp, iqvarAttrs *attrs) {
  int32_t ret = T_ERR_FAIL;
  int32_t x_q = (int32_t)X->scale_;
  int32_t y_q = (int32_t)Y->scale_;
  int8_t *src = (int8_t *)X->dptr_;
  int8_t *dst = (int8_t *)Y->dptr_;
  int32_t n_dim = X->shape_.ndim_;
  int32_t dims = attrs->dims;
  int32_t leading = X->shape_.dims_[n_dim - 3] * X->shape_.dims_[n_dim - 2];
  int32_t F = X->shape_.dims_[n_dim - 1];
  size_t input_size = getTensorSize(X);

  if (Int8 == X->dtype_) {
    int32_t shift = x_q * 2 - y_q;
    int8_t *p_tmp = (int8_t *)temp->dptr_;
    int32_t *p_tmp2 = (int32_t *)((int8_t *)temp->dptr_ + input_size);
    if (-1 == dims || (n_dim - 1) == dims) {
      p_tmp2 = (int32_t *)temp->dptr_;
    } else {
      leading = X->shape_.dims_[n_dim - 3] * X->shape_.dims_[n_dim - 1];
      F = X->shape_.dims_[n_dim - 2];
      uint32_t axis[3] = {0, 2, 1};
      uint32_t in_shape[3];
      in_shape[0] = X->shape_.dims_[n_dim - 3];
      in_shape[1] = X->shape_.dims_[n_dim - 2];
      in_shape[2] = X->shape_.dims_[n_dim - 1];
      luna_trans_axis_q7(src, p_tmp, in_shape, axis, 3);
      src = p_tmp;
    }

    int32_t *sum_x = p_tmp2 + F;
    int32_t *sum_x2 = p_tmp2 + F + 1;
    for (int32_t i = 0; i < leading; i++) {
      int8_t *p_src_once = src + i * F;
      int8_t *p_dst_once = dst + i;

      // cal sum(x^2)
      luna_mul_q7_int32(p_src_once, p_src_once, p_tmp2, F, 0);
      luna_vector_sum_q31_int32((const int32_t *)p_tmp2, sum_x2, F, 0);
      // cal sum(x)
      luna_vector_sum_q7_int32(p_src_once, sum_x, F, 0);

      int32_t sum_x_val = *sum_x;
      int32_t sum_x2_val = *sum_x2;
      int32_t numerator = F * sum_x2_val - sum_x_val * sum_x_val;
      float tmp_out;
      if (F > 1) {
        tmp_out = numerator * 1.0f / (F * (F - 1) * (1 << shift));
      } else {
        tmp_out = numerator * 1.0f / (1 << shift);
      }
      quant(&tmp_out, p_dst_once, 1, 0);
    }
  }
  return 0;
}
#endif
