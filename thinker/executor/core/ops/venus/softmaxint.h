#ifndef _SOFTMAXINT_LUNA_H_
#define _SOFTMAXINT_LUNA_H_

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "hifi/NatureDSP_Signal_math.h"
#include "hifi/NatureDSP_Signal_vector.h"
#include "luna/luna_math.h"

int32_t softmaxint_luna(tTensor *data, tTensor *out, tTensor *Workspace,
                        SoftmaxIntAttrs *attrs)
{
  const int32_t SOFTMAX_Q_IN = 25;
  const int32_t SOFTMAX_Q_OUT = 15;
  int32_t leading = 1, stride = 1;
  int32_t i = 0;
  int32_t axis = 1;
  if (-1 == attrs->axis)
  {
    axis = data->shape_.ndim_ - 1;
  }
  for (; i < axis; ++i)
  {
    leading *= data->shape_.dims_[i];
  }
  for (; i < data->shape_.ndim_; ++i)
  {
    stride *= data->shape_.dims_[i];
  }
  int32_t data_size = leading * stride;

  if (Int8 == data->dtype_)
  {
    int8_t *src = (int8_t *)(data->dptr_);
    int8_t *dst = (int8_t *)(out->dptr_);
    int32_t *tmp1 = (int32_t *)(Workspace->dptr_);
    int32_t *tmp2 = (int32_t *)tmp1 + stride;
    int32_t x_scale = (int32_t)data->scale_;
    int32_t y_scale = (int32_t)out->scale_;
    int32_t workspace_size = Workspace->shape_.dims_[0];
    if (workspace_size >= (data_size << 2)) // data_size * sizeof(int32_t)
    {
      int8_t *src_tmp = src;
      int8_t *dst_tmp = dst;
      tmp2 = (int32_t *)tmp1 + data_size;
      if (data->mem_.type_ != 2)
      {
        src_tmp = (int8_t *)((int32_t *)(Workspace->dptr_) + data_size);
        memcpy(src_tmp, src, data_size);
      }
      if (out->mem_.type_ != 2)
        dst_tmp = (int8_t *)((int32_t *)(Workspace->dptr_) + data_size);

      luna_scale_q7_int32(src_tmp, 1, tmp1, data_size, 0);  // Q4->Q25
      luna_scale_q31_int32(tmp1, (1 << (SOFTMAX_Q_IN - x_scale)), tmp1, data_size, 0); // Q4->Q25
      for (int32_t l = 0; l < leading; ++l)
      {
        int32_t offset = l * stride;
        vec_softmax32x32((int32_t *)tmp1 + offset, (int32_t *)tmp1 + offset, stride);  // Q25->Q15        
      }
      luna_scale_q31_int8(tmp1, 1, dst_tmp, data_size, (SOFTMAX_Q_OUT - y_scale));
      if (out->mem_.type_ != 2)
      {
        memcpy(dst, dst_tmp, data_size);
      }
    }
    else
    {
      for (int32_t l = 0; l < leading; ++l)
      {
        int8_t *lsrc = src + l * stride;
        int8_t *ldst = dst + l * stride;
        if (data->mem_.type_ != 2)
        {
          lsrc = (int8_t *)((int32_t *)(Workspace->dptr_) + stride);
          memcpy(lsrc, src + l * stride, stride);
        }
        luna_scale_q7_int32(lsrc, 1, tmp1, stride, 0);  // Q4->Q25

        if (out->mem_.type_ != 2)
          ldst = (int8_t *)((int32_t *)(Workspace->dptr_) + stride);

        luna_scale_q31_int32(tmp1, (1 << (SOFTMAX_Q_IN - x_scale)), tmp2, stride,
                              0);                                     // Q4->Q25
        vec_softmax32x32((int32_t *)tmp1, (int32_t *)tmp2, stride);  // Q25->Q15
        luna_scale_q31_int8(tmp1, 1, ldst, stride, (SOFTMAX_Q_OUT - y_scale));

        if (out->mem_.type_ != 2)
        {
          memcpy(dst + l * stride, ldst, stride);
        }
      }
    }    
  }
  else
  {
    THINKER_LOG_FATAL("SoftmaxInt support int8 data type only.");
  }
  return 0;
}
#endif
