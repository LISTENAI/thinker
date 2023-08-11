#ifndef _SUM_LUNA_H_
#define _SUM_LUNA_H_

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "luna/luna_math.h"
#include "thinker_status.h"

typedef void *luna_sum_api_item;
typedef int32_t (*luna_sum_handle)(const void *, int32_t *, uint32_t, uint32_t);
static luna_sum_api_item luna_sum_items[3] = {
    luna_vector_sum_q7_int32,
    luna_vector_sum_q15_int32,
    luna_vector_sum_q31_int32,
};

int32_t iqsum_luna(tTensor *input, tTensor *Temp, tTensor *output,
                   iqSumAttrs *attrs) {
  int32_t ret = T_SUCCESS;
  int32_t axis = attrs->axis;
  size_t size = getTensorSize(input);

  if (axis < 0)
    axis += input->shape_.ndim_;

  if (axis != (input->shape_.ndim_ - 1))
  {
    return T_ERR_INVALID_PARA;
  }
  
  int32_t len = size / input->shape_.dims_[axis];
  int32_t shift = input->scale_ - output->scale_;
  for (int32_t i = 0 ; i < len; i++)
  {
    ret |= luna_vector_sum_q7_int32((const q7_t *)input->dptr_, (int32_t *)Temp->dptr_, input->shape_.dims_[axis], shift);
  }

  ret |= luna_scale_q31_int8((const q31_t *)Temp->dptr_, 1, (int8_t *)output->dptr_, size, 0);
  
  return ret;
}

#endif
