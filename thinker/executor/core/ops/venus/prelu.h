#ifndef __PRELU_H__
#define __PRELU_H__

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "luna/luna_math.h"
#include "thinker_status.h"

typedef void *luna_prelu_api_item;
typedef int32_t (*luna_prelu_api)(const void *, uint32_t, void *, uint32_t size,
                                  uint32_t post_shift);
static luna_prelu_api_item luna_prelu_api_items[][3] = {
    {
        luna_prelu_q7_int8,
        luna_prelu_q7_int16,
        luna_prelu_q7_int32,
    },
    {
        luna_prelu_q15_int8,
        luna_prelu_q15_int16,
        luna_prelu_q15_int32,
    },
    {
        luna_prelu_q31_int8,
        luna_prelu_q31_int16,
        luna_prelu_q31_int32,
    }};

static int32_t calc_prelu(tTensor *X, tTensor *Y, uint32_t size, int32_t slope,
                          int32_t post_shift) {
  int32_t in_idx = (X->dtype_ & 0xF) >> 1;
  int32_t out_idx = (Y->dtype_ & 0xF) >> 1;
  luna_prelu_api luna_prelu =
      (luna_prelu_api)(luna_prelu_api_items[in_idx][out_idx]);
  int32_t ret = luna_prelu((const void *)X->dptr_, slope, (void *)Y->dptr_,
                           size, post_shift);
  return ret;
}

tStatus prelu_luna(tTensor *X, tTensor *Y, PreluAttrs *attrs) {
  tStatus status = T_ERR_FAIL;
  int32_t slope = attrs->slope;
  int32_t post_shift = attrs->post_shift;
  uint32_t size = getTensorSize(X);
  status = (tStatus)calc_prelu(X, Y, size, slope, post_shift);
  return T_SUCCESS;
}

#endif