#ifndef _EXPAND_LUNA_H_
#define _EXPAND_LUNA_H_

#include <stdio.h>
#include <string.h>

#include "../../comm/type_switch.h"
#include "core/operator_register.h"

int32_t expand_luna(tTensor *X, tTensor *Y) {
  int32_t xdim = X->shape_.ndim_;
  int32_t ydim = Y->shape_.ndim_;
  const uint32_t *tShape = X->shape_.dims_;
  const uint32_t *yshape = Y->shape_.dims_;

  int32_t bl = ydim - xdim;
  int32_t leading = 1;
  for (int32_t i = 0; i < bl; ++i) {
    leading *= yshape[i];
  }

  int32_t size = 1;
  uint32_t expandshape[7];
  for (int32_t i = bl; i < ydim; ++i) {
    size *= yshape[i];
    expandshape[i - bl] = yshape[i];
  }

  DATA_TYPE_SWITCH_ALL(X->dtype_, Type, {
    const Type *input = (Type *)X->dptr_;
    Type *output = (Type *)Y->dptr_;
    int32_t ndim = xdim;
    int32_t input_accumu[7];
    int32_t output_accumu[7];
    input_accumu[ndim - 1] = output_accumu[ndim - 1] = 1;
    for (int32_t i = ndim - 1; i > 0; i--) {
      input_accumu[i - 1] = input_accumu[i] * tShape[i];
      output_accumu[i - 1] = output_accumu[i] * expandshape[i];
    }
    for (int32_t i = 0; i < size; ++i) {
      int32_t inputIdx = 0;
      int32_t i_ = i;
      for (int32_t j = 0; j < ndim; ++j) {
        int32_t outIdx = i_ / output_accumu[j];
        inputIdx += (outIdx % tShape[j]) * input_accumu[j];
        i_ %= output_accumu[j];
      }
      output[i] = input[inputIdx];
    }
    for (int32_t i = 1; i < leading; ++i)
      memcpy(output + i * size, output, size * sizeof(Type));
  });

  return 0;
}
#endif  //_EXPAND_VENUS_H_
