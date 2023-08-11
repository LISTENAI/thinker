#ifndef __QUANT_H__
#define __QUANT_H__

#include <stdint.h>
#include <stdio.h>

#include "core/comm/utils.h"
#include "hifi/NatureDSP_Signal_math.h"
#include "hifi/NatureDSP_Signal_vector.h"
#include "luna/luna_math.h"

int32_t quantize_linear_luna(tTensor *X, tTensor *Y, tTensor *Workspace,
                             QuantAttrs *attr) {
  int32_t data_bits = attr->data_bits;
  int32_t quant_type = attr->quant_type;
  if (data_bits != 8 && data_bits != 16 && data_bits != 32) return -1;

  if (X->dtype_ != Float32) {
    return -1;
  }

  size_t size = getTensorSize(X);

  if (X->dtype_ == Float32) {
    float *input = (float *)X->dptr_;
    int8_t scale = Y->scale_;
    int8_t *output = (int8_t *)Y->dptr_;
    quant(input, output, size, scale);
  }

  return 0;
}
#endif
