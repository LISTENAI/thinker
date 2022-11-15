#include <math.h>
#include <stdio.h>
#include <string.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "luna/luna_math.h"

int32_t requantizelinear_luna(tTensor* X, tTensor* Y) {
  if (X->dtype_ != Int8) return -1;

  size_t size = getTensorSize(X);
  int8_t* input = (int8_t*)X->dptr_;

  int32_t src_bits = (X->byte_) * 8;
  int32_t dst_bits = (Y->byte_) * 8;
  int32_t q_x = (int32_t)X->scale_;
  int32_t q_y = (int32_t)Y->scale_;

  if (dst_bits > src_bits) {
    int8_t* input = (int8_t*)X->dptr_;
    if (32 == dst_bits) {
      int32_t* output = (int32_t*)Y->dptr_;
      for (int32_t i = 0; i < size; ++i) {
        output[i] = input[i] << (q_y - q_x);
      }
    } else if (16 == dst_bits) {
      int16_t* output = (int16_t*)Y->dptr_;
      for (int32_t i = 0; i < size; ++i) {
        output[i] = input[i] << (q_y - q_x);
      }
    }
  } else {
    ;
  }

  return 0;
}
