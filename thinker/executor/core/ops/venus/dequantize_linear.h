#include <string.h>

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "hifi/NatureDSP_Signal_math.h"
#include "hifi/NatureDSP_Signal_vector.h"
#include "luna/luna_math.h"
#include "thinker_status.h"

int32_t dequantize_linear_luna(tTensor *X, tTensor *Y, tTensor *Workspace) {
  if ((X->dtype_ != Int8) && (X->dtype_ != Uint8) && (X->dtype_ != Int32))
    return -1;
  // if(X_scale->dtype_ != Float32)
  // 	return -1;
  Y->shape_ = X->shape_;
  Y->dtype_ = Float32;

  size_t size = getTensorSize(X);
  int8_t scale = X->scale_;
  float *output = (float *)Y->dptr_;

  if (X->dtype_ == Int8) {
    int8_t *input = (int8_t *)X->dptr_;
    dequant8bit(input, output, size, scale);
  } else if (X->dtype_ == Uint8) {
    uint8_t *input = (uint8_t *)X->dptr_;
    dequantU8bit(input, output, size, scale);
  } else if (X->dptr_ == Int32) {
    int32_t *input = (int32_t *)X->dptr_;
    dequant32bit(input, output, size, scale);
  } else {
    return -1;
  }

  return 0;
}
