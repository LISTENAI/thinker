#include <math.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "luna/luna_math.h"
#include "thinker_status.h"

int32_t iqsigmoid(tTensor *X, tTensor *Y) {
  const int32_t Q_INPUT = 11;
  const int32_t Q_OUTPUT = 7;
  int32_t ret = -1;
  int32_t x_q = (int32_t)X->scale_;
  int32_t y_q = (int32_t)Y->scale_;

  int16_t *src = (int16_t *)X->dptr_;
  int8_t *dst = (int8_t *)Y->dptr_;
  uint32_t size = getTensorSize(X);
  if (Q_INPUT != x_q) {
    luna_scale_q15_int16(src, 1, src, size, x_q - Q_INPUT);
  }

  ret = luna_sigmoid_int8(src, dst, size);

  return ret;
}
