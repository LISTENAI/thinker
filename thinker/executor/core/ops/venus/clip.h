#include <math.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "thinker_status.h"

#define CLIP(x, y, size, max, min)     \
  for (int32_t i = 0; i < size; i++) { \
    if (x[i] < min) {                  \
      y[i] = min;                      \
    } else if (x[i] > max) {           \
      y[i] = max;                      \
    } else {                           \
      y[i] = x[i];                     \
    }                                  \
  }

int32_t clip_venus(tTensor *X, tTensor *Y, ClipAttrs *attrs, float max,
                   float min) {
  int32_t size = getTensorSize(X);
  int16_t dtype = X->dtype_;
  switch (dtype) {
    case Int8: {
      int8_t *input = (int8_t *)X->dptr_;
      int8_t *output = (int8_t *)Y->dptr_;
      CLIP(input, output, size, max, min);
    } break;
    case Int16: {
      int16_t *input = (int16_t *)X->dptr_;
      int16_t *output = (int16_t *)Y->dptr_;
      CLIP(input, output, size, max, min);
    } break;
    case Int32: {
      int32_t *input = (int32_t *)X->dptr_;
      int32_t *output = (int32_t *)Y->dptr_;
      CLIP(input, output, size, max, min);
    } break;
    case Float32: {
      float *input = (float *)X->dptr_;
      float *output = (float *)Y->dptr_;
      CLIP(input, output, size, max, min);
    } break;
    default:
      THINKER_LOG_FATAL("not support data type!");
      break;
  }
  return 0;
}
