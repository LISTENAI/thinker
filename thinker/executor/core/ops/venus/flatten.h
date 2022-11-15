#include <math.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"

int32_t flatten_luna(tTensor *X, tTensor *Y, FlattenAttrs *attr) {
  int8_t *input = (int8_t *)X->dptr_;
  int8_t *output = (int8_t *)Y->dptr_;
  if (input != output) {
    size_t size = getTensorSize(X);
    memcpy(output, input, X->byte_ * size);
  }

  return 0;
}