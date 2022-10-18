#include <math.h>
#include <string.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"

int32_t unsqueeze_luna(tTensor* X, tTensor* Y) {
  int32_t size = getShapeSize(&X->shape_);
  if (X->dptr_ == Y->dptr_) {
    return 0;
  }
  memcpy((int8_t*)Y->dptr_, (int8_t*)X->dptr_, X->byte_ * size);

  return 0;
}
