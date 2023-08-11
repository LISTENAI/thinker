#ifndef _SQUEEZE_LUNA_H_
#define _SQUEEZE_LUNA_H_

#include <math.h>
#include <string.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"

int32_t squeeze_luna(tTensor* X, tTensor* Y) {
  int32_t size = getShapeSize(&X->shape_);
  if (X->dptr_ == Y->dptr_) {
    return 0;
  }
  memcpy((void*)Y->dptr_, (void*)X->dptr_, X->byte_ * size);
  return 0;
}
#endif
