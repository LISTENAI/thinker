#include <math.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "thinker_status.h"

int32_t gather_luna(tTensor *X, tTensor *indices, tTensor *Y,
                    GatherAttrs *attr) {
  int32_t axis = attr->axis;
  axis = (axis < 0) ? (X->shape_.ndim_ + axis) : axis;

  int32_t ndim = indices->shape_.dims_[0];
  for (int32_t i = 1; i < indices->shape_.ndim_; ++i) {
    ndim *= indices->shape_.dims_[i];
  }

  if (ndim == 0) {
    ndim = 1;
  }

  int32_t leading = 1;
  int32_t i = 0;
  for (; i < attr->axis; ++i) {
    leading *= X->shape_.dims_[i];
  }
  int32_t middle = X->shape_.dims_[i++];
  int32_t tail = 1;
  for (; i < X->shape_.ndim_; ++i) {
    tail *= X->shape_.dims_[i];
  }

  const int8_t *input = (int8_t *)(X->dptr_);
  int32_t *index = (int32_t *)(indices->dptr_);
  int8_t *output = (int8_t *)(Y->dptr_);

  for (int32_t l = 0; l < leading; ++l) {
    for (int32_t j = 0; j < ndim; ++j) {
      int32_t idx = index[j];
      if (idx == -1) {
        idx = X->shape_.dims_[axis] - 1;
      }
      memcpy(output + (l * ndim * tail + j * tail) * X->byte_,
             input + (l * middle * tail + idx * tail) * X->byte_,
             X->byte_ * tail);
    }
  }

  return 0;
}
