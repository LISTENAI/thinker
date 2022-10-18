#include <math.h>
#include <string.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"

void tile_float(float *input, float *output, int32_t ndim,
                const uint32_t *inShape, const uint32_t *outShape,
                int64_t *repeat, int32_t in_size) {
  // Step 1: Place(memcpy) the original value to the right place
  uint32_t ishape_last = inShape[ndim - 1];
  uint32_t size = in_size / ishape_last;
  for (int32_t is = 0; is < size; is++) {
    int32_t istmp = is;
    int32_t os = 0;
    int32_t ostride = 1;
    for (int32_t i = 0; i < ndim - 1; i++) {
      ostride *= outShape[ndim - 1 - i];
      os += (istmp % inShape[ndim - 2 - i]) * ostride;
      istmp = istmp / inShape[ndim - 2 - i];
    }
    memcpy(output + os, input + is * ishape_last, sizeof(float) * ishape_last);
  }

  // Step 2: Repeat(memcpy) dim by dim in reverse order.
  int32_t len = ishape_last;
  for (int32_t od = ndim - 1; od >= 0; od--) {
    int32_t repeat_size = 1;
    for (int32_t id = 0; id < od; id++) {
      repeat_size *= inShape[id];
    }
    for (int32_t rl = 0; rl < repeat_size; rl++) {
      int32_t istmp = rl;
      int32_t ros = 0;
      int32_t ostride = 1;
      for (int32_t osi = ndim - 1; osi > od; osi--) {
        ostride *= outShape[osi];
      }
      for (int32_t i = 0; i < od; i++) {
        ostride *= outShape[od - i];
        ros += (istmp % inShape[od - 1 - i]) * ostride;
        istmp = istmp / inShape[od - 1 - i];
      }

      for (int32_t r = 1; r < repeat[od]; r++) {
        memcpy(output + ros + r * len, output + ros, sizeof(float) * len);
      }
    }
    len *= repeat[od];
    if (od > 0) {
      len *= inShape[od - 1];
    }
  }
}

void tile_int8(int8_t *input, int8_t *output, int32_t ndim,
               const uint32_t *inShape, const uint32_t *outShape,
               int64_t *repeat, int32_t in_size) {
  // Step 1: Place(memcpy) the original value to the right place
  uint32_t ishape_last = inShape[ndim - 1];
  uint32_t size = in_size / ishape_last;
  for (int32_t is = 0; is < size; is++) {
    int32_t istmp = is;
    int32_t os = 0;
    int32_t ostride = 1;
    for (int32_t i = 0; i < ndim - 1; i++) {
      ostride *= outShape[ndim - 1 - i];
      os += (istmp % inShape[ndim - 2 - i]) * ostride;
      istmp = istmp / inShape[ndim - 2 - i];
    }
    memcpy(output + os, input + is * ishape_last, sizeof(int8_t) * ishape_last);
  }

  // Step 2: Repeat(memcpy) dim by dim in reverse order.
  int32_t len = ishape_last;
  for (int32_t od = ndim - 1; od >= 0; od--) {
    int32_t repeat_size = 1;
    for (int32_t id = 0; id < od; id++) {
      repeat_size *= inShape[id];
    }
    for (int32_t rl = 0; rl < repeat_size; rl++) {
      int32_t istmp = rl;
      int32_t ros = 0;
      int32_t ostride = 1;
      for (int32_t osi = ndim - 1; osi > od; osi--) {
        ostride *= outShape[osi];
      }
      for (int32_t i = 0; i < od; i++) {
        ostride *= outShape[od - i];
        ros += (istmp % inShape[od - 1 - i]) * ostride;
        istmp = istmp / inShape[od - 1 - i];
      }

      for (int32_t r = 1; r < repeat[od]; r++) {
        memcpy(output + ros + r * len, output + ros, sizeof(int8_t) * len);
      }
    }
    len *= repeat[od];
    if (od > 0) {
      len *= inShape[od - 1];
    }
  }
}

int32_t tile_luna(tTensor *X, tTensor *xRepeat, tTensor *Y) {
  int32_t ndim = X->shape_.ndim_;
  if (ndim != xRepeat->shape_.dims_[0]) {
    return -1;
  }

  const uint32_t *inShape = X->shape_.dims_;
  const uint32_t *outShape = Y->shape_.dims_;
  int64_t *repeat = (int64_t *)xRepeat->dptr_;
  int32_t size = getShapeSize(&X->shape_);
  if (X->dtype_ == Int8) {
    int8_t *input = (int8_t *)X->dptr_;
    int8_t *output = (int8_t *)Y->dptr_;
    tile_int8(input, output, ndim, inShape, outShape, repeat, size);
  } else if (X->dtype_ == Float32) {
    float *input = (float *)X->dptr_;
    float *output = (float *)Y->dptr_;
    tile_float(input, output, ndim, inShape, outShape, repeat, size);
  }

  return 0;
}
