#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "luna/luna_math.h"
#include "thinker_status.h"

static int32_t calc_relu_luna(int32_t X_dtype, int32_t Y_dtype, void *src,
                              void *dst, int32_t size, int32_t shift) {
  int32_t ret = -1;
  switch (X_dtype) {
    case Int8: {
      switch (Y_dtype) {
        case Int8:
          ret = luna_relu_q7_int8((const q7_t *)src, (q7_t *)dst, size, shift);
          break;
        case Int16:
          ret =
              luna_relu_q7_int16((const q7_t *)src, (q15_t *)dst, size, shift);
          break;
        case Int32:
          ret =
              luna_relu_q7_int32((const q7_t *)src, (q31_t *)dst, size, shift);
          break;
      }
    } break;
    case Int16: {
      switch (Y_dtype) {
        case Int8:
          ret =
              luna_relu_q15_int8((const q15_t *)src, (q7_t *)dst, size, shift);
          break;
        case Int16:
          ret = luna_relu_q15_int16((const q15_t *)src, (q15_t *)dst, size,
                                    shift);
          break;
        case Int32:
          ret = luna_relu_q15_int32((const q15_t *)src, (q31_t *)dst, size,
                                    shift);
          break;
      }
    } break;
    case Int32: {
      switch (Y_dtype) {
        case Int8:
          ret =
              luna_relu_q31_int8((const q31_t *)src, (q7_t *)dst, size, shift);
          break;
        case Int16:
          ret = luna_relu_q31_int16((const q31_t *)src, (q15_t *)dst, size,
                                    shift);
          break;
        case Int32:
          ret = luna_relu_q31_int32((const q31_t *)src, (q31_t *)dst, size,
                                    shift);
          break;
      }
    } break;
  }
  return ret;
}

tStatus relu_luna(tTensor *X, tTensor *Y) {
  int32_t ret = -1;
  int32_t shift = 0;
  void *src = (void *)X->dptr_;
  void *dst = (void *)Y->dptr_;
  size_t size = getTensorSize(X);
  ret = calc_relu_luna(X->dtype_, Y->dtype_, src, dst, size, shift);
  return T_SUCCESS;
}
