#ifndef _ADD_LUNA_H_
#define _ADD_LUNA_H_

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "luna/luna_math.h"
#include "thinker_status.h"

int32_t iqadd_luna(tTensor *X1, tTensor *X2, tTensor *Temp, tTensor *Y) {
  int32_t ret = -1;

  int32_t x1_q = (int32_t)X1->scale_;
  int32_t x2_q = (int32_t)X2->scale_;
  int32_t y_q = (int32_t)Y->scale_;
  void *src1 = (void *)X1->dptr_;
  void *src2 = (void *)X2->dptr_;
  void *dst = (void *)Y->dptr_;
  size_t size = getTensorSize(X1);

  if ((x1_q < y_q) || x2_q < y_q) {
    return ret;
  }

  if ((1 == X1->mem_.type_ || 3 == X1->mem_.type_) &&
      (Temp))  // need copy psram to share
  {
    src1 = (void *)Temp->dptr_;
    memcpy(src1, (void *)X1->dptr_, size * X1->byte_);
  }

  if ((1 == X2->mem_.type_ || 3 == X2->mem_.type_) &&
      (Temp))  // need copy psram to share
  {
    src2 = (void *)Temp->dptr_;
    memcpy(src2, (void *)X2->dptr_, size * X2->byte_);
  }

  if (equalShape(&X1->shape_, &X2->shape_) && (X1->dtype_ == X2->dtype_)) {
    int32_t shift1 = x1_q - y_q;
    int32_t shift2 = x2_q - y_q;
    switch (X1->dtype_) {
      case Int8: {
        ret = luna_scale_q7_int8((const q7_t *)src1, (1), (int8_t *)dst, size,
                                 shift1);
        ret = luna_scale_q7_int8((const q7_t *)src2, (1), (int8_t *)src2, size,
                                 shift2);
        ret = luna_add_q7_int8((const q7_t *)dst, (q7_t *)src2, (int8_t *)dst,
                               size, 0);
      } break;
      case Int16: {
        ret = luna_scale_q15_int16((const q15_t *)src1, (1), (int16_t *)dst,
                                   size, shift1);
        ret = luna_scale_q15_int16((const q15_t *)src2, (1), (int16_t *)src2,
                                   size, shift2);
        ret = luna_add_q15_int16((const q15_t *)dst, (q15_t *)src2,
                                 (int16_t *)dst, size, 0);
      } break;
      case Int32: {
        ret = luna_scale_q31_int32((const q31_t *)src1, (1), (int32_t *)dst,
                                   size, shift1);
        ret = luna_scale_q31_int32((const q31_t *)src2, (1), (int32_t *)src2,
                                   size, shift2);
        ret = luna_add_q31_int32((const q31_t *)dst, (q31_t *)src2,
                                 (int32_t *)dst, size, 0);
      } break;
      default:
        break;
    }
  }

  return ret;
}

#endif
