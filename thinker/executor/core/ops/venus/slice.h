#ifndef __SLICE_H__
#define __SLICE_H__

#include <string.h>

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "thinker_status.h"

tStatus slice_luna(tTensor* X, int32_t begin, int32_t end, int32_t axis,
                   int32_t step, tTensor* Y) {
  int32_t real_axis = (axis + X->shape_.ndim_) % X->shape_.ndim_;
  int32_t real_begin;
  int32_t x_shape = X->shape_.dims_[real_axis];
  if (begin + x_shape >= 0) {
    real_begin =
        (begin + X->shape_.dims_[real_axis]) % X->shape_.dims_[real_axis];
  } else {
    real_begin = 0;
  }

  // If the first axis, Implement memcpy directly,
  // Otherwise use the "leading, mid, trailing" architecture.
  if (real_axis == 0) {
    int32_t start = real_begin;
    for (int32_t i = 1; i < X->shape_.ndim_; i++) {
      start *= X->shape_.dims_[i];
    }

    int32_t output_size = 1;
    for (size_t i = 0; i < Y->shape_.ndim_; i++) {
      output_size *= Y->shape_.dims_[i];
    }
    memcpy((int8_t*)Y->dptr_, (int8_t*)X->dptr_ + start * X->byte_,
           output_size * Y->byte_);
    return T_SUCCESS;
  }

  // else
  int32_t leading = 1, trailing = 1;
  int32_t mid = X->shape_.dims_[real_axis];
  for (int32_t i = 0; i < real_axis; ++i) {
    leading *= X->shape_.dims_[i];
  }
  for (int32_t i = real_axis + 1; i < X->shape_.ndim_; ++i) {
    trailing *= X->shape_.dims_[i];
  }
  int32_t i_mt = mid * trailing;
  int32_t o_mt = Y->shape_.dims_[real_axis] * trailing;
  int32_t offset = real_begin * trailing;
  if (1 == X->byte_){
    for (int32_t l = 0; l < leading; l++) {
      int32_t i_lmt_this = l * i_mt + offset;
      int32_t o_lmt_this = l * o_mt;
      memcpy((int8_t*)Y->dptr_ + o_lmt_this,
            (int8_t*)X->dptr_ + i_lmt_this,
            o_mt);
    }
  }
  else{
      for (int32_t l = 0; l < leading; l++) {
      int32_t i_lmt_this = l * i_mt + offset;
      int32_t o_lmt_this = l * o_mt;
      memcpy((int8_t*)Y->dptr_ + o_lmt_this* Y->byte_,
            (int8_t*)X->dptr_ + i_lmt_this * X->byte_,
            o_mt * Y->byte_);
    }
  }
  return T_SUCCESS;
}
#endif
