#include <string.h>

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "thinker_status.h"

int32_t concat_luna(tTensor **tensors, int32_t axis, int32_t input_num,
                    tTensor *output) {
  int32_t x1_q = tensors[0]->scale_;
  int32_t x2_q = tensors[1]->scale_;
  int32_t y_q = output->scale_;
  float temp = 0;

  void *src1 = (void *)tensors[0]->dptr_;
  void *src2 = (void *)tensors[1]->dptr_;

  size_t size_x1 = getTensorSize(tensors[0]);
  size_t size_x2 = getTensorSize(tensors[1]);

  int32_t leading = 1, mid = output->shape_.dims_[axis], trailing = 1;

  if (x1_q != x2_q) {
    if (x1_q < x2_q && x1_q == y_q) {
      for (int32_t i = 0; i < size_x2; ++i) {
        // temp = round(*((int8_t*)src2+i) / 2);
        temp = *((int8_t *)src2 + i) / 2.0;
        temp = temp + 0.5;
        if (temp > 127) {
          temp = 127;
        }
        if (temp < -128) {
          temp = -128;
        }
        *((int8_t *)src2 + i) = temp;
      }
    } else if (x1_q > x2_q && x1_q == y_q) {
      for (int32_t i = 0; i < size_x2; ++i) {
        temp = *((int8_t *)src2 + i) * 2;
        if (temp > 127) {
          temp = 127;
        }
        if (temp < -128) {
          temp = -128;
        }
        *((int8_t *)src2 + i) = temp;
      }
    }

    else if (x1_q < x2_q && x2_q == y_q) {
      for (int32_t i = 0; i < size_x1; ++i) {
        temp = *((int8_t *)src1 + i) * 2;
        if (temp > 127) {
          temp = 127;
        }
        if (temp < -128) {
          temp = -128;
        }
        *((int8_t *)src1 + i) = temp;
      }
    }

    else if (x1_q > x2_q && x2_q == y_q) {
      for (int32_t i = 0; i < size_x1; ++i) {
        temp = *((int8_t *)src1 + i) / 2.0;
        temp = temp + 0.5;
        if (temp > 127) {
          temp = 127;
        }
        if (temp < -128) {
          temp = -128;
        }
        *((int8_t *)src1 + i) = temp;
      }
    }
  }

  for (int32_t i = 0; i < axis; ++i) {
    leading *= output->shape_.dims_[i];
  }
  for (int32_t i = axis + 1; i < output->shape_.ndim_; ++i) {
    trailing *= output->shape_.dims_[i];
  }

  if (leading == 1) {
    int8_t *output_ptr = (int8_t *)(output->dptr_);
    for (int32_t i = 0; i < input_num; ++i) {
      int32_t hw_curr =
          tensors[i]->shape_.dims_[axis] * trailing * output->byte_;
      memcpy(output_ptr, (int8_t *)(tensors[i]->dptr_), hw_curr);
      output_ptr += hw_curr;
    }
  } else {
    int32_t hw = mid * trailing;
    for (int32_t l = 0; l < leading; l++) {
      int8_t *output_ptr = (int8_t *)(output->dptr_) + l * hw * output->byte_;
      for (int32_t i = 0; i < input_num; ++i) {
        int32_t hw_curr =
            tensors[i]->shape_.dims_[axis] * trailing * output->byte_;
        int8_t *indptr_curr = (int8_t *)(tensors[i]->dptr_) + l * hw_curr;
        memcpy(output_ptr, indptr_curr, hw_curr);
        output_ptr += hw_curr;
      }
    }
  }
  return 0;
}
