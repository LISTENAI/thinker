#ifndef _CONCAT_VENUS_H_
#define _CONCAT_VENUS_H_

#include <string.h>

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "luna/luna_math.h"
#include "thinker_status.h"

#define SATURATE_BITS(x) ((x) > 127.0+0.01 ? 127.0+0.01 : ((x) < -128.0-0.01 ? -128.0-0.01 : (x)))

void scale_quant(int8_t *src, int8_t *dst, int32_t size, int8_t scale) {
//   float scalef = (float)(1 << scale);
//   for (int32_t i = 0; i < size; ++i) {
//     dst[i] = (int8_t)SATURATE_8BITS(floorf(scalef * src[i] + 0.5));
//   }
  luna_scale_q7_int8(src, (int)(1<<scale), dst, size, 0);
}

void scale_dequant8bit(int8_t *src, int8_t *dst, int32_t size, int8_t scale) {
//   float scale1 = 1.f / (1 << scale);
//   for (int32_t i = size - 1; i >= 0; --i) {
//     dst[i] = (int8_t)SATURATE_8BITS(floorf(src[i] * scale1 + 0.5));
//   }
  luna_scale_q7_int8(src, 1, dst, size, scale);
}

int32_t concat_luna(tTensor **tensors, int32_t axis, int32_t input_num,
                    tTensor *workspace, tTensor *output) {

  int32_t leading = 1, mid = output->shape_.dims_[axis], trailing = 1;
  int8_t *data_temp = (workspace == NULL) ? NULL : (int8_t *)workspace->dptr_;

  switch (tensors[0]->dtype_)
  {
    case Int8:
	{
      for (int32_t i = 0; i < axis; ++i) {
        leading *= output->shape_.dims_[i];
      }
      for (int32_t i = axis + 1; i < output->shape_.ndim_; ++i) {
        trailing *= output->shape_.dims_[i];
      }

      if (leading == 1) {
        int8_t *output_ptr = (int8_t *)(output->dptr_);
        for (int32_t i = 0; i < input_num; ++i) {
          int8_t *src = (int8_t *)tensors[i]->dptr_;
          int32_t input_scale = tensors[i]->scale_;
          int32_t hw_curr = tensors[i]->shape_.dims_[axis] * trailing * output->byte_;

          if(input_scale != output->scale_){
            if (2 != tensors[i]->mem_.type_){
              memcpy(data_temp, src, hw_curr);
              src = data_temp;
              // data_temp += hw_curr;
            }
            if (2 != output->mem_.type_){
              output_ptr = data_temp + hw_curr * (2 == tensors[i]->mem_.type_);
            }

            if (input_scale < output->scale_){
              int32_t sub_scale_= output->scale_ - input_scale;
              scale_quant(src, output_ptr, hw_curr, sub_scale_);
            }
            else{
              int32_t sub_scale_= input_scale - output->scale_;
              scale_dequant8bit(src, output_ptr, hw_curr, sub_scale_);
            }
            
            if (2 != output->mem_.type_){
              memcpy((int8_t *)output->dptr_, output_ptr, hw_curr);
            }
          }
          else{
            if ((2 == tensors[i]->mem_.type_) && (2 == output->mem_.type_))
              luna_memcpy(output_ptr, src, hw_curr);
            else
              memcpy(output_ptr, src, hw_curr);
          }

          output_ptr += hw_curr;
        }
      } 
      else {
        int32_t hw = mid * trailing * output->byte_;
        for (int32_t l = 0; l < leading; l++) {
          int8_t *output_ptr = (int8_t *)(output->dptr_) + l * hw;
          for (int32_t i = 0; i < input_num; ++i) {
            int32_t hw_curr = tensors[i]->shape_.dims_[axis] * trailing * output->byte_;
            int8_t *indptr_curr = (int8_t *)(tensors[i]->dptr_) + l * hw_curr;
            int32_t input_scale = tensors[i]->scale_;

            if(input_scale != output->scale_){
              if (2 != tensors[i]->mem_.type_){
                memcpy(data_temp, indptr_curr, hw_curr);
                indptr_curr = data_temp;
                // data_temp += hw_curr;
              }
              if (2 != output->mem_.type_){
                output_ptr = data_temp + hw_curr * (2 == tensors[i]->mem_.type_);
              }

              if (input_scale < output->scale_){
                int32_t sub_scale_= output->scale_ - input_scale;
                scale_quant(indptr_curr, output_ptr, hw_curr, sub_scale_);
              }
              else{
                int32_t sub_scale_= input_scale - output->scale_;
                scale_dequant8bit(indptr_curr, output_ptr, hw_curr, sub_scale_);
              }

              if (2 != output->mem_.type_){
                memcpy((int8_t *)output->dptr_ + l * hw + i * hw_curr, output_ptr, hw_curr);
              }
            }
            else{
              if ((2 == tensors[i]->mem_.type_) && (2 == output->mem_.type_))
                luna_memcpy(output_ptr, indptr_curr, hw_curr);
              else
                memcpy(output_ptr, indptr_curr, hw_curr);
            }
              output_ptr += hw_curr;

          }
        }
      }
    }
    break;
    case Int16:
    {
      for (int32_t i = 0; i < axis; ++i) {
        leading *= output->shape_.dims_[i];
      }
      for (int32_t i = axis + 1; i < output->shape_.ndim_; ++i) {
        trailing *= output->shape_.dims_[i];
      }

      if (leading == 1) {
        int16_t *output_ptr = (int16_t *)(output->dptr_);
        for (int32_t i = 0; i < input_num; ++i) {
          int32_t hw_curr =
              tensors[i]->shape_.dims_[axis] * trailing;
          memcpy(output_ptr, (int16_t *)(tensors[i]->dptr_), hw_curr * output->byte_);
          output_ptr += hw_curr;
        }
      } else {
        int32_t hw = mid * trailing;
        for (int32_t l = 0; l < leading; l++) {
          int16_t *output_ptr = (int16_t *)(output->dptr_) + l * hw;
          for (int32_t i = 0; i < input_num; ++i) {
            int32_t hw_curr =
                tensors[i]->shape_.dims_[axis] * trailing;
            int16_t *indptr_curr = (int16_t *)(tensors[i]->dptr_) + l * hw_curr;
            memcpy(output_ptr, indptr_curr, hw_curr * output->byte_);
            output_ptr += hw_curr;
          }
        }
      }
    }
    break;
    case Int32:
    {
      for (int32_t i = 0; i < axis; ++i) {
        leading *= output->shape_.dims_[i];
      }
      for (int32_t i = axis + 1; i < output->shape_.ndim_; ++i) {
        trailing *= output->shape_.dims_[i];
      }

      if (leading == 1) {
        int32_t *output_ptr = (int32_t *)(output->dptr_);
        for (int32_t i = 0; i < input_num; ++i) {
          int32_t hw_curr =
              tensors[i]->shape_.dims_[axis] * trailing;
          memcpy(output_ptr, (int32_t *)(tensors[i]->dptr_), hw_curr * output->byte_);
          output_ptr += hw_curr;
        }
      } else {
        int32_t hw = mid * trailing;
        for (int32_t l = 0; l < leading; l++) {
          int32_t *output_ptr = (int32_t *)(output->dptr_) + l * hw;
          for (int32_t i = 0; i < input_num; ++i) {
            int32_t hw_curr =
                tensors[i]->shape_.dims_[axis] * trailing;
            int32_t *indptr_curr = (int32_t *)(tensors[i]->dptr_) + l * hw_curr;
            memcpy(output_ptr, indptr_curr, hw_curr * output->byte_);
            output_ptr += hw_curr;
          }
        }
      }
    }
    break;
    default:
    break;
  }

  
  return 0;
}
#endif  //_CONCAT_VENUS_H_
