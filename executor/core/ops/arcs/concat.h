#ifndef _CONCAT_VENUS_H_
#define _CONCAT_VENUS_H_

#include <string.h>

#include "core/comm/utils.h"
#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif
#include "thinker_status.h"

int32_t concat_luna(tTensor **tensors, int32_t axis, int32_t input_num, tTensor *workspace, tTensor *output) {
#if THINKER_PARAM_CHECK
  if (output->dtype_ != Int8 && output->dtype_ != Int32) {
      return (T_ERR_INVALID_DATATYPE);
  }

  for (int32_t i = 0; i < input_num; ++i) {
    if (tensors[i]->dtype_ != output->dtype_) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (output->dtype_ == Int32 &&
                        tensors[i]->scale_ != output->scale_) {
        return (T_ERR_INVALID_PARA);
    }
  }
#endif
  int32_t leading = 1, middle = 1, trailing = 1;
  for (int32_t i = 0; i < axis; ++i) 
  {
    leading *= output->shape_.dims_[i];
  }
  middle   = output->shape_.dims_[axis];
  for (int32_t i = axis + 1; i < output->shape_.ndim_; ++i) 
  {
    trailing *= output->shape_.dims_[i];
  }
  int32_t hw = middle * trailing;

  int8_t *dst = (int8_t *)output->dptr_;
  int32_t output_scale = output->scale_;
  if (Int8 == output->dtype_) {
    int32_t required_workspace = 0;
    for (int32_t i = 0; i < input_num; ++i) {
      int32_t scale_shift = tensors[i]->scale_ - output_scale;
#if THINKER_PARAM_CHECK
      if (scale_shift < -6 || scale_shift > 63) {
          return (T_ERR_INVALID_PARA);
      }
#endif
      if (tensors[i]->scale_ != output_scale &&
          (tensors[i]->mem_.type_ != 2 || output->mem_.type_ != 2)) {
        int32_t size = tensors[i]->shape_.dims_[axis] * trailing;
        if (size > required_workspace) required_workspace = size;
      }
    }
    if (required_workspace > 65536) required_workspace = 65536;
#if THINKER_RUNTIME_CHECK
    if (required_workspace != 0 &&
                          (workspace == NULL || workspace->dptr_ == 0 ||
                           workspace->shape_.dims_[0] < required_workspace)) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif
  }

  if (Int8 == output->dtype_) {
    int8_t *dst = (int8_t *)output->dptr_;
    if (leading == 1) {    // 最外层维度进行拼接
      for (int32_t i = 0; i < input_num; ++i)  // 支持多个输入
      {
        if (Int8 != tensors[i]->dtype_) return T_ERR_INVALID_DATATYPE;
          
        int8_t *src         = (int8_t *)tensors[i]->dptr_;
        int32_t input_scale = tensors[i]->scale_;
        int32_t hw_curr 	  = tensors[i]->shape_.dims_[axis] * trailing;

        if (0 == hw_curr)
          continue;

        if (input_scale == output_scale) {
          if (2 == output->mem_.type_) {
            THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(dst, src, hw_curr), "luna_memcpy_i8o8");
          } 
          else {
            opi_psram_cpy_out(dst, src, hw_curr);
          }
        } 
        else {
          if (2 == tensors[i]->mem_.type_ && 2 == output->mem_.type_) { // both input and output on share
            if (input_scale < output_scale) {
              THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(src, 1UL<<(output_scale - input_scale), dst, hw_curr, 0), "luna_scale_i8i8o8");
            }
            else {
              THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(src, 1, dst, hw_curr, (input_scale - output_scale)), "luna_scale_i8i8o8");
            }
          }
          else if (2 != tensors[i]->mem_.type_ && 2 == output->mem_.type_) { // input on psram, output on share
            int32_t workspace_size = workspace->shape_.dims_[0];
            int8_t *tmp_ptr = (int8_t *)workspace->dptr_;

            int past_size = 0;

            while (past_size < hw_curr)
            {
              int32_t remain_size = hw_curr - past_size;
              int32_t cur_size = (workspace_size < remain_size)? workspace_size : remain_size; 

              THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(tmp_ptr, src + past_size, cur_size), "luna_memcpy_i8i8o8");

              if (input_scale < output_scale) {
                THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(tmp_ptr, 1UL<<(output_scale - input_scale), dst + past_size, cur_size, 0), "luna_scale_i8i8o8");
              }
              else {
                THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(tmp_ptr, 1, dst + past_size, cur_size, (input_scale - output_scale)), "luna_scale_i8i8o8");
              }

              past_size += cur_size;
            }
          }
          else if (2 == tensors[i]->mem_.type_ && 2 != output->mem_.type_) { // input on share, output on psram
            int32_t workspace_size = workspace->shape_.dims_[0];
            int8_t *tmp_ptr = (int8_t *)workspace->dptr_;

            int past_size = 0;

            while (past_size < hw_curr)
            {
              int32_t remain_size = hw_curr - past_size;
              int32_t cur_size = (workspace_size < remain_size)? workspace_size : remain_size; 

              if (input_scale < output_scale) {
                THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(src + past_size, 1UL<<(output_scale - input_scale), tmp_ptr, cur_size, 0), "luna_scale_i8i8o8");
              }
              else {
                THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(src + past_size, 1, tmp_ptr, cur_size, (input_scale - output_scale)), "luna_scale_i8i8o8");
              }

              opi_psram_cpy_out(dst + past_size, tmp_ptr, cur_size);
              past_size += cur_size;
            }
          }
          else { // both input and output on psram
            int32_t workspace_size = workspace->shape_.dims_[0];
            int8_t *tmp_ptr = (int8_t *)workspace->dptr_;

            int past_size = 0;

            while (past_size < hw_curr)
            {
              int32_t remain_size = hw_curr - past_size;
              int32_t cur_size = (workspace_size < remain_size)? workspace_size : remain_size; 

              THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(tmp_ptr, src + past_size, cur_size), "luna_memcpy_i8o8");

              if (input_scale < output_scale) {
                THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(tmp_ptr, 1UL<<(output_scale - input_scale), tmp_ptr, cur_size, 0), "luna_scale_i8i8o8");
              }
              else {
                THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(tmp_ptr, 1, tmp_ptr, cur_size, (input_scale - output_scale)), "luna_scale_i8i8o8");
              }

              opi_psram_cpy_out(dst + past_size, tmp_ptr, cur_size);
              past_size += cur_size;
            }
          }
        }
        dst += hw_curr;
      }
    }
    else {                // 中间或者最里层维度拼接
      for (int32_t i = 0; i < input_num; ++i) { // 支持多个输入
        if (Int8 != tensors[i]->dtype_) return T_ERR_INVALID_DATATYPE;

        int8_t *src         = (int8_t *)tensors[i]->dptr_;
        int32_t input_scale = tensors[i]->scale_;
        int32_t hw_curr     = tensors[i]->shape_.dims_[axis] * trailing;

        if (0 == hw_curr)
          continue;

        if (input_scale == output_scale) {
          if (2 == tensors[i]->mem_.type_ && 2 == output->mem_.type_) {
            if (trailing != 1)                     // 3维矩阵拼接
              THINKER_RET_CHECK(API_LIB(mat_copy_i8o8)(src, dst, leading, tensors[i]->shape_.dims_[axis], trailing, trailing * tensors[i]->shape_.dims_[axis], trailing, trailing*middle, trailing), "luna_mat_copy_i8o8");
            else                                   // trailing为1时，可转换为2维矩阵
              THINKER_RET_CHECK(API_LIB(mat_copy_i8o8)(src, dst, 1, leading, tensors[i]->shape_.dims_[axis], leading * tensors[i]->shape_.dims_[axis], tensors[i]->shape_.dims_[axis], leading * middle, middle), "luna_mat_copy_i8o8");
          } 
          else {
            for (int32_t l = 0; l < leading; l++) 
            {
              int8_t *indptr_curr = (int8_t *)src + l * hw_curr;
              int8_t *output_ptr  = (int8_t *)dst + l * hw;
              if (2 == output->mem_.type_) {
                THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(output_ptr, indptr_curr, trailing * tensors[i]->shape_.dims_[axis]), "luna_memcpy_i8o8");
              }
              else {
                opi_psram_cpy_out(output_ptr, indptr_curr, trailing * tensors[i]->shape_.dims_[axis]);
              }
            }
          }
          dst += hw_curr;
        } 
        else {
          int8_t scalar = input_scale < output_scale? 1UL<<(output_scale - input_scale) : 1;
          int8_t shift = input_scale < output_scale? 0 : input_scale - output_scale;
          if (2 == output->mem_.type_) {  // output on share
            for (int32_t l = 0; l < leading; l++) {
              int8_t *src_ptr = src + l * hw_curr;

              if (2 != tensors[i]->mem_.type_) {
                src_ptr = (int8_t *)workspace->dptr_;
                THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(src_ptr, src + l * hw_curr, hw_curr), "luna_memcpy_i8o8");
              }

              THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(src_ptr, scalar, dst + l * hw, hw_curr, shift), "luna_scale_i8i8o8");
            }
          } 
          else { // output on psram
            int8_t *tmp = (int8_t *)workspace->dptr_;
            for (int32_t l = 0; l < leading; l++) {
              int8_t *src_ptr = src + l * hw_curr;

              if (2 != tensors[i]->mem_.type_) {
                src_ptr = tmp;
                THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(src_ptr, src + l * hw_curr, hw_curr), "luna_memcpy_i8o8");
              }

              THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(src_ptr, scalar, tmp, hw_curr, shift), "luna_scale_i8i8o8");
              opi_psram_cpy_out(dst + l * hw, tmp, hw_curr);
            }
          }
          dst  += hw_curr;
        }
      }
    }
  }
  else if (Int32 == output->dtype_) {
    int32_t *dst = (int32_t *)output->dptr_;
    if (leading == 1) {    // 最外层维度进行拼接
      if (2 == output->mem_.type_) {
        for (int32_t i = 0; i < input_num; ++i)  // 支持多个输入
        {
#if THINKER_PARAM_CHECK
          if (Int32 != tensors[i]->dtype_) {
              return (T_ERR_INVALID_DATATYPE);
          }
#endif

          int32_t *src 		= (int32_t *)tensors[i]->dptr_;
          int32_t input_scale = tensors[i]->scale_;
          int32_t hw_curr 	= tensors[i]->shape_.dims_[axis] * trailing;
          if (0 == hw_curr)
            continue;

          if (input_scale == output_scale) {
            THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)dst, (int8_t *)src, hw_curr * 4), "luna_memcpy_i8o8");
            dst += hw_curr;
          }
          else {
            return T_ERR_INVALID_PARA;
          }
        }
      }
      else {
        for (int32_t i = 0; i < input_num; ++i)  // 支持多个输入
        {
#if THINKER_PARAM_CHECK
          if (Int32 != tensors[i]->dtype_) {
              return (T_ERR_INVALID_DATATYPE);
          }
#endif

          int32_t *src 		= (int32_t *)tensors[i]->dptr_;
          int32_t input_scale = tensors[i]->scale_;
          int32_t hw_curr 	= tensors[i]->shape_.dims_[axis] * trailing;
          if (0 == hw_curr)
            continue;

          if (input_scale == output_scale) {
            opi_psram_cpy_out(dst, src, hw_curr * 4);
            dst += hw_curr;
          }
          else {
            return T_ERR_INVALID_PARA;
          }
        }
      }
    }
    else {                // 中间或者最里层维度拼接
      int32_t output_offset = 0;
      if (2 == output->mem_.type_) {
        for (int32_t i = 0; i < input_num; ++i) { // 支持多个输入
#if THINKER_PARAM_CHECK
          if (Int32 != tensors[i]->dtype_) {
              return (T_ERR_INVALID_DATATYPE);
          }
#endif

          int32_t *src        = (int32_t *)tensors[i]->dptr_;
          int32_t input_scale = tensors[i]->scale_;
          int32_t hw_curr     = tensors[i]->shape_.dims_[axis] * trailing;
          if (0 == hw_curr)
            continue;

          if (input_scale == output_scale) {
            for (int32_t l = 0; l < leading; l++) {
            int32_t *indptr_curr = (int32_t *)src + l * hw_curr;
            int32_t *output_ptr  = (int32_t *)dst + l * hw + output_offset;
            THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)output_ptr, (int8_t *)indptr_curr, hw_curr * 4), "luna_memcpy_i8o8");
            }
          }
          else {
            return T_ERR_INVALID_DATATYPE;
          }
          output_offset += hw_curr;
        }
      }
      else {
        for (int32_t i = 0; i < input_num; ++i) { // 支持多个输入
#if THINKER_PARAM_CHECK
          if (Int32 != tensors[i]->dtype_) {
              return (T_ERR_INVALID_DATATYPE);
          }
#endif

          int32_t *src        = (int32_t *)tensors[i]->dptr_;
          int32_t input_scale = tensors[i]->scale_;
          int32_t hw_curr     = tensors[i]->shape_.dims_[axis] * trailing;
          if (0 == hw_curr)
            continue;

          if (input_scale == output_scale) {
            for (int32_t l = 0; l < leading; l++) {
            int32_t *indptr_curr = (int32_t *)src + l * hw_curr;
            int32_t *output_ptr  = (int32_t *)dst + l * hw + output_offset;
            opi_psram_cpy_out((int8_t *)output_ptr, (int8_t *)indptr_curr, hw_curr * 4);
            }
          }
          else {
            return T_ERR_INVALID_DATATYPE;
          }
          output_offset += hw_curr;
        }
      }
    }
  }
  else {
    return T_ERR_INVALID_DATATYPE;
  }

#if !(defined(WIN32) || defined(linux))
  if (2 != output->mem_.type_)
      HAL_FlushInvalidateDCache_by_Addr((uint32_t *)output->dptr_,
                                        leading * middle * trailing * output->byte_);
#endif

  return T_SUCCESS;
}
#endif
