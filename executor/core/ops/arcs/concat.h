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

void scale_requant8bit_cpu(int8_t *src, int8_t *dst, int32_t size, int8_t scale)
{
  int32_t i = 0;
  if (scale > 0)
  {
    for (i = 0; i < size; i++)
      dst[i] = SATURATE_8BITS(src[i] << scale);
  }
  else
  {
    for (i = 0; i < size; i++)
      dst[i] = floor(src[i] * pow(2, scale) + 0.5);
  }
}

void scale_requant16bit_cpu(int16_t *src, int16_t *dst, int32_t size, int8_t scale)
{
  int32_t i = 0;
  if (scale > 0)
  {
    for (i = 0; i < size; i++)
      dst[i] = SATURATE_16BITS(src[i] << scale);
  }
  else
  {
    for (i = 0; i < size; i++)
      dst[i] = floor(src[i] * pow(2, scale) + 0.5);
  }
}

void scale_requant32bit_cpu(int32_t *src, int32_t *dst, int32_t size, int8_t scale)
{
  int32_t i = 0;
  if (scale > 0)
  {
    for (i = 0; i < size; i++)
      dst[i] = src[i] << scale;
  }
  else
  {
    for (i = 0; i < size; i++)
      dst[i] = SATURATE_8BITS(floor(src[i] * pow(2, scale) + 0.5));
  }
}

int32_t concat_luna(tTensor **tensors, int32_t axis, int32_t input_num, tTensor *workspace, tTensor *output) {
  int32_t ret = T_ERR_NO_IMPLEMENTED;

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
            ret = API_LIB(memcpy_i8o8)(dst, src, hw_curr);
          } 
          else {
            opi_psram_cpy_out(dst, src, hw_curr);
            ret = T_SUCCESS;
          }
        } 
        else {
          if (2 == tensors[i]->mem_.type_ && 2 == output->mem_.type_) { // both input and output on share
            if (input_scale < output_scale) {
              ret = API_LIB(scale_i8i8o8)(src, 1UL<<(output_scale - input_scale), dst, hw_curr, 0);
            }
            else {
              ret = API_LIB(scale_i8i8o8)(src, 1, dst, hw_curr, (input_scale - output_scale));
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

              ret = API_LIB(memcpy_i8o8)(tmp_ptr, src + past_size, cur_size);

              if (input_scale < output_scale) {
                ret = API_LIB(scale_i8i8o8)(tmp_ptr, 1UL<<(output_scale - input_scale), dst + past_size, cur_size, 0);
              }
              else {
                ret = API_LIB(scale_i8i8o8)(tmp_ptr, 1, dst + past_size, cur_size, (input_scale - output_scale));
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
                ret = API_LIB(scale_i8i8o8)(src + past_size, 1UL<<(output_scale - input_scale), tmp_ptr, cur_size, 0);
              }
              else {
                ret = API_LIB(scale_i8i8o8)(src + past_size, 1, tmp_ptr, cur_size, (input_scale - output_scale));
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

              ret = API_LIB(memcpy_i8o8)(tmp_ptr, src + past_size, cur_size);

              if (input_scale < output_scale) {
                ret = API_LIB(scale_i8i8o8)(tmp_ptr, 1UL<<(output_scale - input_scale), tmp_ptr, cur_size, 0);
              }
              else {
                ret = API_LIB(scale_i8i8o8)(tmp_ptr, 1, tmp_ptr, cur_size, (input_scale - output_scale));
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
              ret = API_LIB(mat_copy_i8o8)(src, dst, leading, tensors[i]->shape_.dims_[axis], trailing, trailing * tensors[i]->shape_.dims_[axis], trailing, trailing*middle, trailing);
            else                                   // trailing为1时，可转换为2维矩阵
              ret = API_LIB(mat_copy_i8o8)(src, dst, 1, leading, tensors[i]->shape_.dims_[axis], leading * tensors[i]->shape_.dims_[axis], tensors[i]->shape_.dims_[axis], leading * middle, middle);
          } 
          else {
            for (int32_t l = 0; l < leading; l++) 
            {
              int8_t *indptr_curr = (int8_t *)src + l * hw_curr;
              int8_t *output_ptr  = (int8_t *)dst + l * hw;
              if (2 == output->mem_.type_) {
                ret = API_LIB(memcpy_i8o8)(output_ptr, indptr_curr, trailing * tensors[i]->shape_.dims_[axis]);
              }
              else {
                opi_psram_cpy_out(output_ptr, indptr_curr, trailing * tensors[i]->shape_.dims_[axis]);
                ret = T_SUCCESS;
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
                ret = API_LIB(memcpy_i8o8)(src_ptr, src + l * hw_curr, hw_curr);
              }

              ret = API_LIB(scale_i8i8o8)(src_ptr, scalar, dst + l * hw, hw_curr, shift);
            }
          } 
          else { // output on psram
            int8_t *tmp = (int8_t *)workspace->dptr_;
            for (int32_t l = 0; l < leading; l++) {
              int8_t *src_ptr = src + l * hw_curr;

              if (2 != tensors[i]->mem_.type_) {
                src_ptr = tmp;
                ret = API_LIB(memcpy_i8o8)(src_ptr, src + l * hw_curr, hw_curr);
              }

              ret = API_LIB(scale_i8i8o8)(src_ptr, scalar, tmp, hw_curr, shift);
              opi_psram_cpy_out(dst + l * hw, tmp, hw_curr);
            }
          }
          dst  += hw_curr;
        }
      }
    }
  }
  else if (Int16 == output->dtype_) {
    int16_t *dst = (int16_t *)output->dptr_;
    if (leading == 1) {    // 最外层维度进行拼接
      for (int32_t i = 0; i < input_num; ++i)  // 支持多个输入
      {
        if (Int16 != tensors[i]->dtype_)
          return T_ERR_INVALID_DATATYPE;

        int16_t *src 		= (int16_t *)tensors[i]->dptr_;
        int32_t input_scale = tensors[i]->scale_;
        int32_t hw_curr 	= tensors[i]->shape_.dims_[axis] * trailing;
        if (0 == hw_curr)
          continue;

        if (input_scale != output_scale) {
          scale_requant16bit_cpu(src, dst + i * hw_curr * 2, hw_curr, output_scale-input_scale);
          ret = T_SUCCESS;
        }
        else {
        	if (2 == output->mem_.type_) {
        		ret = API_LIB(memcpy_i8o8)((int8_t *)dst + i * hw_curr * 2, (int8_t *)src, hw_curr * 2);
        	}
        	else {
        		opi_psram_cpy_out((int8_t *)dst + i * hw_curr * 2, (int8_t *)src, hw_curr * 2);
        		ret = T_SUCCESS;
        	}
        }
      }
    }
    else {                // 中间或者最里层维度拼接
      for (int32_t i = 0; i < input_num; ++i) { // 支持多个输入
        if (Int16 != tensors[i]->dtype_)
          return T_ERR_INVALID_DATATYPE;

        int16_t *src        = (int16_t *)tensors[i]->dptr_;
        int32_t input_scale = tensors[i]->scale_;
        int32_t hw_curr     = tensors[i]->shape_.dims_[axis] * trailing;
        if (0 == hw_curr)
          continue;

        if (input_scale != output_scale) {   // 输入输出scale值不一致
          for (int32_t l = 0; l < leading; l++) {
            int16_t *indptr_curr = (int16_t *)src + l * hw_curr;
            int16_t *output_ptr  = (int16_t *)dst + l * hw + i * hw_curr;

            scale_requant16bit_cpu(indptr_curr, output_ptr, hw_curr, output_scale-input_scale);
            ret = T_SUCCESS;
          }
        }
        else { 
          for (int32_t l = 0; l < leading; l++) {
            int16_t *indptr_curr = (int16_t *)src + l * hw_curr;
            int16_t *output_ptr  = (int16_t *)dst + l * hw + i * hw_curr;
            if (2 == output->mem_.type_) {
            	ret = API_LIB(memcpy_i8o8)((int8_t *)output_ptr, (int8_t *)indptr_curr, hw_curr * 2);
            }
            else {
            	opi_psram_cpy_out((int8_t *)output_ptr, (int8_t *)indptr_curr, hw_curr * 2);
            	ret = T_SUCCESS;
            }
          }
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
			if (Int32 != tensors[i]->dtype_)
			  return T_ERR_INVALID_DATATYPE;

			int32_t *src 		= (int32_t *)tensors[i]->dptr_;
			int32_t input_scale = tensors[i]->scale_;
			int32_t hw_curr 	= tensors[i]->shape_.dims_[axis] * trailing;
			if (0 == hw_curr)
			  continue;

			if (input_scale == output_scale) {
				ret = API_LIB(memcpy_i8o8)((int8_t *)dst, (int8_t *)src, hw_curr * 4);
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
			if (Int32 != tensors[i]->dtype_)
			  return T_ERR_INVALID_DATATYPE;

			int32_t *src 		= (int32_t *)tensors[i]->dptr_;
			int32_t input_scale = tensors[i]->scale_;
			int32_t hw_curr 	= tensors[i]->shape_.dims_[axis] * trailing;
			if (0 == hw_curr)
			  continue;

			if (input_scale == output_scale) {
				opi_psram_cpy_out(dst, src, hw_curr * 4);
				dst += hw_curr;
				ret = T_SUCCESS;
			}
			else {
			  return T_ERR_INVALID_PARA;
			}
		  }
      }
    }
    else {                // 中间或者最里层维度拼接
      if (2 == output->mem_.type_) {
		  for (int32_t i = 0; i < input_num; ++i) { // 支持多个输入
			if (Int32 != tensors[i]->dtype_)
			  return T_ERR_INVALID_DATATYPE;

			int32_t *src        = (int32_t *)tensors[i]->dptr_;
			int32_t input_scale = tensors[i]->scale_;
			int32_t hw_curr     = tensors[i]->shape_.dims_[axis] * trailing;
			if (0 == hw_curr)
			  continue;

			if (input_scale == output_scale) {
			  for (int32_t l = 0; l < leading; l++) {
				int32_t *indptr_curr = (int32_t *)src + l * hw_curr;
				int32_t *output_ptr  = (int32_t *)dst + l * hw + i * hw_curr;
				ret = API_LIB(memcpy_i8o8)((int8_t *)output_ptr, (int8_t *)indptr_curr, hw_curr * 4);
			  }
			}
			else {
			  return T_ERR_INVALID_DATATYPE;
			}
		  }
      }
      else {
		  for (int32_t i = 0; i < input_num; ++i) { // 支持多个输入
			if (Int32 != tensors[i]->dtype_)
			  return T_ERR_INVALID_DATATYPE;

			int32_t *src        = (int32_t *)tensors[i]->dptr_;
			int32_t input_scale = tensors[i]->scale_;
			int32_t hw_curr     = tensors[i]->shape_.dims_[axis] * trailing;
			if (0 == hw_curr)
			  continue;

			if (input_scale == output_scale) {
			  for (int32_t l = 0; l < leading; l++) {
				int32_t *indptr_curr = (int32_t *)src + l * hw_curr;
				int32_t *output_ptr  = (int32_t *)dst + l * hw + i * hw_curr;
				opi_psram_cpy_out((int8_t *)output_ptr, (int8_t *)indptr_curr, hw_curr * 4);
				ret = T_SUCCESS;
			  }
			}
			else {
			  return T_ERR_INVALID_DATATYPE;
			}
		  }
      }
    }
  }
  else {
    ret = T_ERR_INVALID_DATATYPE;
  }
  return ret;
}
#endif
