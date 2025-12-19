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

/**
 * @brief Scale and requantize 16-bit integer data
 * @param src Input data pointer
 * @param dst Output data pointer
 * @param size Number of elements to process
 * @param scale Scale factor (positive for left shift, negative for right shift)
 */
void scale_requant16bit_cpu(int16_t *src, int16_t *dst, int32_t size, int8_t scale) {
    if (scale > 0) {
        for (int32_t i = 0; i < size; ++i) {
            dst[i] = SATURATE_16BITS(src[i] << scale);
        }
    } else {
        for (int32_t i = 0; i < size; ++i) {
            dst[i] = floor(src[i] * pow(2, scale) + 0.5);
        }
    }
}

/**
 * @brief Concatenate multiple tensors along a specified axis
 * @param tensors Array of input tensors
 * @param axis Axis along which to concatenate
 * @param input_num Number of input tensors
 * @param workspace Temporary workspace tensor
 * @param output Output tensor
 * @return int32_t Operation status
 */
int32_t concat_luna(tTensor **tensors, int32_t axis, int32_t input_num, tTensor *workspace, tTensor *output) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    // Calculate dimensions
    int32_t leading = 1, middle = 1, trailing = 1;
    for (int32_t i = 0; i < axis; ++i) {
        leading *= output->shape_.dims_[i];
    }
    middle = output->shape_.dims_[axis];
    for (int32_t i = axis + 1; i < output->shape_.ndim_; ++i) {
        trailing *= output->shape_.dims_[i];
    }
    int32_t hw = middle * trailing;

    // Process based on output data type
    if (output->dtype_ == Int8) {
        int8_t *dst = (int8_t *)output->dptr_;
        int32_t output_scale = output->scale_;
        int32_t workspace_size = workspace ? workspace->shape_.dims_[0] : 0;
        int8_t *dst_ptr = workspace ? (int8_t *)workspace->dptr_ : NULL;
        if (leading == 1) {
            for (int32_t i = 0; i < input_num; ++i) {
                if (tensors[i]->dtype_ != Int8) {
                    return T_ERR_INVALID_DATATYPE;
                }

                int8_t *src = (int8_t *)tensors[i]->dptr_;
                int32_t input_scale = tensors[i]->scale_;
                int32_t hw_curr = tensors[i]->shape_.dims_[axis] * trailing;

                if (hw_curr == 0) {
                    continue;
                }

                if (input_scale == output_scale) {
                    if (output->mem_.type_ == 2) {
                        ret = API_LIB(memcpy_i8o8)(dst, src, hw_curr);
                    } 
                    else {
                        opi_psram_cpy_out(dst, src, hw_curr);
                        ret = T_SUCCESS;
                    }
                } 
                else {
                    if (output->mem_.type_ == 2) {
                        // just support (output_scale >= input_scale)
                        ret = API_LIB(scale_i8i8o8)(src, 1UL << (output_scale - input_scale), dst, hw_curr, 0);
                    } 
                    else {
                        int32_t past_size = 0;
                        while (past_size < hw_curr) {
                            int32_t remain_size = hw_curr - past_size;
                            int32_t cur_size = (workspace_size < remain_size) ? workspace_size : remain_size;

                            // just support (output_scale >= input_scale)
                            ret = API_LIB(scale_i8i8o8)(src + past_size, 1UL << (output_scale - input_scale), dst_ptr, cur_size, 0);

                            opi_psram_cpy_out(dst + past_size, dst_ptr, cur_size);
                            past_size += cur_size;
                        }
                    }
                }
                dst += hw_curr;
            }
        } 
        else {
            for (int32_t i = 0; i < input_num; ++i) {
                if (tensors[i]->dtype_ != Int8) {
                    return T_ERR_INVALID_DATATYPE;
                }

                int8_t *src = (int8_t *)tensors[i]->dptr_;
                int32_t input_scale = tensors[i]->scale_;
                int32_t hw_curr = tensors[i]->shape_.dims_[axis] * trailing;

                if (hw_curr == 0) {
                    continue;
                }

                if (input_scale == output_scale) {
                    if (tensors[i]->mem_.type_ == 2 && output->mem_.type_ == 2) {
                        if (trailing != 1) {
                            ret = API_LIB(mat_copy_i8o8)(src, dst, leading, tensors[i]->shape_.dims_[axis], trailing, trailing * tensors[i]->shape_.dims_[axis], trailing, trailing * middle, trailing);
                        } else {
                            ret = API_LIB(mat_copy_i8o8)(src, dst, 1, leading, tensors[i]->shape_.dims_[axis], leading * tensors[i]->shape_.dims_[axis], tensors[i]->shape_.dims_[axis], leading * middle, middle);
                        }
                    } 
                    else {
                        for (int32_t l = 0; l < leading; ++l) {
                            int8_t *indptr_curr = (int8_t *)src + l * hw_curr;
                            int8_t *output_ptr = (int8_t *)dst + l * hw;
                            if (output->mem_.type_ == 2) {
                                ret = API_LIB(memcpy_i8o8)(output_ptr, indptr_curr, trailing * tensors[i]->shape_.dims_[axis]);
                            } else {
                                opi_psram_cpy_out(output_ptr, indptr_curr, trailing * tensors[i]->shape_.dims_[axis]);
                                ret = T_SUCCESS;
                            }
                        }
                    }
                } 
                else {
                    if (output->mem_.type_ == 2) {
                        // just support output_scale >= input_scale
                        for (int32_t l = 0; l < leading; ++l) {
                            ret = API_LIB(scale_i8i8o8)(src + l * hw_curr, 1UL << (output_scale - input_scale), dst + l * hw, hw_curr, 0);
                        }
                    } 
                    else {
                        // just support output_scale >= input_scale
                        if (workspace_size < hw_curr) return T_ERR_NO_WORKSPACE;
                        for (int32_t l = 0; l < leading; ++l) {
                            ret = API_LIB(scale_i8i8o8)(src + l * hw_curr, 1UL << (output_scale - input_scale), dst_ptr, hw_curr, 0);
                            opi_psram_cpy_out(dst + l * hw, dst_ptr, hw_curr);
                        }
                    }
                }
                dst += hw_curr;
            }
        }
    } 
    else if (output->dtype_ == Int16) {
        int16_t *dst = (int16_t *)output->dptr_;
        int32_t output_scale = output->scale_;

        if (leading == 1) {
            for (int32_t i = 0; i < input_num; ++i) {
                if (tensors[i]->dtype_ != Int16) {
                    return T_ERR_INVALID_DATATYPE;
                }

                int16_t *src = (int16_t *)tensors[i]->dptr_;
                int32_t input_scale = tensors[i]->scale_;
                int32_t hw_curr = tensors[i]->shape_.dims_[axis] * trailing;

                if (hw_curr == 0) {
                    continue;
                }

                if (input_scale == output_scale) {
                    if (output->mem_.type_ == 2) {
                        ret = API_LIB(memcpy_i8o8)((int8_t *)dst + i * hw_curr * 2, (int8_t *)src, hw_curr * 2);
                    } else {
                        opi_psram_cpy_out((int8_t *)dst + i * hw_curr * 2, (int8_t *)src, hw_curr * 2);
                        ret = T_SUCCESS;
                    }
                }
                else {
                  if (2 == output->mem_.type_) {
                      // just support (output_scale > input_scale)
                      ret = API_LIB(scale_i16i16o16)(src, 1UL<<(output_scale - input_scale), dst, hw_curr, 0);

                  } 
                  else { // output on psram
                    int32_t workspace_size = workspace->shape_.dims_[0] >> 2;
                    int16_t *dst_ptr = (int16_t *)workspace->dptr_;

                    int32_t past_size = 0;
                    while (past_size < hw_curr)
                    {
                      int32_t remain_size = hw_curr - past_size;
                      int32_t cur_size = (workspace_size < remain_size)? workspace_size : remain_size; 

                      ret = API_LIB(scale_i16i16o16)(src + past_size, 1UL<<(output_scale - input_scale), dst_ptr, cur_size, 0);
                      opi_psram_cpy_out(dst + past_size, dst_ptr, cur_size * sizeof(int16_t));
                      past_size += cur_size;
                    }
                  }
                }
            }
        }
        else {
            for (int32_t i = 0; i < input_num; ++i) {
                if (tensors[i]->dtype_ != Int16) {
                    return T_ERR_INVALID_DATATYPE;
                }

                int16_t *src = (int16_t *)tensors[i]->dptr_;
                int32_t input_scale = tensors[i]->scale_;
                int32_t hw_curr = tensors[i]->shape_.dims_[axis] * trailing;

                if (hw_curr == 0) {
                    continue;
                }

                if (input_scale == output_scale) {
                    for (int32_t l = 0; l < leading; ++l) {
                        int16_t *indptr_curr = (int16_t *)src + l * hw_curr;
                        int16_t *output_ptr = (int16_t *)dst + l * hw + i * hw_curr;
                        if (output->mem_.type_ == 2) {
                            ret = API_LIB(memcpy_i8o8)((int8_t *)output_ptr, (int8_t *)indptr_curr, hw_curr * 2);
                        } 
                        else {
                            opi_psram_cpy_out((int8_t *)output_ptr, (int8_t *)indptr_curr, hw_curr * 2);
                            ret = T_SUCCESS;
                        }
                    }
                }
                else {
                    if (2 == output->mem_.type_) {
                        // just support output_scale >= input_scale
                        ret = API_LIB(scale_i16i16o16)(src, 1UL<<(output_scale - input_scale), dst, hw_curr, 0);
                    } 
                    else { // output on psram
                        int32_t workspace_size = workspace ? (workspace->shape_.dims_[0] >> 2) : 0;
                        int16_t *dst_ptr = workspace ? (int16_t *)workspace->dptr_ : NULL;

                        int past_size = 0;
                        
                        while (past_size < hw_curr)
                        {
                          int32_t remain_size = hw_curr - past_size;
                          int32_t cur_size = (workspace_size < remain_size)? workspace_size : remain_size; 

                          ret = API_LIB(scale_i16i16o16)(src + past_size, 1UL<<(output_scale - input_scale), dst_ptr, cur_size, 0);
                          opi_psram_cpy_out(dst + past_size, dst_ptr, cur_size * sizeof(int16_t));
                          past_size += cur_size;
                        }
                      }
                }
            }
        }
    } else if (output->dtype_ == Int32) {
        int32_t *dst = (int32_t *)output->dptr_;
        int32_t output_scale = output->scale_;

        if (leading == 1) {
            if (output->mem_.type_ == 2) {
                for (int32_t i = 0; i < input_num; ++i) {
                    if (tensors[i]->dtype_ != Int32) {
                        return T_ERR_INVALID_DATATYPE;
                    }

                    int32_t *src = (int32_t *)tensors[i]->dptr_;
                    int32_t input_scale = tensors[i]->scale_;
                    int32_t hw_curr = tensors[i]->shape_.dims_[axis] * trailing;

                    if (hw_curr == 0) {
                        continue;
                    }

                    if (input_scale == output_scale) {
                        ret = API_LIB(memcpy_i8o8)((int8_t *)dst, (int8_t *)src, hw_curr * 4);
                        dst += hw_curr;
                    } else {
                        return T_ERR_INVALID_PARA;
                    }
                }
            } else {
                for (int32_t i = 0; i < input_num; ++i) {
                    if (tensors[i]->dtype_ != Int32) {
                        return T_ERR_INVALID_DATATYPE;
                    }

                    int32_t *src = (int32_t *)tensors[i]->dptr_;
                    int32_t input_scale = tensors[i]->scale_;
                    int32_t hw_curr = tensors[i]->shape_.dims_[axis] * trailing;

                    if (hw_curr == 0) {
                        continue;
                    }

                    if (input_scale == output_scale) {
                        opi_psram_cpy_out(dst, src, hw_curr * 4);
                        dst += hw_curr;
                        ret = T_SUCCESS;
                    } else {
                        return T_ERR_INVALID_PARA;
                    }
                }
            }
        } else {
            if (output->mem_.type_ == 2) {
                for (int32_t i = 0; i < input_num; ++i) {
                    if (tensors[i]->dtype_ != Int32) {
                        return T_ERR_INVALID_DATATYPE;
                    }

                    int32_t *src = (int32_t *)tensors[i]->dptr_;
                    int32_t input_scale = tensors[i]->scale_;
                    int32_t hw_curr = tensors[i]->shape_.dims_[axis] * trailing;

                    if (hw_curr == 0) {
                        continue;
                    }

                    if (input_scale == output_scale) {
                        for (int32_t l = 0; l < leading; ++l) {
                            int32_t *indptr_curr = (int32_t *)src + l * hw_curr;
                            int32_t *output_ptr = (int32_t *)dst + l * hw + i * hw_curr;
                            ret = API_LIB(memcpy_i8o8)((int8_t *)output_ptr, (int8_t *)indptr_curr, hw_curr * 4);
                        }
                    } else {
                        return T_ERR_INVALID_DATATYPE;
                    }
                }
            } else {
                for (int32_t i = 0; i < input_num; ++i) {
                    if (tensors[i]->dtype_ != Int32) {
                        return T_ERR_INVALID_DATATYPE;
                    }

                    int32_t *src = (int32_t *)tensors[i]->dptr_;
                    int32_t input_scale = tensors[i]->scale_;
                    int32_t hw_curr = tensors[i]->shape_.dims_[axis] * trailing;

                    if (hw_curr == 0) {
                        continue;
                    }

                    if (input_scale == output_scale) {
                        for (int32_t l = 0; l < leading; ++l) {
                            int32_t *indptr_curr = (int32_t *)src + l * hw_curr;
                            int32_t *output_ptr = (int32_t *)dst + l * hw + i * hw_curr;
                            opi_psram_cpy_out((void *)output_ptr, (void *)indptr_curr, hw_curr * 4);
                            ret = T_SUCCESS;
                        }
                    } else {
                        return T_ERR_INVALID_DATATYPE;
                    }
                }
            }
        }
    } else {
        ret = T_ERR_INVALID_DATATYPE;
    }

    return ret;
}

#endif  //_CONCAT_VENUS_H_