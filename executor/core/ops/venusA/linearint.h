#ifndef _LINEARINT_LUNA_H_
#define _LINEARINT_LUNA_H_

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#include "luna/include/cache.h"
#define API_LIB(api) luna_##api
#endif
#include "thinker_define.h"


/**
 * @brief Linear transformation with integer quantization
 * @param input Input tensor
 * @param weight Weight tensor
 * @param bias Bias tensor (optional)
 * @param attrs Linear transformation attributes
 * @param workspace Temporary workspace tensor
 * @param output Output tensor
 * @return int32_t Operation status
 */
int32_t linearint_luna(tTensor *input, tTensor *weight, tTensor *bias, LinearIntAttrs *attrs, tTensor *workspace, tTensor *output) {

    tShape new_shape;

    // Reshape input tensor for 2D processing
    if (input->shape_.ndim_ == 1) {
        new_shape.ndim_ = 2;
        new_shape.dims_[1] = input->shape_.dims_[0];
        new_shape.dims_[0] = 1;
    } else {
        new_shape.ndim_ = 2;
        new_shape.dims_[0] = 1;
        for (int32_t i = 0; i < input->shape_.ndim_ - 1; ++i)
            new_shape.dims_[0] *= input->shape_.dims_[i];
        new_shape.dims_[1] = input->shape_.dims_[input->shape_.ndim_ - 1];
    }


    // Check if input and output are in PSram
    int32_t x_in_psram = (input->mem_.type_ != 2);
    int32_t y_in_psram = (output->mem_.type_ != 2);

#if THINKER_PARAM_CHECK
if (!((input->dtype_ == Int8  && (weight->dtype_ == Int4 || weight->dtype_ == Int8 || weight->dtype_ == Int32))  ||
        (input->dtype_ == Int32 && (weight->dtype_ == Int32 || weight->dtype_ == Int8)) ||
        (input->dtype_ == Int16 && weight->dtype_ == Int16))) {
    return (T_ERR_INVALID_DATATYPE);
}
#endif

    if (attrs->transB != 1) {
        return T_ERR_INVALID_PARA;
    }

    // Determine output index and tensor dimensions
    int32_t ou_idx = (output->dtype_ & 0xF) >> 1;
    int32_t n_dim = new_shape.ndim_;
    int32_t M = new_shape.dims_[n_dim - 2];
    int32_t N = new_shape.dims_[n_dim - 1];
    int32_t L = weight->shape_.dims_[0];
    int32_t input_num =  M * N;
    int32_t output_num = M * L;
    int32_t weight_size = L * N;
    // Validate weight dimensions
    if (weight->shape_.dims_[n_dim - 1] != new_shape.dims_[n_dim - 1]) {
        return T_ERR_INVALID_DATATYPE;
    }

    // Data pointers
    int8_t *src = (int8_t *)input->dptr_;
    int8_t *p_weight = (int8_t *)weight->dptr_;
    int32_t *p_bias = bias ? (int32_t *)bias->dptr_ : NULL;
    int8_t *dst = (int8_t *)output->dptr_;
    size_t workspace_size = workspace ? getTensorDataSize(workspace) : 0;
    if (workspace == NULL) {
        return T_ERR_NO_WORKSPACE;
    }

    // Quantization scales and shift
    int32_t q_i = (int32_t)input->scale_;
    int32_t q_w = (int32_t)weight->scale_;
    int32_t q_o = (int32_t)output->scale_;
    int32_t shift = q_i + q_w - q_o;

    // Check shift validity. Negative shifts are implemented as scalar left-shifts.
    if (shift > 63) {
        return T_ERR_INVALID_PARA;
    }
    if (shift < 0) {
        if (output->dtype_ == Int8) {
            return T_ERR_INVALID_DATATYPE;
        }
        int32_t max_lshift = (output->dtype_ == Int16) ? 14 : 30;
        if ((-shift) > max_lshift) {
            return T_ERR_INVALID_PARA;
        }
    }

    // Temporary workspace pointer
    int8_t *p_tmp = (workspace != NULL) ? (int8_t *)workspace->dptr_ : NULL;
    int8_t *p_src = p_tmp;
    int32_t input_dtype_size = input->byte_;
    int32_t output_dtype_size = output->byte_;
    int32_t input_size = input_num * input_dtype_size;
    int32_t output_size = output_num * output_dtype_size;
    int32_t offset = input_size;
    #if THINKER_RUNTIME_CHECK
    if (x_in_psram && workspace_size < (size_t)input_size * 2) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif
    // Main computation based on data types - Handle input transpose
    if(input->dtype_ == Int8) {
        // Handle Int8 input transpose
        if (x_in_psram == 1) {
            int8_t *src_tmp = p_tmp + offset;
            THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(src_tmp, src, M * N), "luna_memcpy_i8o8");
            src = src_tmp;
        }
        if (ALIGN4(M) * ALIGN8(N) <= 65536) {
            THINKER_RET_CHECK(API_LIB(mat_trans_i8o8)(src, p_src, M, N), "luna_mat_trans_i8o8");
        }
        else{
            THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(src, p_src, M, N), "luna_split_mat_trans_i8o8");
        }

        if ((weight->dtype_ == Int4 || weight->dtype_ == Int8)  && output->dtype_ == Int8) {
            int32_t out_trans_large = (ALIGN4(L) * ALIGN8(M) > 65536);
            int32_t required_workspace = input_size;
            if (x_in_psram) {
                required_workspace = MAX(required_workspace, input_size * 2);
            }
            if (y_in_psram) {
                if (out_trans_large) {
                    required_workspace = MAX(required_workspace, MAX(input_size, output_size) + output_size);
                } else {
                    required_workspace = MAX(required_workspace, input_size + output_size);
                }
            } else if (out_trans_large) {
                required_workspace = MAX(required_workspace, input_size + output_size);
            }
            if (workspace_size < required_workspace) {
                return T_ERR_NO_WORKSPACE;
            }

            if (y_in_psram) {
                int8_t *dst_tmp = p_tmp + (out_trans_large ? MAX(input_size, output_size) : input_size);
                if (weight->dtype_ == Int4)
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i4i8i32o8)(p_weight, p_src, p_bias, dst_tmp, L, ALIGN2(N), M, shift), "luna_split_mat_mul_bias_i4i8i32o8");
                else
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o8)(p_weight, p_src, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i8i8i32o8");

                if (!out_trans_large) {
                    THINKER_RET_CHECK(API_LIB(mat_trans_i8o8)(dst_tmp, dst_tmp, L, M), "luna_mat_trans_i8o8");
                }
                else {
                    int8_t *dst_tmp1 = p_tmp;
                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(dst_tmp, dst_tmp1, L, M), "luna_split_mat_trans_i8o8");
                    dst_tmp = dst_tmp1;
                }
                opi_psram_cpy_out((void *)output->dptr_, dst_tmp, output_size);
            }
            else {
                int8_t *dst_tmp = (int8_t *)output->dptr_;
                if (out_trans_large) {
                    dst_tmp = p_tmp + input_size;
                }
                if (weight->dtype_ == Int4)
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i4i8i32o8)(p_weight, p_src, p_bias, dst_tmp, L, ALIGN2(N), M, shift), "luna_split_mat_mul_bias_i4i8i32o8");
                else
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o8)(p_weight, p_src, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i8i8i32o8");

                if (!out_trans_large) {
                    THINKER_RET_CHECK(API_LIB(mat_trans_i8o8)((int8_t *)dst_tmp, (int8_t *)output->dptr_, L, M), "luna_mat_trans_i8o8");
                }
                else {
                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(dst_tmp, (int8_t *)output->dptr_, L, M), "luna_split_mat_trans_i8o8");
                }
            }
        }
        else if (weight->dtype_ == Int8 && output->dtype_ == Int16) {
            if (y_in_psram) {
                int16_t *dst_tmp = (int16_t *)(p_tmp + input_size);
                if (ALIGN4(L) * ALIGN4(M) > 65536) {
                    dst_tmp = (int16_t *)(p_tmp + MAX(input_size, output_size));
                }
                if (shift < 0) {
                    int32_t scale_out = 1UL << (-shift);
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o16)(p_weight, p_src, p_bias, dst_tmp, L, N, M, 0), "luna_split_mat_mul_bias_i8i8i32o16");
                    THINKER_RET_CHECK(API_LIB(scale_i16i16o16)(dst_tmp, scale_out, dst_tmp, M * L, 0), "luna_scale_i16i16o16");
                }
                else {
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o16)(p_weight, p_src, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i8i8i32o16");
                }

                if (ALIGN4(L) * ALIGN4(M) <= 65536) {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (input_num + output_num) * 2) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(mat_trans_i16o16)(dst_tmp, dst_tmp, L, M), "luna_mat_trans_i16o16");
                    opi_psram_cpy_out((void *)output->dptr_, dst_tmp, output_size);
                }
                else {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (output_num + MAX(input_num, output_num)) * 2) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    int16_t *dst_tmp1 = (int16_t *)p_tmp;
                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i16o16)((int16_t *)dst_tmp, (int16_t *)dst_tmp1, L, M), "luna_split_mat_trans_i16o16");
                    opi_psram_cpy_out((void *)output->dptr_, dst_tmp1, output_size);
                }
            }
            else {
                int16_t *dst_tmp = (int16_t *)output->dptr_;
                if (ALIGN4(L) * ALIGN4(M) > 65536) {
                    dst_tmp = (int16_t *)(p_tmp + input_size);
                }

                if (shift < 0) {
                    int32_t scale_out = 1UL << (-shift);
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o16)(p_weight, p_src, p_bias, dst_tmp, L, N, M, 0), "luna_split_mat_mul_bias_i8i8i32o16");
                    THINKER_RET_CHECK(API_LIB(scale_i16i16o16)(dst_tmp, scale_out, dst_tmp, M * L, 0), "luna_scale_i16i16o16");
                }
                else {
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o16)(p_weight, p_src, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i8i8i32o16");
                }

                if (ALIGN4(L) * ALIGN4(M) <= 65536) {
#if THINKER_RUNTIME_CHECK
if (workspace_size < input_num * 2) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(mat_trans_i16o16)(dst_tmp, (int16_t *)output->dptr_, L, M), "luna_mat_trans_i16o16");
                }
                else {
#if THINKER_RUNTIME_CHECK
if (workspace_size < ((input_num + output_num) * 2)) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i16o16)(dst_tmp,  (int16_t *)output->dptr_, L, M), "luna_split_mat_trans_i16o16");
                }
            }
        }
        else if (weight->dtype_ == Int8 && output->dtype_ == Int32) {
            if (y_in_psram) {
                int32_t *dst_tmp = (int32_t *)(p_tmp + input_num * 4);
                if (ALIGN2(L) * ALIGN4(M) > 32768) {
                    dst_tmp = (int32_t *)(p_tmp + MAX(input_num * 4, output_size));
                }
                if (shift < 0) {
                    int32_t scale_out = 1UL << (-shift);
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o32)(p_weight, p_src, p_bias, dst_tmp, L, N, M, 0), "luna_split_mat_mul_bias_i8i8i32o32");
                    THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(dst_tmp, scale_out, dst_tmp, M * L, 0), "luna_scale_i32i32o32");
                }
                else {
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o32)(p_weight, p_src, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i8i8i32o32");
                }

                if (ALIGN2(L) * ALIGN4(M) <= 32768) {
#if THINKER_RUNTIME_CHECK
if (workspace_size < ((input_num + output_num) * 4)) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(mat_trans_i32o32)((int32_t *)dst_tmp, (int32_t *)dst_tmp, L, M), "luna_mat_trans_i32o32");
                    opi_psram_cpy_out((void *)output->dptr_, dst_tmp, output_size);
                }
                else {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (MAX(input_num, output_num) + output_num) * 4) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    int32_t *dst_tmp1 = (int32_t *)p_tmp;
                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i32o32)((int32_t *)dst_tmp, (int32_t *)dst_tmp1, L, M), "luna_split_mat_trans_i32o32");
                    opi_psram_cpy_out((void *)output->dptr_, dst_tmp1, output_size);
                }
            }
            else {
                int32_t *dst_tmp = (int32_t *)output->dptr_;
                if (ALIGN2(L) * ALIGN4(M) > 32768) {
                    dst_tmp = (int32_t *)(p_tmp + input_num * 4);
                }
                if (shift < 0) {
                    int32_t scale_out = 1UL << (-shift);
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o32)(p_weight, p_src, p_bias, dst_tmp, L, N, M, 0), "luna_split_mat_mul_bias_i8i8i32o32");
                    THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(dst_tmp, scale_out, dst_tmp, M * L, 0), "luna_scale_i32i32o32");
                }
                else {
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o32)(p_weight, p_src, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i8i8i32o32");
                }
                if (ALIGN2(L) * ALIGN4(M) <= 32768) {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (input_num * 4)) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(mat_trans_i32o32)(dst_tmp, (int32_t *)output->dptr_, L, M), "luna_mat_trans_i32o32");
                }
                else {
#if THINKER_RUNTIME_CHECK
if (workspace_size < input_num * 4 + output_size) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i32o32)(dst_tmp, (int32_t *)output->dptr_, L, M), "luna_split_mat_trans_i32o32");
                }
            }
        }

        else if (weight->dtype_ == Int32 && output->dtype_ == Int32){
            int32_t *p_32_weight = (int32_t *)p_weight;
            int32_t *p_src_i32 = (int32_t *)p_src;
            int16_t *p_src_i16 = (int16_t *)(p_src + input_num * 4);
            THINKER_RET_CHECK(API_LIB(scale_i8i8o16)(p_src, 1, p_src_i16, input_num, 0), "luna_scale_i8i8o16");
            THINKER_RET_CHECK(API_LIB(scale_i16i16o32)(p_src_i16, 1L, p_src_i32, input_num, 0), "luna_scale_i16i16o32");
            int32_t input_32_size = input_num * 4;

            if (y_in_psram) {
                int32_t *dst_tmp = (int32_t *)(p_tmp + input_32_size);
                if (ALIGN2(L) * ALIGN4(M) > 32768) {
                    dst_tmp =(int32_t *)(p_tmp + MAX(input_32_size, output_size));
                }
                THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i32i32i32o32)((int32_t *)p_32_weight, (int32_t *)p_src_i32, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i32i32i32o32");

                if (ALIGN2(L) * ALIGN4(M) <= 32768) {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (input_32_size +  MAX(output_size, input_num * 2))) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(mat_trans_i32o32)(dst_tmp, dst_tmp, L, M), "luna_mat_trans_i32o32");
                    opi_psram_cpy_out((void *)output->dptr_, dst_tmp, output_size);
                }
                else {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (MAX(input_32_size, output_size)  + MAX(output_size, input_num * 2))) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    int32_t *dst_tmp1 = (int32_t *)p_tmp;
                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i32o32)(dst_tmp, dst_tmp1, L, M), "luna_split_mat_trans_i32o32");
                    opi_psram_cpy_out((void *)output->dptr_, dst_tmp1, output_size);
                }
            }
            else {
                int32_t *dst_tmp = (int32_t *)output->dptr_;
                if (ALIGN2(L) * ALIGN4(M) > 32768) {
                    dst_tmp =(int32_t *)(p_tmp + input_32_size);
                }
                THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i32i32i32o32)((int32_t *)p_32_weight, (int32_t *)p_src_i32, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i32i32i32o32");
                if (ALIGN2(L) * ALIGN4(M) <= 32768) {
#if THINKER_RUNTIME_CHECK
if (workspace_size < input_32_size + input_num * 2) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(mat_trans_i32o32)(dst_tmp, (int32_t *)output->dptr_, L, M), "luna_mat_trans_i32o32");
                }
                else {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (input_32_size + MAX(output_size, input_num * 2))) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i32o32)(dst_tmp, (int32_t *)output->dptr_, L, M), "luna_split_mat_trans_i32o32");
                }
            }
        }
        else{
            return T_ERR_INVALID_DATATYPE;
        }
    }
    else if (input->dtype_ == Int16) {
        if (weight->dtype_ != Int16) {
            return T_ERR_INVALID_DATATYPE;
        }

        int16_t *p_src_i16 = (int16_t *)p_src;
        int16_t *p_weight_i16 = (int16_t *)p_weight;

        if (x_in_psram == 1) {
            int8_t *src_tmp = (int8_t *)p_tmp + offset;
            THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(src_tmp, src, M * N * 2), "luna_memcpy_i8o8");
            src = src_tmp;
        }
        if (ALIGN4(M) * ALIGN4(N) <= 65536) {
            THINKER_RET_CHECK(API_LIB(mat_trans_i16o16)((int16_t *)src, (int16_t *)p_src, M, N), "luna_mat_trans_i16o16");
        }
        else{
            THINKER_RET_CHECK(API_LIB(split_mat_trans_i16o16)((int16_t *)src, (int16_t *)p_src, M, N), "luna_split_mat_trans_i16o16");
        }

        // Int16 x Int16 -> Int8 output
        if (output->dtype_ == Int8) {
            if (y_in_psram) {
                int8_t *dst_tmp = (int8_t *)p_tmp + input_size;
                if (ALIGN4(L) * ALIGN8(M) > 65536) {
                    dst_tmp = (int8_t *)(p_tmp + MAX(input_size, output_size));
                }

                if (shift < 0) {
                    int32_t scale_out = 1UL << (-shift);
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i16i16i32o8)(p_weight_i16, p_src_i16, p_bias, dst_tmp, L, N, M, 0), "luna_split_mat_mul_bias_i16i16i32o8");
                    THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(dst_tmp, scale_out, dst_tmp, M * L, 0), "luna_scale_i8i8o8");
                }else{
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i16i16i32o8)(p_weight_i16, p_src_i16, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i16i16i32o8");
                }
                if (ALIGN4(L) * ALIGN8(M) <= 65536) {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (input_size + MAX(input_size, output_size))) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(mat_trans_i8o8)(dst_tmp, dst_tmp, L, M), "luna_mat_trans_i8o8");
                    opi_psram_cpy_out((void *)output->dptr_, dst_tmp, output_size);
                } else {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (MAX(input_size, output_size) * 2)) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    int8_t *dst_tmp1 = p_tmp;
                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(dst_tmp, dst_tmp1, L, M), "luna_split_mat_trans_i8o8");
                    opi_psram_cpy_out((void *)output->dptr_, dst_tmp1, output_size);
                }
            } else {
                int8_t *dst_tmp = (int8_t *)output->dptr_;
                if (shift < 0) {
                    int32_t scale_out = 1UL << (-shift);
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i16i16i32o8)(p_weight_i16, p_src_i16, p_bias, dst_tmp, L, N, M, 0), "luna_split_mat_mul_bias_i16i16i32o8");
                    THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(dst_tmp, scale_out, dst_tmp, M * L, 0), "luna_scale_i8i8o8");
                }else{
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i16i16i32o8)(p_weight_i16, p_src_i16, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i16i16i32o8");
                }
                if (ALIGN4(L) * ALIGN8(M) <= 65536) {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (input_size + MAX(input_size, output_size))) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(mat_trans_i8o8)(dst_tmp, (int8_t *)output->dptr_, L, M), "luna_mat_trans_i8o8");
                } else {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (MAX(input_size, output_size) * 2)) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(dst_tmp, (int8_t *)output->dptr_, L, M), "luna_split_mat_trans_i8o8");
                }
            }
        }
        else if (output->dtype_ == Int16) {
            if (y_in_psram) {
                int16_t *dst_tmp = (int16_t *)(p_tmp + input_size);
                if (ALIGN4(L) * ALIGN4(M) > 65536) {
                    dst_tmp = (int16_t *)(p_tmp + MAX(input_size, output_size));
                }
                if (shift < 0) {
                    int32_t scale_out = 1UL << (-shift);
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i16i16i32o16)(p_weight_i16, p_src_i16, p_bias, dst_tmp, L, N, M, 0), "luna_split_mat_mul_bias_i8i8i32o16");
                    THINKER_RET_CHECK(API_LIB(scale_i16i16o16)(dst_tmp, scale_out, dst_tmp, M * L, 0), "luna_scale_i16i16o16");
                }else{
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i16i16i32o16)(p_weight_i16, p_src_i16, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i16i16i32o16");
                }
                if (ALIGN4(L) * ALIGN4(M) <= 65536) {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (input_size + MAX(output_size, output_size))) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(mat_trans_i16o16)(dst_tmp, dst_tmp, L, M), "luna_mat_trans_i16o16");
                    opi_psram_cpy_out((void *)output->dptr_, dst_tmp, output_size);
                } else {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (MAX(input_size , output_size)* 2)) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    int16_t *dst_tmp1 = (int16_t *)p_tmp;
                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i16o16)(dst_tmp, dst_tmp1, L, M), "luna_split_mat_trans_i16o16");
                    opi_psram_cpy_out((void *)output->dptr_, dst_tmp1, output_size);
                }
            } else {
                int16_t *dst_tmp = (int16_t *)output->dptr_;
                if (ALIGN4(L) * ALIGN4(M) > 65536) {
                    dst_tmp = (int16_t *)(p_tmp + input_size);
                }
                if (shift < 0) {
                    int32_t scale_out = 1UL << (-shift);
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i16i16i32o16)(p_weight_i16, p_src_i16, p_bias, dst_tmp, L, N, M, 0), "luna_split_mat_mul_bias_i8i8i32o16");
                    THINKER_RET_CHECK(API_LIB(scale_i16i16o16)(dst_tmp, scale_out, dst_tmp, M * L, 0), "luna_scale_i16i16o16");
                }else{
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i16i16i32o16)(p_weight_i16, p_src_i16, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i16i16i32o16");
                }

                if (ALIGN4(L) * ALIGN4(M) <= 65536) {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (input_size + MAX(input_size, output_size) )) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(mat_trans_i16o16)(dst_tmp, (int16_t *)output->dptr_, L, M), "luna_mat_trans_i16o16");
                } else {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (MAX(input_size , output_size) * 2)) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i16o16)(dst_tmp, (int16_t *)output->dptr_, L, M), "luna_split_mat_trans_i16o16");
                }
            }
        }
        // Int16 x Int16 -> Int32 output
        else if (output->dtype_ == Int32) {
            if (y_in_psram) {
                int32_t *dst_tmp = (int32_t *)(p_tmp + input_size);
                if (ALIGN2(L) * ALIGN4(M) > 32768) {
                    dst_tmp = (int32_t *)(p_tmp + MAX(input_size, output_size));
                }
                if (shift < 0) {
                    int32_t scale_out = 1UL << (-shift);
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i16i16i32o32)(p_weight_i16, p_src_i16, p_bias, dst_tmp, L, N, M, 0), "luna_split_mat_mul_bias_i16i16i32o32");
                    THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(dst_tmp, scale_out, dst_tmp, M * L, 0), "luna_scale_i32i32o32");
                }
                else {
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i16i16i32o32)(p_weight_i16, p_src_i16, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i16i16i32o32");
                }

                if (ALIGN2(L) * ALIGN4(M) <= 32768) {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (input_size + output_size)) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(mat_trans_i32o32)((int32_t *)dst_tmp, (int32_t *)dst_tmp, L, M), "luna_mat_trans_i32o32");
                    opi_psram_cpy_out((void *)output->dptr_, dst_tmp, output_size);
                }
                else {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (MAX(input_size, output_size) + output_size)) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    int32_t *dst_tmp1 = (int32_t *)p_tmp;
                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i32o32)((int32_t *)dst_tmp, (int32_t *)dst_tmp1, L, M), "luna_split_mat_trans_i32o32");
                    opi_psram_cpy_out((void *)output->dptr_, dst_tmp1, output_size);
                }
            }
            else {
                int32_t *dst_tmp = (int32_t *)output->dptr_;
                if (ALIGN2(L) * ALIGN4(M) > 32768) {
                    dst_tmp = (int32_t *)(p_tmp + input_size);
                }
                if (shift < 0) {
                    int32_t scale_out = 1UL << (-shift);
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i16i16i32o32)(p_weight_i16, p_src_i16, p_bias, dst_tmp, L, N, M, 0), "luna_split_mat_mul_bias_i16i16i32o32");
                    THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(dst_tmp, scale_out, dst_tmp, M * L, 0), "luna_scale_i32i32o32");
                }
                else {
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i16i16i32o32)(p_weight_i16, p_src_i16, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i16i16i32o32");
                }
                if (ALIGN2(L) * ALIGN4(M) <= 32768) {
#if THINKER_RUNTIME_CHECK
if (workspace_size < input_size) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(mat_trans_i32o32)(dst_tmp, (int32_t *)output->dptr_, L, M), "luna_mat_trans_i32o32");
                }
                else {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (input_size + output_size)) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i32o32)(dst_tmp, (int32_t *)output->dptr_, L, M), "luna_split_mat_trans_i32o32");
                }
            }
        }
        else {
            return T_ERR_INVALID_DATATYPE;
        }
    }
    else if (input->dtype_ == Int32) {
        if (x_in_psram == 1) {
            int32_t *src_tmp = (int32_t *)(p_tmp + offset);
            THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)src_tmp, (int8_t *)src, M * N *4), "luna_memcpy_i8o8");
            THINKER_RET_CHECK(API_LIB(split_mat_trans_i32o32)(src_tmp, (int32_t *)p_src, M, N), "luna_split_mat_trans_i32o32");
        }
        else {
            // Handle Int32 input transpose
            if (ALIGN4(M) * ALIGN4(N) <= 65536) {
                THINKER_RET_CHECK(API_LIB(mat_trans_i32o32)((int32_t *)src, (int32_t *)p_src, M, N), "luna_mat_trans_i32o32");
            }else{
                THINKER_RET_CHECK(API_LIB(split_mat_trans_i32o32)((int32_t *)src, (int32_t *)p_src, M, N), "luna_split_mat_trans_i32o32");
            }
        }

        if (weight->dtype_ == Int32 && output->dtype_ == Int32) {
            if (y_in_psram) {
                int32_t *dst_tmp = (int32_t *)(p_tmp + input_size);
                if (ALIGN2(L) * ALIGN4(M) > 32768) {
                    dst_tmp = (int32_t *)(p_tmp + MAX(input_size, output_size));
                }
                if (shift < 0) {
                    int32_t scale_out = 1UL << (-shift);
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i32i32i32o32)((int32_t *)p_weight, (int32_t *)p_src, p_bias, dst_tmp, L, N, M, 0), "luna_split_mat_mul_bias_i32i32i32o32");
                    THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(dst_tmp, scale_out, dst_tmp, M * L, 0), "luna_scale_i32i32o32");
                }
                else {
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i32i32i32o32)((int32_t *)p_weight, (int32_t *)p_src, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i32i32i32o32");
                }
                if (ALIGN2(L) * ALIGN4(M) <= 32768) {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (input_size + output_size)) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(mat_trans_i32o32)((int32_t *)dst_tmp, (int32_t *)dst_tmp, L, M), "luna_mat_trans_i32o32");
                    opi_psram_cpy_out((void *)output->dptr_, dst_tmp, output_size );
                }
                else {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (MAX(input_size, output_size)  + output_size )) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    int32_t *dst_tmp1 = (int32_t *)p_tmp;
                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i32o32)((int32_t *)dst_tmp, (int32_t *)dst_tmp1, L, M), "luna_split_mat_trans_i32o32");
                    opi_psram_cpy_out((void *)output->dptr_, dst_tmp1, output_size );
                }
            }
            else {
                int32_t *dst_tmp = (int32_t *)output->dptr_;
                if (ALIGN2(L) * ALIGN4(M) > 32768) {
                    dst_tmp = (int32_t *)(p_tmp + input_size);
                }
                if (shift < 0) {
                    int32_t scale_out = 1UL << (-shift);
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i32i32i32o32)((int32_t *)p_weight, (int32_t *)p_src, p_bias, dst_tmp, L, N, M, 0), "luna_split_mat_mul_bias_i32i32i32o32");
                    THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(dst_tmp, scale_out, dst_tmp, M * L, 0), "luna_scale_i32i32o32");
                }
                else {
                    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i32i32i32o32)((int32_t *)p_weight, (int32_t *)p_src, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i32i32i32o32");
                }
                if (ALIGN2(L) * ALIGN4(M) <= 32768) {
#if THINKER_RUNTIME_CHECK
if (workspace_size < input_size) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(mat_trans_i32o32)(dst_tmp, (int32_t *)output->dptr_, L, M), "luna_mat_trans_i32o32");
                }
                else {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (input_size + output_size)) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i32o32)(dst_tmp, (int32_t *)output->dptr_, L, M), "luna_split_mat_trans_i32o32");
                }
            }
        }
        else if ((weight->dtype_ == Int32 || weight->dtype_ == Int8) && output->dtype_ == Int8){
            int32_t *p_weight_i32 = (int32_t *)p_weight;
            int32_t weight_input_size = 0;
            int32_t weight_sram_size = 0;
            if (weight->dtype_ == Int8){
                p_weight_i32 = (int32_t *)(p_tmp + input_size);
                int16_t *p_weight_i16 = (int16_t *)(p_tmp + input_size + weight_size * 4);
                THINKER_RET_CHECK(API_LIB(scale_i8i8o16)(p_weight, 1, p_weight_i16, weight_size, 0), "luna_scale_i8i8o16");
                THINKER_RET_CHECK(API_LIB(scale_i16i16o32)(p_weight_i16, 1L, p_weight_i32, weight_size, 0), "luna_scale_i16i16o32");
                weight_input_size = weight_size * 4;
                weight_sram_size = weight_size * 6;
            }
            if (y_in_psram) {
                int8_t *dst_tmp = p_tmp + input_size + weight_input_size;
                if (ALIGN4(L) * ALIGN8(M) > 65536) {
                    dst_tmp = p_tmp + MAX(input_size, output_size) + weight_input_size;
                }
                THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i32i32i32o8)((int32_t *)p_weight_i32, (int32_t *)p_src, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i32i32i32o8");

                if (ALIGN4(L) * ALIGN8(M) <= 65536) {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (input_size +  output_size + weight_sram_size)) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(mat_trans_i8o8)(dst_tmp, dst_tmp, L, M), "luna_mat_trans_i8o8");
                    opi_psram_cpy_out((void *)output->dptr_, dst_tmp, output_size);
                }
                else {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (MAX(input_size, output_size)  + output_size + weight_sram_size)) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    int8_t *dst_tmp1 = p_tmp;
                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(dst_tmp, dst_tmp1, L, M), "luna_split_mat_trans_i8o8");
                    opi_psram_cpy_out((void *)output->dptr_, dst_tmp1, output_size);
                }
            }
            else {
                int8_t *dst_tmp = (int8_t *)output->dptr_;
                if (ALIGN4(L) * ALIGN8(M) > 65536) {
                    dst_tmp = p_tmp + input_size + weight_input_size;
                }

                THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i32i32i32o8)((int32_t *)p_weight_i32, (int32_t *)p_src, p_bias, dst_tmp, L, N, M, shift), "luna_split_mat_mul_bias_i32i32i32o8");

                if (ALIGN4(L) * ALIGN8(M) <= 65536) {
#if THINKER_RUNTIME_CHECK
if (workspace_size < input_size + weight_sram_size) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(mat_trans_i8o8)(dst_tmp, (int8_t *)output->dptr_, L, M), "luna_mat_trans_i8o8");
                }
                else {
#if THINKER_RUNTIME_CHECK
if (workspace_size < (input_size + output_size + weight_sram_size)) {
    return (T_ERR_NO_WORKSPACE);
}
#endif
                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(dst_tmp, (int8_t *)output->dptr_, L, M), "luna_split_mat_trans_i8o8");
                }
            }
        }
        else{
            return T_ERR_INVALID_DATATYPE;
        }
    }
    else {
        return T_ERR_INVALID_DATATYPE;
    }

    return T_SUCCESS;
}

#endif  // _LINEARINT_LUNA_H_
