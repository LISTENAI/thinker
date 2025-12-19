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
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    tShape new_shape;

    // Reshape input tensor for 2D processing
    if (input->shape_.ndim_ == 1) {
        new_shape.ndim_ = 2;
        new_shape.dims_[1] = input->shape_.dims_[0];
        new_shape.dims_[0] = 1;
    } else if (input->shape_.ndim_ == 3) {
        new_shape.ndim_ = 2;
        new_shape.dims_[0] = input->shape_.dims_[0] * input->shape_.dims_[1];
        new_shape.dims_[1] = input->shape_.dims_[2];
    } else {
        new_shape = input->shape_;
    }

    // Check if input and output are in PSram
    int32_t in_is_psram = (input->mem_.type_ != 2);
    int32_t ou_is_psram = (output->mem_.type_ != 2);

    // Validate input data type
    if (input->dtype_ != Int8) {
        return T_ERR_INVALID_DATATYPE;
    }

    // Determine output index and tensor dimensions
    int32_t ou_idx = (output->dtype_ & 0xF) >> 1;
    int32_t n_dim = new_shape.ndim_;
    int32_t M = new_shape.dims_[n_dim - 2];
    int32_t N = new_shape.dims_[n_dim - 1];
    int32_t L = weight->shape_.dims_[0];

    // Validate weight dimensions
    if (weight->shape_.dims_[n_dim - 1] != new_shape.dims_[n_dim - 1]) {
        return T_ERR_INVALID_DATATYPE;
    }

    // Data pointers
    int8_t *src = (int8_t *)input->dptr_;
    int8_t *p_weight = (int8_t *)weight->dptr_;
    int32_t *p_bias = bias ? (int32_t *)bias->dptr_ : NULL;
    int8_t *dst = (int8_t *)output->dptr_;
    int32_t workspace_size = (workspace != NULL) ? workspace->shape_.dims_[0] : 0;

    // Quantization scales and shift
    int32_t q_i = (int32_t)input->scale_;
    int32_t q_w = (int32_t)weight->scale_;
    int32_t q_o = (int32_t)output->scale_;
    int32_t shift = q_i + q_w - q_o;

    // Check shift validity
    if ((shift < 0) && (output->dtype_ == Int8)) {
        return ret;
    }

    // Temporary workspace pointer
    int8_t *p_tmp = (workspace != NULL) ? (int8_t *)workspace->dptr_ : NULL;

    // Input size
    int32_t input_size = getShapeSize(&(input->shape_));

    // Main computation based on data types
    if (weight->dtype_ == Int8) {
        ret = API_LIB(split_mat_trans_i8o8)(src, p_tmp, M, N);
    } else {
        ret = API_LIB(scale_i8i8o32)(src, 1, (int32_t *)p_tmp, M * N, 0);
        ret = API_LIB(split_mat_trans_i32o32)((int32_t *)p_tmp, (int32_t *)p_tmp, M, N);
    }

    // Output size
    int32_t output_size = getShapeSize(&(output->shape_));

    // Determine destination pointer based on memory type
    if (ou_is_psram) {
        int32_t offset = (input_size * weight->byte_ > output_size * output->byte_) ? input_size * weight->byte_ : output_size * output->byte_;
        dst = (int8_t *)workspace->dptr_ + offset;
    }

    // Execute matrix multiplication and bias addition based on data types
    if (weight->dtype_ == Int4 && output->dtype_ == Int8) {
        ret |= API_LIB(split_mat_mul_bias_i4i8i32o8)(p_weight, p_tmp, p_bias, dst, L, ALIGN2(N), M, shift);
        if (ou_is_psram) {
            ret |= API_LIB(split_mat_trans_i8o8)(dst, p_tmp, L, M);
            opi_psram_cpy_out((void *)output->dptr_, p_tmp, L * M);
        } else {
            ret |= API_LIB(split_mat_trans_i8o8)(dst, (int8_t *)output->dptr_, L, M);
        }
    } else if (weight->dtype_ == Int8 && output->dtype_ == Int8) {
        ret |= API_LIB(split_mat_mul_bias_i8i8i32o8)(p_weight, p_tmp, p_bias, dst, L, N, M, shift);
        if (ou_is_psram) {
            ret |= API_LIB(split_mat_trans_i8o8)(dst, p_tmp, L, M);
            opi_psram_cpy_out((int8_t *)output->dptr_, p_tmp, L * M);
        } else {
            ret |= API_LIB(split_mat_trans_i8o8)(dst, (int8_t *)output->dptr_, L, M);
        }
    } else if (weight->dtype_ == Int8 && output->dtype_ == Int32) {
        if (shift < 0) {
            int32_t scale_out = 1UL << (-shift);
            ret = API_LIB(split_mat_mul_bias_i8i8i32o32)(p_weight, p_tmp, p_bias, (int32_t *)dst, L, N, M, 0);
            ret = API_LIB(scale_i32i32o32)((int32_t *)dst, scale_out, (int32_t *)dst, M * L, 0);
        } else {
            ret = API_LIB(split_mat_mul_bias_i8i8i32o32)(p_weight, p_tmp, p_bias, (int32_t *)dst, L, N, M, shift);
        }
        if (ou_is_psram) {
            ret |= API_LIB(split_mat_trans_i32o32)((int32_t *)dst, (int32_t *)p_tmp, L, M);
            opi_psram_cpy_out((void *)output->dptr_, p_tmp, L * M * output->byte_);
        } else {
            ret |= API_LIB(split_mat_trans_i32o32)((int32_t *)dst, (int32_t *)output->dptr_, L, M);
        }
    } else if (weight->dtype_ == Int32 && output->dtype_ == Int32) {
        if (shift < 0) {
            int32_t scale_out = 1UL << (-shift);
            ret = API_LIB(split_mat_mul_bias_i32i32i32o32)((int32_t *)p_weight, (int32_t *)p_tmp, p_bias, (int32_t *)dst, L, N, M, 0);
            ret = API_LIB(scale_i32i32o32)((int32_t *)dst, scale_out, (int32_t *)dst, M * L, 0);
        } else {
            ret = API_LIB(split_mat_mul_bias_i32i32i32o32)((int32_t *)p_weight, (int32_t *)p_tmp, p_bias, (int32_t *)dst, L, N, M, shift);
        }
        if (ou_is_psram) {
            ret |= API_LIB(split_mat_trans_i32o32)((int32_t *)dst, (int32_t *)p_tmp, L, M);
            opi_psram_cpy_out((void *)output->dptr_, p_tmp, L * M * output->byte_);
        } else {
            ret |= API_LIB(split_mat_trans_i32o32)((int32_t *)dst, (int32_t *)output->dptr_, L, M);
        }
    } else {
        return T_ERR_INVALID_DATATYPE;
    }

    return ret;
}

#endif  // _LINEARINT_LUNA_H_