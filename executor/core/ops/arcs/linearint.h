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
#define API_LIB(api) luna_##api
#endif

#include "thinker_define.h"

// Function pointer types for different operations
typedef int32_t (*FC_MAT_MUL_LUNA_API)(int8_t *src1, int8_t *src2, void *dst,
                                       int32_t row, int32_t col, int32_t col2, int32_t shift);
typedef int32_t (*FC_SPLIT_MAT_MUL_LUNA_API)(int8_t *src1, int8_t *src2, void *dst, 
                                            int32_t row, int32_t col, int32_t col2, int32_t shift);
typedef int32_t (*FC_VEC_ADD_LUNA_API)(void *src1, void *src2, void *dst,
                                       int32_t size, int32_t shift);

// API function table
struct fc_luna_api_item {
    void *luna_api;
};

struct fc_luna_api_item fc_luna_api_list[][2] = {
    {{API_LIB(mat_mul_i8i8o8)},       {API_LIB(mat_mul_i8i8o32)}},
    {{API_LIB(split_mat_mul_i8i8o8)}, {API_LIB(split_mat_mul_i8i8o32)}},
    {{API_LIB(add_i8i8o8)},           {API_LIB(add_i8i8o32)}},
    {{API_LIB(add_i32i32o8)},         {API_LIB(add_i32i32o32)}}};

/**
 * @brief Transposes a 2D int8 matrix
 * @param src Source matrix
 * @param dst Destination matrix
 * @param height Number of rows
 * @param width Number of columns
 */
static void transpose2dInt8to32(int8_t *src, int32_t *dst, int32_t height, int32_t width) {
    for (int32_t i = 0; i < height; i++)
        for(int32_t j = 0; j < width; j++)
            dst[j * height + i] = src[i * width + j];
}

/**
 * @brief Transposes a 2D int8 matrix
 * @param src Source matrix
 * @param dst Destination matrix
 * @param height Number of rows
 * @param width Number of columns
 */
static void transpose2dInt8(int8_t *src, int8_t *dst, int32_t height, int32_t width) {
    for (int32_t i = 0; i < height; i++)
        for(int32_t j = 0; j < width; j++)
            dst[j * height + i] = src[i * width + j];
}

/**
 * @brief Transposes a 2D int32 matrix
 * @param src Source matrix
 * @param dst Destination matrix
 * @param height Number of rows
 * @param width Number of columns
 */
static void transpose2dInt32(int32_t *src, int32_t *dst, int32_t height, int32_t width) {
    for (int32_t i = 0; i < height; i++)
        for(int32_t j = 0; j < width; j++)
            dst[j * height + i] = src[i * width + j];
}

/**
 * @brief Ceiling division operation
 * @param x Dividend
 * @param shift Right shift amount
 * @return Result of ceiling division
 */
static int32_t luna_ceil(int32_t x, int32_t shift) {
    if (x & ~(0xFFFFFFFF << shift)) {
        return (x >> shift) + 1;
    } else {
        return (x >> shift);
    }
}

/**
 * @brief Linear layer implementation for integer tensors
 * @param input Input tensor
 * @param weight Weight tensor
 * @param bias Bias tensor (can be NULL)
 * @param attrs Linear layer attributes
 * @param workspace Workspace buffer
 * @param output Output tensor
 * @return Status code indicating success or failure
 */
int32_t linearint_luna(tTensor *input, tTensor *weight, tTensor *bias, LinearIntAttrs *attrs, tTensor *workspace, tTensor *output) {
    #if THINKER_PARAM_CHECK
    if (input == NULL || weight == NULL || attrs == NULL ||
                        output == NULL || input->dptr_ == 0 || weight->dptr_ == 0 ||
                        output->dptr_ == 0 || attrs->transB != 1 ||
                        input->dtype_ != Int8 ||
                        (weight->dtype_ != Int4 && weight->dtype_ != Int8 &&
                         weight->dtype_ != Int32) ||
                        (output->dtype_ != Int8 && output->dtype_ != Int32) ||
                        (weight->dtype_ == Int32 && output->dtype_ != Int32) ||
                        (bias != NULL && (bias->dptr_ == 0 || bias->dtype_ != Int32))) {
        return (T_ERR_INVALID_DATATYPE);
    }
#endif
    tShape new_shape;
    
    // Handle different input dimensions
    if (1 == input->shape_.ndim_) {
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

    // Check memory types
    int32_t in_is_psram = (2 != input->mem_.type_) ? 1 : 0;
    int32_t ou_is_psram = (2 != output->mem_.type_) ? 1 : 0;

    // Validate input data type
    #ifdef RUNTIME_PARAM_CHECK
    if (Int8 != input->dtype_)
        return T_ERR_INVALID_DATATYPE;
    #endif

    // Get dimensions
    int32_t ou_idx = (output->dtype_ & 0xF) >> 1;
    int32_t n_dim = new_shape.ndim_;
    int32_t M = new_shape.dims_[n_dim - 2];
    int32_t N = new_shape.dims_[n_dim - 1];
    int32_t L = weight->shape_.dims_[0];
    
    #ifdef RUNTIME_PARAM_CHECK
    if (weight->shape_.dims_[n_dim - 1] != new_shape.dims_[n_dim - 1])
        return T_ERR_INVALID_DATATYPE;
    #endif

    // Get pointers and scales
    int8_t *src = (int8_t *)input->dptr_;
    int8_t *p_weight = (int8_t *)weight->dptr_;
    int32_t *p_bias = bias ? (int32_t *)bias->dptr_ : NULL;
    int8_t *dst = (int8_t *)output->dptr_;
    size_t workspace_size = workspace ? getTensorDataSize(workspace) : 0;

    int32_t q_i = (int32_t)input->scale_;
    int32_t q_w = (int32_t)weight->scale_;
    int32_t q_o = (int32_t)output->scale_;
    int32_t shift = q_i + q_w - q_o;
    #if THINKER_PARAM_CHECK
    if (shift > 63 || (output->dtype_ == Int8 && shift < 0) ||
                        (output->dtype_ == Int32 && shift < -30)) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
    #ifdef RUNTIME_PARAM_CHECK
    if ((shift < 0) & (Int8 == output->dtype_)) {
        return T_ERR_INVALID_DATATYPE;
    }
    #endif

    // Temporary buffer allocation

    int32_t input_size = getShapeSize(&(input->shape_));
    size_t required_workspace = (size_t)input_size *
                                (weight->dtype_ == Int32 ? 4 : 1) +
                                getTensorDataSize(output);
    #if THINKER_RUNTIME_CHECK
    if (workspace == NULL || workspace->dptr_ == 0 ||
                          workspace_size < required_workspace) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif
    int8_t *p_tmp = (NULL != workspace) ? (int8_t *)workspace->dptr_ : NULL;
    
    // Handle PSRAM input case
    if (in_is_psram) {
#if !(defined(WIN32) || defined(linux))
        HAL_InvalidateDCache_by_Addr(src, M*N*weight->byte_);
#endif
        
        if ((weight->dtype_ == Int8) || (weight->dtype_ == Int4))
            transpose2dInt8(src, p_tmp, M, N);
        else if (weight->dtype_ == Int32)
            transpose2dInt8to32(src, (int32_t *)p_tmp, M, N);
        else
            return T_ERR_NO_SUPPORT_OP;
    } 
    else {
        if ((weight->dtype_ == Int8) || (weight->dtype_ == Int4))
            THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(src, p_tmp, M, N), "luna_split_mat_trans_i8o8");
        else {
            THINKER_RET_CHECK(API_LIB(scale_i8i8o32)(src, 1, (int32_t *)p_tmp, M * N, 0), "luna_scale_i8i8o32");
            THINKER_RET_CHECK(API_LIB(split_mat_trans_i32o32)((int32_t *)p_tmp, (int32_t *)p_tmp, M, N), "luna_split_mat_trans_i32o32");
        }
    }

    // Set up output buffer
    int32_t output_size = getShapeSize(&(output->shape_));
    if (weight->dtype_ == Int4)
        dst = (int8_t *)workspace->dptr_ + input_size;
    else
        dst = (int8_t *)workspace->dptr_ + input_size * weight->byte_;

    // Execute different matrix multiplication paths based on data types
    if ((Int4 == weight->dtype_) & (Int8 == output->dtype_)) {
        THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i4i8i32o8)(p_weight, p_tmp, p_bias, dst, L, ALIGN2(N), M, shift), "luna_split_mat_mul_bias_i4i8i32o8");
        if (ou_is_psram) {
            transpose2dInt8(dst, (int8_t *)output->dptr_, L, M);
#if !(defined(WIN32) || defined(linux))
            thinker_psram_write_complete((void *)output->dptr_, M * L);
#endif
        } 
        else {
            THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(dst, (int8_t *)output->dptr_, L, M), "luna_split_mat_trans_i8o8");
        }
    } 
    else if ((Int8 == weight->dtype_) & (Int8 == output->dtype_)) {
        THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o8)(p_weight, p_tmp, p_bias, dst, L, N, M, shift), "luna_split_mat_mul_bias_i8i8i32o8");
        if (ou_is_psram) {
            transpose2dInt8(dst, (int8_t *)output->dptr_, L, M);
#if !(defined(WIN32) || defined(linux))
            thinker_psram_write_complete((void *)output->dptr_, M * L);
#endif
        } 
        else {
            THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(dst, (int8_t *)output->dptr_, L, M), "luna_split_mat_trans_i8o8");
        }
    } 
    else if ((Int8 == weight->dtype_) & (Int32 == output->dtype_)) {
        if(shift < 0) {
            int32_t scale_out = 1UL<<(-shift);
            THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o32)(p_weight, p_tmp, p_bias, (int32_t *)dst, L, N, M, 0), "luna_split_mat_mul_bias_i8i8i32o32");
            THINKER_RET_CHECK(API_LIB(scale_i32i32o32)((int32_t *)dst, scale_out, (int32_t *)dst, M * L, 0), "luna_scale_i32i32o32");
        } 
        else {
            THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o32)(p_weight, p_tmp, p_bias, (int32_t *)dst, L, N, M, shift), "luna_split_mat_mul_bias_i8i8i32o32");
        }
        if (ou_is_psram) {
            transpose2dInt32((int32_t *)dst, (int32_t *)output->dptr_, L, M);
#if !(defined(WIN32) || defined(linux))
            thinker_psram_write_complete((void *)output->dptr_, M * L * 4);
#endif
        } 
        else {
            THINKER_RET_CHECK(API_LIB(split_mat_trans_i32o32)((int32_t *)dst, (int32_t *)output->dptr_, L, M), "luna_split_mat_trans_i32o32");
        }
    } else if ((Int32 == weight->dtype_) & (Int32 == output->dtype_)) {
        if(shift < 0) {
            int32_t scale_out = 1UL<<(-shift);
            THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i32i32i32o32)((int32_t *)p_weight, (int32_t *)p_tmp, p_bias, (int32_t *)dst, L, N, M, 0), "luna_split_mat_mul_bias_i32i32i32o32");
            THINKER_RET_CHECK(API_LIB(scale_i32i32o32)((int32_t *)dst, scale_out, (int32_t *)dst, M * L, 0), "luna_scale_i32i32o32");
        } 
        else {
            THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i32i32i32o32)((int32_t *)p_weight, (int32_t *)p_tmp, p_bias, (int32_t *)dst, L, N, M, shift), "luna_split_mat_mul_bias_i32i32i32o32");
        }
        if (ou_is_psram) {
            transpose2dInt32((int32_t *)dst, (int32_t *)output->dptr_, L, M);
#if !(defined(WIN32) || defined(linux))
            thinker_psram_write_complete((void *)output->dptr_, M * L * 4);
#endif
        } else {
            THINKER_RET_CHECK(API_LIB(split_mat_trans_i32o32)((int32_t *)dst, (int32_t *)output->dptr_, L, M), "luna_split_mat_trans_i32o32");
        }
    } else {
        return T_ERR_INVALID_DATATYPE;
    }

    return T_SUCCESS;
}

#endif
