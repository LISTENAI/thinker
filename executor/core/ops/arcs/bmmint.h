#ifndef _BMMINT_VENUS_H_
#define _BMMINT_VENUS_H_

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief Execute batch matrix multiplication with integer precision
 * @param lhs Left-hand side tensor (batch of matrices)
 * @param rhs Right-hand side tensor (batch of matrices)
 * @param out Output tensor (batch of resulting matrices)
 * @param workspace Workspace tensor (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t bmmint_luna(tTensor *lhs, tTensor *rhs, tTensor *out, tTensor *workspace) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
    // Check input and output data types
    if (lhs->dtype_ != Int8 && lhs->dtype_ != Int32) {
        return T_ERR_INVALID_DATATYPE;
    }
    if (out->dtype_ != Int8 && out->dtype_ != Int32) {
        return T_ERR_INVALID_DATATYPE;
    }

    // Batch matrix multiplication parameters
    int32_t batch = 1;
    int32_t n_dim = lhs->shape_.ndim_;
    int32_t M = lhs->shape_.dims_[n_dim - 2]; // Number of rows in left matrix
    int32_t N = lhs->shape_.dims_[n_dim - 1]; // Number of columns in left matrix
    int32_t L = rhs->shape_.dims_[n_dim - 1]; // Number of columns in right matrix

    void *src1 = (void *)lhs->dptr_; // Pointer to left-hand side data
    void *src2 = (void *)rhs->dptr_; // Pointer to right-hand side data
    void *dst = (void *)out->dptr_;  // Pointer to output data

    int32_t src1_offset = M * N;  // Offset for left-hand side in batched mode
    int32_t src2_offset = N * L;  // Offset for right-hand side in batched mode
    int32_t dst_offset = M * L;   // Offset for output in batched mode

    int32_t q_l = (int32_t)lhs->scale_;  // Left-hand side quantization scale
    int32_t q_r = (int32_t)rhs->scale_;  // Right-hand side quantization scale
    int32_t q_o = (int32_t)out->scale_;  // Output quantization scale
    int32_t shift = q_l + q_r - q_o;     // Quantization shift value

    if (shift < 0) {
        return T_ERR_FAIL;
    }

    if (n_dim == 3) {
        batch = lhs->shape_.dims_[0]; // Batch size for 3D tensors
    }

    // Perform batched matrix multiplication based on data types
    if (lhs->dtype_ == Int8 && out->dtype_ == Int8) {
        for (int32_t i = 0; i < batch; i++) {
            int8_t *tsrc1 = (int8_t *)src1 + i * src1_offset;
            int8_t *tsrc2 = (int8_t *)src2 + i * src2_offset;
            int8_t *tdst = (int8_t *)dst + i * dst_offset;
            ret = API_LIB(split_mat_mul_i8i8o8)(tsrc1, tsrc2, tdst, M, N, L, shift);
        }
    } else if (lhs->dtype_ == Int8 && out->dtype_ == Int32) {
        for (int32_t i = 0; i < batch; i++) {
            int8_t *tsrc1 = (int8_t *)src1 + i * src1_offset;
            int8_t *tsrc2 = (int8_t *)src2 + i * src2_offset;
            int32_t *tdst = (int32_t *)dst + i * dst_offset;
            ret = API_LIB(split_mat_mul_i8i8o32)(tsrc1, tsrc2, tdst, M, N, L, shift);
        }
    } else if (lhs->dtype_ == Int32 && out->dtype_ == Int8) {
        for (int32_t i = 0; i < batch; i++) {
            int32_t *tsrc1 = (int32_t *)src1 + i * src1_offset;
            int32_t *tsrc2 = (int32_t *)src2 + i * src2_offset;
            int8_t *tdst = (int8_t *)dst + i * dst_offset;
            ret = API_LIB(split_mat_mul_i32i32o8)(tsrc1, tsrc2, tdst, M, N, L, shift);
        }
    } else {
        for (int32_t i = 0; i < batch; i++) {
            int32_t *tsrc1 = (int32_t *)src1 + i * src1_offset;
            int32_t *tsrc2 = (int32_t *)src2 + i * src2_offset;
            int32_t *tdst = (int32_t *)dst + i * dst_offset;
            ret = API_LIB(split_mat_mul_i32i32o32)(tsrc1, tsrc2, tdst, M, N, L, shift);
        }
    }

    return ret;
}

#endif  // _BMMINT_VENUS_H_