#ifndef _BMMINT_ARCS_H_
#define _BMMINT_ARCS_H_

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
 * @param workspace SRAM temporary output when out is in PSRAM
 * @return int32_t Execution status
 */
int32_t bmmint_luna(tTensor *lhs, tTensor *rhs, tTensor *out, tTensor *workspace) {
    #if THINKER_PARAM_CHECK
    if (lhs == NULL || rhs == NULL || out == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if ((lhs->dtype_ != Int8 && lhs->dtype_ != Int32) ||
                        rhs->dtype_ != lhs->dtype_ ||
                        (out->dtype_ != Int8 && out->dtype_ != Int32)) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    int32_t batch = 1;
    int32_t n_dim = lhs->shape_.ndim_;
    #if THINKER_PARAM_CHECK
    if (n_dim < 2 || n_dim > 3 || rhs->shape_.ndim_ != n_dim ||
                        out->shape_.ndim_ != n_dim) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    int32_t M = lhs->shape_.dims_[n_dim - 2]; // Number of rows in left matrix
    int32_t N = lhs->shape_.dims_[n_dim - 1]; // Number of columns in left matrix
    int32_t L = rhs->shape_.dims_[n_dim - 1]; // Number of columns in right matrix
    #if THINKER_RUNTIME_CHECK
    if (M <= 0 || N <= 0 || L <= 0 ||
                          rhs->shape_.dims_[n_dim - 2] != N ||
                          out->shape_.dims_[n_dim - 2] != M ||
                          out->shape_.dims_[n_dim - 1] != L || lhs->dptr_ == 0 ||
                          rhs->dptr_ == 0 || out->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    void *src1 = (void *)lhs->dptr_; // Pointer to left-hand side data
    void *src2 = (void *)rhs->dptr_; // Pointer to right-hand side data
    void *dst = (void *)out->dptr_;  // Pointer to output data
    #if THINKER_RUNTIME_CHECK
    if (out->dptr_ == lhs->dptr_ || out->dptr_ == rhs->dptr_) {
        return (T_ERR_NO_SUPPORT_OP);
    }
#endif
    int32_t out_is_psram = (out->mem_.type_ != 2);

    int32_t src1_offset = M * N;  // Offset for left-hand side in batched mode
    int32_t src2_offset = N * L;  // Offset for right-hand side in batched mode
    int32_t dst_offset = M * L;   // Offset for output in batched mode
    int32_t dst_batch_bytes = dst_offset * out->byte_;
    #if THINKER_RUNTIME_CHECK
    if (out_is_psram &&
                          (workspace == NULL || workspace->dptr_ == 0 ||
                           workspace->mem_.type_ != 2 || workspace->dtype_ != Int8 ||
                           workspace->shape_.ndim_ != 1 ||
                           workspace->shape_.dims_[0] < (uint32_t)dst_batch_bytes)) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif
    void *dst_tmp = out_is_psram ? (void *)workspace->dptr_ : dst;

    int32_t q_l = (int32_t)lhs->scale_;  // Left-hand side quantization scale
    int32_t q_r = (int32_t)rhs->scale_;  // Right-hand side quantization scale
    int32_t q_o = (int32_t)out->scale_;  // Output quantization scale
    int64_t shift_value = (int64_t)q_l + q_r - q_o;
    int32_t shift;

    #if THINKER_PARAM_CHECK
    if (shift_value < 0 || shift_value > 63) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    shift = (int32_t)shift_value;

    if (n_dim == 3) {
        batch = lhs->shape_.dims_[0]; // Batch size for 3D tensors
        #if THINKER_RUNTIME_CHECK
        if (batch <= 0 || rhs->shape_.dims_[0] != batch ||
                              out->shape_.dims_[0] != batch) {
            return (T_ERR_INVALID_PARA);
        }
#endif
    }

    // Perform batched matrix multiplication based on data types
    if (lhs->dtype_ == Int8 && out->dtype_ == Int8) {
        for (int32_t i = 0; i < batch; i++) {
            int8_t *tsrc1 = (int8_t *)src1 + i * src1_offset;
            int8_t *tsrc2 = (int8_t *)src2 + i * src2_offset;
            int8_t *tdst = out_is_psram ? (int8_t *)dst_tmp : (int8_t *)dst + i * dst_offset;
            THINKER_RET_CHECK(API_LIB(split_mat_mul_i8i8o8)(tsrc1, tsrc2, tdst, M, N, L, shift), "luna_split_mat_mul_i8i8o8");
            if (out_is_psram) {
                opi_psram_cpy_out((int8_t *)dst + i * dst_batch_bytes, (int8_t *)tdst, dst_batch_bytes);
            }
        }
    } else if (lhs->dtype_ == Int8 && out->dtype_ == Int32) {
        for (int32_t i = 0; i < batch; i++) {
            int8_t *tsrc1 = (int8_t *)src1 + i * src1_offset;
            int8_t *tsrc2 = (int8_t *)src2 + i * src2_offset;
            int32_t *tdst = out_is_psram ? (int32_t *)dst_tmp : (int32_t *)dst + i * dst_offset;
            THINKER_RET_CHECK(API_LIB(split_mat_mul_i8i8o32)(tsrc1, tsrc2, tdst, M, N, L, shift), "luna_split_mat_mul_i8i8o32");
            if (out_is_psram) {
                opi_psram_cpy_out((int8_t *)dst + i * dst_batch_bytes, (int8_t *)tdst, dst_batch_bytes);
            }
        }
    } else if (lhs->dtype_ == Int32 && out->dtype_ == Int8) {
        for (int32_t i = 0; i < batch; i++) {
            int32_t *tsrc1 = (int32_t *)src1 + i * src1_offset;
            int32_t *tsrc2 = (int32_t *)src2 + i * src2_offset;
            int8_t *tdst = out_is_psram ? (int8_t *)dst_tmp : (int8_t *)dst + i * dst_offset;
            THINKER_RET_CHECK(API_LIB(split_mat_mul_i32i32o8)(tsrc1, tsrc2, tdst, M, N, L, shift), "luna_split_mat_mul_i32i32o8");
            if (out_is_psram) {
                opi_psram_cpy_out((int8_t *)dst + i * dst_batch_bytes, (int8_t *)tdst, dst_batch_bytes);
            }
        }
    } else {
        for (int32_t i = 0; i < batch; i++) {
            int32_t *tsrc1 = (int32_t *)src1 + i * src1_offset;
            int32_t *tsrc2 = (int32_t *)src2 + i * src2_offset;
            int32_t *tdst = out_is_psram ? (int32_t *)dst_tmp : (int32_t *)dst + i * dst_offset;
            THINKER_RET_CHECK(API_LIB(split_mat_mul_i32i32o32)(tsrc1, tsrc2, tdst, M, N, L, shift), "luna_split_mat_mul_i32i32o32");
            if (out_is_psram) {
                opi_psram_cpy_out((int8_t *)dst + i * dst_batch_bytes, (int8_t *)tdst, dst_batch_bytes);
            }
        }
    }

    return T_SUCCESS;
}

#endif  // _BMMINT_ARCS_H_
