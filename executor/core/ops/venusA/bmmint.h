#ifndef _BMMINT_VENUSA_H_
#define _BMMINT_VENUSA_H_

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
 * @brief Perform batch matrix multiplication on integer data
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param out Output tensor
 * @param workspace Temporary workspace tensor
 * @return int32_t Operation status
 */
int32_t bmmint_luna(tTensor *lhs, tTensor *rhs, tTensor *out, tTensor *workspace) {
    #if THINKER_PARAM_CHECK
    if (lhs == NULL || rhs == NULL || out == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if ((lhs->dtype_ != Int8 && lhs->dtype_ != Int16 && lhs->dtype_ != Int32) ||
                        rhs->dtype_ != lhs->dtype_ ||
                        (out->dtype_ != Int8 && out->dtype_ != Int16 && out->dtype_ != Int32)) {
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
    int32_t M = lhs->shape_.dims_[n_dim - 2];
    int32_t N = lhs->shape_.dims_[n_dim - 1];
    int32_t L = rhs->shape_.dims_[n_dim - 1];
    int32_t src1_offset = M * N;
    int32_t src2_offset = N * L;
    int32_t dst_offset = M * L;
    #if THINKER_RUNTIME_CHECK
    if (M <= 0 || N <= 0 || L <= 0 ||
                          rhs->shape_.dims_[n_dim - 2] != N ||
                          out->shape_.dims_[n_dim - 2] != M ||
                          out->shape_.dims_[n_dim - 1] != L || lhs->dptr_ == 0 ||
                          rhs->dptr_ == 0 || out->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    #if THINKER_RUNTIME_CHECK
    if (out->dptr_ == lhs->dptr_ || out->dptr_ == rhs->dptr_) {
        return (T_ERR_NO_SUPPORT_OP);
    }
#endif

    // Calculate scale factors
    int32_t q_l = (int32_t)lhs->scale_;
    int32_t q_r = (int32_t)rhs->scale_;
    int32_t q_o = (int32_t)out->scale_;
    int64_t shift_value = (int64_t)q_l + q_r - q_o;
    int32_t y_in_psram = (out->mem_.type_ != 2);
    int32_t output_size = dst_offset * out->byte_;

    #if THINKER_PARAM_CHECK
    if (shift_value < 0 || shift_value > 63) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    int32_t shift = (int32_t)shift_value;

    if (n_dim == 3) {
        batch = lhs->shape_.dims_[0];
        #if THINKER_RUNTIME_CHECK
        if (batch <= 0 || rhs->shape_.dims_[0] != batch ||
                              out->shape_.dims_[0] != batch) {
            return (T_ERR_INVALID_PARA);
        }
#endif
    }

    #if THINKER_RUNTIME_CHECK
    if (y_in_psram &&
                          (workspace == NULL || workspace->dptr_ == 0 ||
                           workspace->mem_.type_ != 2 || workspace->dtype_ != Int8 ||
                           workspace->shape_.ndim_ != 1 ||
                           workspace->shape_.dims_[0] < (uint32_t)output_size)) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif

    // Dispatch based on input and output data types
    if (lhs->dtype_ == Int8) {
        if (out->dtype_ == Int8) {
            for (int32_t i = 0; i < batch; ++i) {
                int8_t *tsrc1 = (int8_t *)lhs->dptr_ + i * src1_offset;
                int8_t *tsrc2 = (int8_t *)rhs->dptr_ + i * src2_offset;
                int8_t *tdst = y_in_psram ? (int8_t *)workspace->dptr_ : (int8_t *)out->dptr_ + i * dst_offset;
                THINKER_RET_CHECK(API_LIB(split_mat_mul_i8i8o8)(tsrc1, tsrc2, tdst, M, N, L, shift), "luna_split_mat_mul_i8i8o8");
                if (y_in_psram)
                    opi_psram_cpy_out((int8_t *)out->dptr_ + i * dst_offset, tdst, output_size);
            }
        } else if (out->dtype_ == Int16) {
            for (int32_t i = 0; i < batch; ++i) {
                int8_t *tsrc1 = (int8_t *)lhs->dptr_ + i * src1_offset;
                int8_t *tsrc2 = (int8_t *)rhs->dptr_ + i * src2_offset;
                int16_t *tdst = y_in_psram ? (int16_t *)workspace->dptr_ : (int16_t *)out->dptr_ + i * dst_offset;
                THINKER_RET_CHECK(API_LIB(split_mat_mul_i8i8o16)(tsrc1, tsrc2, tdst, M, N, L, shift), "luna_split_mat_mul_i8i8o16");
                if (y_in_psram)
                    opi_psram_cpy_out((int16_t *)out->dptr_ + i * dst_offset, tdst, output_size);
            }
        } else { // out->dtype_ == Int32
            for (int32_t i = 0; i < batch; ++i) {
                int8_t *tsrc1 = (int8_t *)lhs->dptr_ + i * src1_offset;
                int8_t *tsrc2 = (int8_t *)rhs->dptr_ + i * src2_offset;
                int32_t *tdst = y_in_psram ? (int32_t *)workspace->dptr_ : (int32_t *)out->dptr_ + i * dst_offset;
                THINKER_RET_CHECK(API_LIB(split_mat_mul_i8i8o32)(tsrc1, tsrc2, tdst, M, N, L, shift), "luna_split_mat_mul_i8i8o32");
                if (y_in_psram)
                    opi_psram_cpy_out((int32_t *)out->dptr_ + i * dst_offset, tdst, output_size);
            }
        }
    } else if (lhs->dtype_ == Int16) {
        if (out->dtype_ == Int8) {
            for (int32_t i = 0; i < batch; ++i) {
                int16_t *tsrc1 = (int16_t *)lhs->dptr_ + i * src1_offset;
                int16_t *tsrc2 = (int16_t *)rhs->dptr_ + i * src2_offset;
                int8_t *tdst = y_in_psram ? (int8_t *)workspace->dptr_ : (int8_t *)out->dptr_ + i * dst_offset;
                THINKER_RET_CHECK(API_LIB(split_mat_mul_i16i16o8)(tsrc1, tsrc2, tdst, M, N, L, shift), "luna_split_mat_mul_i16i16o8");
                if (y_in_psram)
                    opi_psram_cpy_out((int8_t *)out->dptr_ + i * dst_offset, tdst, output_size);
            }
        } else if (out->dtype_ == Int16) {
            for (int32_t i = 0; i < batch; ++i) {
                int16_t *tsrc1 = (int16_t *)lhs->dptr_ + i * src1_offset;
                int16_t *tsrc2 = (int16_t *)rhs->dptr_ + i * src2_offset;
                int16_t *tdst = y_in_psram ? (int16_t *)workspace->dptr_ : (int16_t *)out->dptr_ + i * dst_offset;
                THINKER_RET_CHECK(API_LIB(split_mat_mul_i16i16o16)(tsrc1, tsrc2, tdst, M, N, L, shift), "luna_split_mat_mul_i16i16o16");
                if (y_in_psram)
                    opi_psram_cpy_out((int16_t *)out->dptr_ + i * dst_offset, tdst, output_size);
            }
        } else { // out->dtype_ == Int32
            for (int32_t i = 0; i < batch; ++i) {
                int16_t *tsrc1 = (int16_t *)lhs->dptr_ + i * src1_offset;
                int16_t *tsrc2 = (int16_t *)rhs->dptr_ + i * src2_offset;
                int32_t *tdst = y_in_psram ? (int32_t *)workspace->dptr_ : (int32_t *)out->dptr_ + i * dst_offset;
                THINKER_RET_CHECK(API_LIB(split_mat_mul_i16i16o32)(tsrc1, tsrc2, tdst, M, N, L, shift), "luna_split_mat_mul_i16i16o32");
                if (y_in_psram)
                    opi_psram_cpy_out((int32_t *)out->dptr_ + i * dst_offset, tdst, output_size);
            }
        }
    } else { // lhs->dtype_ == Int32
        if (out->dtype_ == Int8) {
            for (int32_t i = 0; i < batch; ++i) {
                int32_t *tsrc1 = (int32_t *)lhs->dptr_ + i * src1_offset;
                int32_t *tsrc2 = (int32_t *)rhs->dptr_ + i * src2_offset;
                int8_t *tdst = y_in_psram ? (int8_t *)workspace->dptr_ : (int8_t *)out->dptr_ + i * dst_offset;
                THINKER_RET_CHECK(API_LIB(split_mat_mul_i32i32o8)(tsrc1, tsrc2, tdst, M, N, L, shift), "luna_split_mat_mul_i32i32o8");
                if (y_in_psram)
                    opi_psram_cpy_out((int8_t *)out->dptr_ + i * dst_offset, tdst, output_size);
            }
        } else if (out->dtype_ == Int16) {
            for (int32_t i = 0; i < batch; ++i) {
                int32_t *tsrc1 = (int32_t *)lhs->dptr_ + i * src1_offset;
                int32_t *tsrc2 = (int32_t *)rhs->dptr_ + i * src2_offset;
                int16_t *tdst = y_in_psram ? (int16_t *)workspace->dptr_ : (int16_t *)out->dptr_ + i * dst_offset;
                THINKER_RET_CHECK(API_LIB(split_mat_mul_i32i32o16)(tsrc1, tsrc2, tdst, M, N, L, shift), "luna_split_mat_mul_i32i32o16");
                if (y_in_psram)
                    opi_psram_cpy_out((int16_t *)out->dptr_ + i * dst_offset, tdst, output_size);
            }
        } else { // out->dtype_ == Int32
            for (int32_t i = 0; i < batch; ++i) {
                int32_t *tsrc1 = (int32_t *)lhs->dptr_ + i * src1_offset;
                int32_t *tsrc2 = (int32_t *)rhs->dptr_ + i * src2_offset;
                int32_t *tdst = y_in_psram ? (int32_t *)workspace->dptr_ : (int32_t *)out->dptr_ + i * dst_offset;
                THINKER_RET_CHECK(API_LIB(split_mat_mul_i32i32o32)(tsrc1, tsrc2, tdst, M, N, L, shift), "luna_split_mat_mul_i32i32o32");
                if (y_in_psram)
                    opi_psram_cpy_out((int32_t *)out->dptr_ + i * dst_offset, tdst, output_size);
            }
        }
    }

    return T_SUCCESS;
}

#endif  //_BMMINT_VENUS_H_
