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
 * @brief Perform batch matrix multiplication on integer data
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param out Output tensor
 * @param workspace Temporary workspace tensor
 * @return int32_t Operation status
 */
int32_t bmmint_luna(tTensor *lhs, tTensor *rhs, tTensor *out, tTensor *workspace) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    // Check input and output data types
    if (lhs->dtype_ != Int8 && lhs->dtype_ != Int16 && lhs->dtype_ != Int32) {
        return T_ERR_INVALID_DATATYPE;
    }
    if (out->dtype_ != Int8 && out->dtype_ != Int16 && out->dtype_ != Int32) {
        return T_ERR_INVALID_DATATYPE;
    }

    // Initialize dimensions and offsets
    int32_t batch = 1;
    int32_t n_dim = lhs->shape_.ndim_;
    int32_t M = lhs->shape_.dims_[n_dim - 2];
    int32_t N = lhs->shape_.dims_[n_dim - 1];
    int32_t L = rhs->shape_.dims_[n_dim - 1];
    int32_t src1_offset = M * N;
    int32_t src2_offset = N * L;
    int32_t dst_offset = M * L;

    // Calculate scale factors
    int32_t q_l = (int32_t)lhs->scale_;
    int32_t q_r = (int32_t)rhs->scale_;
    int32_t q_o = (int32_t)out->scale_;
    int32_t shift = q_l + q_r - q_o;

    if (shift < 0) {
        return T_ERR_FAIL;
    }

    if (n_dim == 3) {
        batch = lhs->shape_.dims_[0];
    }

    // Dispatch based on input and output data types
    if (lhs->dtype_ == Int8) {
        if (out->dtype_ == Int8) {
            for (int32_t i = 0; i < batch; ++i) {
                int8_t *tsrc1 = (int8_t *)lhs->dptr_ + i * src1_offset;
                int8_t *tsrc2 = (int8_t *)rhs->dptr_ + i * src2_offset;
                int8_t *tdst = (int8_t *)out->dptr_ + i * dst_offset;
                ret = API_LIB(split_mat_mul_i8i8o8)(tsrc1, tsrc2, tdst, M, N, L, shift);
            }
        } else if (out->dtype_ == Int16) {
            for (int32_t i = 0; i < batch; ++i) {
                int8_t *tsrc1 = (int8_t *)lhs->dptr_ + i * src1_offset;
                int8_t *tsrc2 = (int8_t *)rhs->dptr_ + i * src2_offset;
                int16_t *tdst = (int16_t *)out->dptr_ + i * dst_offset;
                ret = API_LIB(split_mat_mul_i8i8o16)(tsrc1, tsrc2, tdst, M, N, L, shift);
            }
        } else { // out->dtype_ == Int32
            for (int32_t i = 0; i < batch; ++i) {
                int8_t *tsrc1 = (int8_t *)lhs->dptr_ + i * src1_offset;
                int8_t *tsrc2 = (int8_t *)rhs->dptr_ + i * src2_offset;
                int32_t *tdst = (int32_t *)out->dptr_ + i * dst_offset;
                ret = API_LIB(split_mat_mul_i8i8o32)(tsrc1, tsrc2, tdst, M, N, L, shift);
            }
        }
    } else if (lhs->dtype_ == Int16) {
        if (out->dtype_ == Int8) {
            for (int32_t i = 0; i < batch; ++i) {
                int16_t *tsrc1 = (int16_t *)lhs->dptr_ + i * src1_offset;
                int16_t *tsrc2 = (int16_t *)rhs->dptr_ + i * src2_offset;
                int8_t *tdst = (int8_t *)out->dptr_ + i * dst_offset;
                ret = API_LIB(split_mat_mul_i16i16o8)(tsrc1, tsrc2, tdst, M, N, L, shift);
            }
        } else if (out->dtype_ == Int16) {
            for (int32_t i = 0; i < batch; ++i) {
                int16_t *tsrc1 = (int16_t *)lhs->dptr_ + i * src1_offset;
                int16_t *tsrc2 = (int16_t *)rhs->dptr_ + i * src2_offset;
                int16_t *tdst = (int16_t *)out->dptr_ + i * dst_offset;
                ret = API_LIB(split_mat_mul_i16i16o16)(tsrc1, tsrc2, tdst, M, N, L, shift);
            }
        } else { // out->dtype_ == Int32
            for (int32_t i = 0; i < batch; ++i) {
                int16_t *tsrc1 = (int16_t *)lhs->dptr_ + i * src1_offset;
                int16_t *tsrc2 = (int16_t *)rhs->dptr_ + i * src2_offset;
                int32_t *tdst = (int32_t *)out->dptr_ + i * dst_offset;
                ret = API_LIB(split_mat_mul_i16i16o32)(tsrc1, tsrc2, tdst, M, N, L, shift);
            }
        }
    } else { // lhs->dtype_ == Int32
        if (out->dtype_ == Int8) {
            for (int32_t i = 0; i < batch; ++i) {
                int32_t *tsrc1 = (int32_t *)lhs->dptr_ + i * src1_offset;
                int32_t *tsrc2 = (int32_t *)rhs->dptr_ + i * src2_offset;
                int8_t *tdst = (int8_t *)out->dptr_ + i * dst_offset;
                ret = API_LIB(split_mat_mul_i32i32o8)(tsrc1, tsrc2, tdst, M, N, L, shift);
            }
        } else if (out->dtype_ == Int16) {
            for (int32_t i = 0; i < batch; ++i) {
                int32_t *tsrc1 = (int32_t *)lhs->dptr_ + i * src1_offset;
                int32_t *tsrc2 = (int32_t *)rhs->dptr_ + i * src2_offset;
                int16_t *tdst = (int16_t *)out->dptr_ + i * dst_offset;
                ret = API_LIB(split_mat_mul_i32i32o16)(tsrc1, tsrc2, tdst, M, N, L, shift);
            }
        } else { // out->dtype_ == Int32
            for (int32_t i = 0; i < batch; ++i) {
                int32_t *tsrc1 = (int32_t *)lhs->dptr_ + i * src1_offset;
                int32_t *tsrc2 = (int32_t *)rhs->dptr_ + i * src2_offset;
                int32_t *tdst = (int32_t *)out->dptr_ + i * dst_offset;
                ret = API_LIB(split_mat_mul_i32i32o32)(tsrc1, tsrc2, tdst, M, N, L, shift);
            }
        }
    }

    return ret;
}

#endif  //_BMMINT_VENUS_H_