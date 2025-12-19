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
 * @brief Matrix multiplication function pointer type
 * @param src1 Pointer to first input matrix
 * @param src2 Pointer to second input matrix
 * @param dst Pointer to output matrix
 * @param row Number of rows in first matrix
 * @param col Number of columns in first matrix (and rows in second matrix)
 * @param col2 Number of columns in second matrix
 * @param shift Bit shift value for scaling
 * @return Operation status
 */
typedef int32_t (*BMM_MAT_MUL_LUNA_API)(void *src1, void *src2, void *dst,
                                        int32_t row, int32_t col, int32_t col2,
                                        int32_t shift);

/**
 * @brief Split matrix multiplication function pointer type
 * @param src1 Pointer to first input matrix
 * @param src2 Pointer to second input matrix
 * @param dst Pointer to output matrix
 * @param split_num Number of splits
 * @param row Number of rows in first matrix
 * @param col Number of columns in first matrix (and rows in second matrix)
 * @param col2 Number of columns in second matrix
 * @param shift Bit shift value for scaling
 * @return Operation status
 */
typedef int32_t (*BMM_SPLIT_MAT_MUL_LUNA_API)(void *src1, void *src2, void *dst,
                                              int32_t split_num, int32_t row,
                                              int32_t col, int32_t col2,
                                              int32_t shift);

/**
 * @brief Structure storing matrix multiplication API pointers
 */
struct luna_bmm_item {
    void *luna_api;
};

/**
 * @brief List of matrix multiplication APIs for different data types
 */
static struct luna_bmm_item luna_bmm_api_list[][3] = {
    {{API_LIB(mat_mul_q7_int8)}, {API_LIB(mat_mul_q7_int16)}, {API_LIB(mat_mul_q7_int32)}},
    {{API_LIB(mat_mul_q15_int8)},
     {API_LIB(mat_mul_q15_int16)},
     {API_LIB(mat_mul_q15_int32)}},
    {{API_LIB(mat_mul_q31_int8)},
     {API_LIB(mat_mul_q31_int16)},
     {API_LIB(mat_mul_q31_int32)}},
    {{API_LIB(split_mat_mul_q7_int8)},
     {API_LIB(split_mat_mul_q7_int16)},
     {API_LIB(split_mat_mul_q7_int32)}},
    {{API_LIB(split_mat_mul_q15_int8)},
     {API_LIB(split_mat_mul_q15_int16)},
     {API_LIB(split_mat_mul_q15_int32)}},
    {{API_LIB(split_mat_mul_q31_int8)},
     {API_LIB(split_mat_mul_q31_int16)},
     {API_LIB(split_mat_mul_q31_int32)}}
};

/**
 * @brief Calculate the ceiling of a division using bit shifting
 * @param x Numerator
 * @param shift Bit shift amount (equivalent to dividing by 2^shift)
 * @return Ceiling of x divided by 2^shift
 */
static int32_t luna_ceil(int32_t x, int32_t shift) {
    if (x & ~(0xFFFFFFFF << shift)) {
        return (x >> shift) + 1;
    } else {
        return (x >> shift);
    }
}

/**
 * @brief Perform batch matrix multiplication on quantized integer tensors
 * @param lhs First input tensor
 * @param rhs Second input tensor
 * @param out Output tensor
 * @param workspace Workspace buffer for temporary data
 * @return Operation status
 */
int32_t bmmint_luna(tTensor *lhs, tTensor *rhs, tTensor *out, tTensor *workspace) {
    int32_t ret = T_ERR_FAIL;

    if ((lhs->shape_.ndim_ != rhs->shape_.ndim_) || (lhs->dtype_ != rhs->dtype_)) {
        return ret;
    }

    const int32_t left_limit = 64 * 1024;
    const int32_t right_limit = 32 * 1024;
    int32_t batch = 1;
    int32_t lhs_is_psram = 0;
    int32_t rhs_is_psram = 0;
    int32_t out_is_psram = 0;
    int32_t n_dim = lhs->shape_.ndim_;
    int32_t in_idx = (lhs->dtype_ & 0xF) >> 1;
    int32_t ou_idx = (out->dtype_ & 0xF) >> 1;
    int32_t M = lhs->shape_.dims_[n_dim - 2];
    int32_t N = lhs->shape_.dims_[n_dim - 1];
    int32_t L = rhs->shape_.dims_[n_dim - 1];
    void *src1 = (void *)lhs->dptr_;
    void *src2 = (void *)rhs->dptr_;
    void *dst = (void *)out->dptr_;
    void *tmp_buf = NULL;
    int32_t workspace_size = 0;

    if (workspace != NULL) {
        tmp_buf = (void *)workspace->dptr_;
        workspace_size = workspace->shape_.dims_[0];
    }

    int32_t src1_offset = M * N * lhs->byte_;
    int32_t src2_offset = N * L * rhs->byte_;
    int32_t dst_offset = M * L * out->byte_;
    int32_t q_l = (int32_t)lhs->scale_;
    int32_t q_r = (int32_t)rhs->scale_;
    int32_t q_o = (int32_t)out->scale_;
    int32_t shift = q_l + q_r - q_o;

    if ((shift < 0) || (n_dim < 2) || (n_dim > 3)) {
        return ret;
    }

    if ((0 != in_idx) || (0 != ou_idx)) {
        return ret;
    }

    if (3 == n_dim) {
        batch = lhs->shape_.dims_[0];
    }

    if (2 != rhs->mem_.type_) {
        rhs_is_psram = 1;
    }

    if (2 != lhs->mem_.type_) {
        lhs_is_psram = 1;
    }

    if (2 != out->mem_.type_) {
        out_is_psram = 1;
    }

    for (int32_t i = 0; i < batch; i++) {
        int8_t *tsrc1 = (int8_t *)src1 + i * src1_offset;
        int8_t *tsrc2 = (int8_t *)src2 + i * src2_offset;
        int8_t *tdst = (int8_t *)dst + i * dst_offset;

        int8_t *p_tmp_src2 = (int8_t *)tsrc2;
        int32_t right_size = N * L;

        if (rhs_is_psram) {
            p_tmp_src2 = (int8_t *)tmp_buf;
            memcpy(p_tmp_src2, (int8_t *)tsrc2, right_size);
        }

        int32_t int8_condition_l = (luna_ceil(M, 2) << 2) * (luna_ceil(N, 3) << 3);
        int32_t int8_condition_r = (luna_ceil(N, 3) << 3) * (luna_ceil(L, 2) << 2);

        if (int8_condition_l > left_limit) {
            int s_num = 2;
            int32_t split_M = (0 == (M % s_num)) ? (M / s_num) : (M / s_num + 1);
            int32_t final_s_M = 0;
            int8_condition_l = (luna_ceil(split_M, 2) << 2) * (luna_ceil(N, 3) << 3);

            while (int8_condition_l > left_limit) {
                s_num++;
                split_M = (0 == (M % s_num)) ? (M / s_num) : (M / s_num + 1);
                int8_condition_l = (luna_ceil(split_M, 2) << 2) * (luna_ceil(N, 3) << 3);
            }

            final_s_M = (0 == (M % s_num)) ? (split_M) : (M - (split_M * (s_num - 1)));

            int32_t split_left_size = split_M * N;
            int32_t split_out_size = split_M * L;

            if (int8_condition_r <= right_limit) {
                BMM_MAT_MUL_LUNA_API luna_mat_mul_api =
                    (BMM_MAT_MUL_LUNA_API)luna_bmm_api_list[in_idx][ou_idx].luna_api;

                for (int32_t j = 0; j < s_num; j++) {
                    int8_t *p_tmp_src1 = (int8_t *)tsrc1 + split_left_size * j;
                    int8_t *p_tmp_dst = (int8_t *)tdst + split_out_size * j;
                    int32_t in_oft = split_left_size * j;
                    int32_t ou_oft = split_out_size * j;
                    int32_t tmp_size = 0;

                    if (j == (s_num - 1)) {
                        split_M = final_s_M;
                        split_left_size = split_M * N;
                        split_out_size = split_M * L;
                    }

                    if (lhs_is_psram) {
                        p_tmp_src1 = (rhs_is_psram == 1) ? ((int8_t *)tmp_buf + right_size) : (int8_t *)tmp_buf;
                        tmp_size = (rhs_is_psram == 1) ? (split_left_size + right_size) : split_left_size;
                        memcpy(p_tmp_src1, (int8_t *)tsrc1 + in_oft, split_left_size);
                    }

                    if (out_is_psram) {
                        p_tmp_dst = (int8_t *)tmp_buf + tmp_size;
                        tmp_size += split_out_size;
                        if (tmp_size > workspace_size) {
                            return -1;
                        }
                        ret = luna_mat_mul_api(p_tmp_src1, p_tmp_src2, p_tmp_dst, split_M, N, L, shift);
                        memcpy((int8_t *)tdst + ou_oft, p_tmp_dst, split_out_size);
                    } else {
                        ret = luna_mat_mul_api(p_tmp_src1, p_tmp_src2, p_tmp_dst, split_M, N, L, shift);
                    }
                }
            } else {
                int32_t split_num = 2;
                int32_t split_L = L / split_num;
                int8_condition_r = (luna_ceil(N, 3) << 3) * (luna_ceil(split_L, 2) << 2);

                while (int8_condition_r > right_limit || (0 != (L % split_num))) {
                    split_num++;
                    split_L = L / split_num;
                    int8_condition_r = (luna_ceil(N, 3) << 3) * (luna_ceil(split_L, 2) << 2);
                }

                BMM_SPLIT_MAT_MUL_LUNA_API luna_mat_mul_api =
                    (BMM_SPLIT_MAT_MUL_LUNA_API)luna_bmm_api_list[in_idx + 3][ou_idx].luna_api;

                for (int32_t j = 0; j < s_num; j++) {
                    int8_t *p_tmp_src1 = (int8_t *)tsrc1 + split_left_size * j;
                    int8_t *p_tmp_dst = (int8_t *)tdst + split_out_size * j;
                    int32_t in_oft = split_left_size * j;
                    int32_t ou_oft = split_out_size * j;
                    int32_t tmp_size = 0;

                    if (j == (s_num - 1)) {
                        split_M = final_s_M;
                        split_left_size = split_M * N;
                        split_out_size = split_M * L;
                    }

                    if (lhs_is_psram) {
                        p_tmp_src1 = (rhs_is_psram == 1) ? ((int8_t *)tmp_buf + right_size) : (int8_t *)tmp_buf;
                        tmp_size = (rhs_is_psram == 1) ? (split_left_size + right_size) : split_left_size;
                        memcpy(p_tmp_src1, (int8_t *)tsrc1 + in_oft, split_left_size);
                    }

                    if (out_is_psram) {
                        p_tmp_dst = (int8_t *)tmp_buf + tmp_size;
                        tmp_size += split_out_size;
                        if (tmp_size > workspace_size) {
                            return -1;
                        }
                        ret = luna_mat_mul_api(p_tmp_src1, p_tmp_src2, p_tmp_dst, split_num, split_M, N, L, shift);
                        memcpy((int8_t *)tdst + ou_oft, p_tmp_dst, split_out_size);
                    } else {
                        ret = luna_mat_mul_api(p_tmp_src1, p_tmp_src2, p_tmp_dst, split_num, split_M, N, L, shift);
                    }
                }
            }
        } else {
            int8_t *p_tmp_src1 = (int8_t *)tsrc1;
            int8_t *p_tmp_dst = (int8_t *)tdst;
            int32_t tmp_size = 0;

            if (lhs_is_psram) {
                p_tmp_src1 = (rhs_is_psram == 1) ? ((int8_t *)tmp_buf + right_size) : (int8_t *)tmp_buf;
                tmp_size = (rhs_is_psram == 1) ? (M * N + right_size) : M * N;
                memcpy(p_tmp_src1, tsrc1, M * N);
            }

            if (out_is_psram) {
                p_tmp_dst = (int8_t *)tmp_buf + tmp_size;
                tmp_size += M * L;
                if (tmp_size > workspace_size) {
                    printf("workspace exceed!\n");
                    return -1;
                }
            }

            if (int8_condition_r <= right_limit) {
                BMM_MAT_MUL_LUNA_API luna_mat_mul_api =
                    (BMM_MAT_MUL_LUNA_API)luna_bmm_api_list[in_idx][ou_idx].luna_api;
                ret = luna_mat_mul_api(p_tmp_src1, p_tmp_src2, p_tmp_dst, M, N, L, shift);
            } else {
                int32_t split_num = 2;
                int32_t split_L = L / split_num;
                int8_condition_r = (luna_ceil(N, 3) << 3) * (luna_ceil(split_L, 2) << 2);

                while (int8_condition_r > right_limit || (0 != (L % split_num))) {
                    split_num++;
                    split_L = L / split_num;
                    int8_condition_r = (luna_ceil(N, 3) << 3) * (luna_ceil(split_L, 2) << 2);
                }

                BMM_SPLIT_MAT_MUL_LUNA_API luna_mat_mul_api =
                    (BMM_SPLIT_MAT_MUL_LUNA_API)luna_bmm_api_list[in_idx + 3][ou_idx].luna_api;
                ret = luna_mat_mul_api(p_tmp_src1, p_tmp_src2, p_tmp_dst, split_num, M, N, L, shift);
            }

            if (out_is_psram) {
                memcpy(tdst, p_tmp_dst, M * L);
            }
        }
    }

    return ret;
}

#endif  //_BMMINT_VENUS_H_