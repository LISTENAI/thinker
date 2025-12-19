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

/**
 * @brief Matrix multiplication API for integer quantized tensors
 */
typedef int32_t (*FC_MAT_MUL_LUNA_API)(int8_t *src1, int8_t *src2, void *dst,
                                       int32_t row, int32_t col, int32_t col2,
                                       int32_t shift);

/**
 * @brief Split matrix multiplication API for integer quantized tensors
 */
typedef int32_t (*FC_SPLIT_MAT_MUL_LUNA_API)(int8_t *src1, int8_t *src2,
                                            void *dst, int32_t split_num,
                                            int32_t row, int32_t col,
                                            int32_t col2, int32_t shift);

/**
 * @brief Vector addition API for integer quantized tensors
 */
typedef int32_t (*FC_VEC_ADD_LUNA_API)(void *src1, void *src2, void *dst,
                                       int32_t size, int32_t shift);

/**
 * @brief Structure to hold Luna API function pointers
 */
struct fc_luna_api_item {
    void *luna_api;
};

/**
 * @brief List of Luna API functions for different data types
 */
static struct fc_luna_api_item fc_luna_api_list[][3] = {
    {{API_LIB(mat_mul_q7_int8)}, {API_LIB(mat_mul_q7_int16)}, {API_LIB(mat_mul_q7_int32)}},
    {{API_LIB(split_mat_mul_q7_int8)}, {API_LIB(split_mat_mul_q7_int16)}, {API_LIB(split_mat_mul_q7_int32)}},
    {{API_LIB(add_q7_int8)}, {API_LIB(add_q7_int16)}, {API_LIB(add_q7_int32)}},
    {{API_LIB(add_q15_int8)}, {API_LIB(add_q15_int16)}, {API_LIB(add_q15_int32)}},
    {{API_LIB(add_q31_int8)}, {API_LIB(add_q31_int16)}, {API_LIB(add_q31_int32)}}
};

/**
 * @brief Matrix multiplication API for 16-bit integer quantized tensors
 */
typedef int32_t (*FC_MAT_MUL_LUNA_INT16_API)(int16_t *src1, int16_t *src2, void *dst,
                                            int32_t row, int32_t col, int32_t col2,
                                            int32_t shift);

/**
 * @brief Split matrix multiplication API for 16-bit integer quantized tensors
 */
typedef int32_t (*FC_SPLIT_MAT_MUL_LUNA_INT16_API)(int16_t *src1, int16_t *src2,
                                                  void *dst, int32_t split_num,
                                                  int32_t row, int32_t col,
                                                  int32_t col2, int32_t shift);

/**
 * @brief List of Luna API functions for 16-bit data types
 */
static struct fc_luna_api_item fc_luna_int16_api_list[][3] = {
    {{API_LIB(mat_mul_q15_int8)}, {API_LIB(mat_mul_q15_int16)}, {API_LIB(mat_mul_q15_int32)}},
    {{API_LIB(split_mat_mul_q15_int8)}, {API_LIB(split_mat_mul_q15_int16)}, {API_LIB(split_mat_mul_q15_int32)}},
    {{API_LIB(add_q7_int8)}, {API_LIB(add_q7_int16)}, {API_LIB(add_q7_int32)}},
    {{API_LIB(add_q15_int8)}, {API_LIB(add_q15_int16)}, {API_LIB(add_q15_int32)}},
    {{API_LIB(add_q31_int8)}, {API_LIB(add_q31_int16)}, {API_LIB(add_q31_int32)}}
};

/**
 * @brief Ceiling function for integer division
 * @param x Input integer
 * @param shift Number of bits to shift
 * @return int32_t Result after ceiling operation
 */
static int32_t luna_ceil(int32_t x, int32_t shift) {
    if (x & ~(0xFFFFFFFF << shift)) {
        return (x >> shift) + 1;
    } else {
        return (x >> shift);
    }
}

/**
 * @brief Perform linear transformation with integer quantization
 * @param input Input tensor
 * @param weight Weight tensor
 * @param bias Bias tensor (optional)
 * @param output Output tensor
 * @param tmp Temporary workspace tensor
 * @return int32_t Operation status
 */
static int32_t calc_linearint_luna(tTensor *input, tTensor *weight,
                                   tTensor *bias, tTensor *output,
                                   tTensor *tmp) {
    int32_t ret = T_ERR_FAIL;
    tShape new_shape;

    // Adjust input shape if necessary
    if (1 == input->shape_.ndim_) {
        new_shape.ndim_ = 2;
        new_shape.dims_[1] = input->shape_.dims_[0];
        new_shape.dims_[0] = 1;
    } else if (input->shape_.ndim_ == 3) {
        new_shape.ndim_ = 2;
        new_shape.dims_[0] = input->shape_.dims_[0] * input->shape_.dims_[1];
        new_shape.dims_[1] = input->shape_.dims_[2];
    } else {
        memcpy(&new_shape, &input->shape_, sizeof(tShape));
    }

    const int32_t left_limit = 64 * 1024;
    const int32_t right_limit = 32 * 1024;
    int32_t in_is_psram = 0;
    int32_t ou_is_psram = 0;
    int32_t ou_idx = (output->dtype_ & 0xF) >> 1;
    int32_t bias_idx = 0;
    int32_t n_dim = new_shape.ndim_;
    int32_t M = new_shape.dims_[n_dim - 2];
    int32_t N = new_shape.dims_[n_dim - 1];
    int32_t L = weight->shape_.dims_[n_dim - 1];
    int8_t *p_in = (int8_t *)input->dptr_;
    int8_t *p_weight = (int8_t *)weight->dptr_;
    void *p_out = (void *)output->dptr_;
    void *p_bis = NULL;
    int32_t workspace_size = 0;

    if (tmp != NULL) {
        workspace_size = tmp->shape_.dims_[0];
    }

    int32_t q_i = (int32_t)input->scale_;
    int32_t q_w = (int32_t)weight->scale_;
    int32_t q_o = (int32_t)output->scale_;
    int32_t shift = q_i + q_w - q_o;
    int32_t has_bias = 0;

    // Check if input or output is in PSRAM
    if (input->mem_.type_ == 1 || input->mem_.type_ == 3) {
        in_is_psram = 1;
    }
    if (output->mem_.type_ == 1 || output->mem_.type_ == 3) {
        ou_is_psram = 1;
    }

    // Handle bias tensor if present
    if (bias != NULL) {
        bias_idx = (bias->dtype_ & 0xF) >> 1;
        p_bis = (void *)bias->dptr_;
        has_bias = (bias->shape_.ndim_ != 0) ? 1 : 0;
    }

    if (shift < 0) {
        return ret;
    }

    switch (input->dtype_) {
        case Int8: {
            int32_t int8_condition_l = (luna_ceil(M, 2) << 2) * (luna_ceil(N, 3) << 3);
            int32_t int8_condition_r = (luna_ceil(N, 3) << 3) * (luna_ceil(L, 2) << 2);

            if (int8_condition_l > left_limit) {
                int s_num = 2;
                int32_t split_M = (M % s_num) ? (M / s_num + 1) : (M / s_num);
                int32_t final_s_M = 0;
                int32_t split_left_size = split_M * N;
                int32_t split_out_size = split_M * L;

                while (int8_condition_l > left_limit) {
                    s_num++;
                    split_M = (M % s_num) ? (M / s_num + 1) : (M / s_num);
                    int8_condition_l = (luna_ceil(split_M, 2) << 2) * (luna_ceil(N, 3) << 3);
                }
                final_s_M = (M % s_num) ? (M - (split_M * (s_num - 1))) : split_M;

                if (int8_condition_r <= right_limit) {
                    FC_MAT_MUL_LUNA_API luna_mat_mul_api;
                    for (int32_t i = 0; i < s_num; i++) {
                        int8_t *p_tmp_in = p_in + split_left_size * i;
                        int8_t *p_tmp_ou = (int8_t *)p_out + split_out_size * i;
                        int32_t in_oft = split_left_size * i;
                        int32_t ou_oft = split_out_size * i;
                        int32_t tmp_size = 0;

                        if (i == s_num - 1) {
                            split_M = final_s_M;
                            split_left_size = split_M * N;
                            split_out_size = split_M * L;
                        }

                        if (in_is_psram) {
                            p_tmp_in = (int8_t *)tmp->dptr_;
                            tmp_size = (split_left_size > split_out_size) ? split_left_size : split_out_size;
                            memcpy(p_tmp_in, (int8_t *)p_in + in_oft, split_left_size);
                        }

                        if (ou_is_psram) {
                            if (has_bias) {
                                int32_t *p_tmp = (int32_t *)((int8_t *)tmp->dptr_ + tmp_size);
                                p_tmp_ou = p_tmp_in;
                                tmp_size += split_out_size * 4;
                                if (tmp_size > workspace_size) {
                                    return -1;
                                }
                                luna_mat_mul_api = (FC_MAT_MUL_LUNA_API)fc_luna_api_list[0][bias_idx].luna_api;
                                ret = luna_mat_mul_api(p_tmp_in, p_weight, p_tmp, split_M, N, L, 0);

                                for (int32_t j = 0; j < split_M; j++) {
                                    FC_VEC_ADD_LUNA_API luna_add_api = (FC_VEC_ADD_LUNA_API)fc_luna_api_list[2 + bias_idx][ou_idx].luna_api;
                                    int8_t *tsrc1 = (int8_t *)p_tmp + j * L * (bias->dtype_ & 0xF);
                                    int8_t *tdst = (int8_t *)p_tmp_ou + j * L * (output->dtype_ & 0xF);
                                    ret |= luna_add_api(tsrc1, p_bis, tdst, L, shift);
                                }
                                memcpy(p_out + ou_oft, p_tmp_ou, split_out_size);
                            } else {
                                p_tmp_ou = (int8_t *)tmp->dptr_ + tmp_size;
                                tmp_size += split_out_size;
                                if (tmp_size > workspace_size) {
                                    return -1;
                                }
                                luna_mat_mul_api = (FC_MAT_MUL_LUNA_API)fc_luna_api_list[0][ou_idx].luna_api;
                                ret = luna_mat_mul_api(p_tmp_in, p_weight, p_tmp_ou, split_M, N, L, shift);
                                memcpy(p_out + ou_oft, p_tmp_ou, split_out_size);
                            }
                        } else {
                            if (has_bias) {
                                int32_t *p_tmp = (int32_t *)((int8_t *)tmp->dptr_ + tmp_size);
                                luna_mat_mul_api = (FC_MAT_MUL_LUNA_API)fc_luna_api_list[0][bias_idx].luna_api;
                                ret = luna_mat_mul_api(p_tmp_in, p_weight, p_tmp, split_M, N, L, 0);

                                for (int32_t j = 0; j < split_M; j++) {
                                    FC_VEC_ADD_LUNA_API luna_add_api = (FC_VEC_ADD_LUNA_API)fc_luna_api_list[2 + bias_idx][ou_idx].luna_api;
                                    int8_t *tsrc1 = (int8_t *)p_tmp + j * L * (bias->dtype_ & 0xF);
                                    int8_t *tdst = (int8_t *)p_tmp_ou + j * L * (output->dtype_ & 0xF);
                                    ret |= luna_add_api(tsrc1, p_bis, tdst, L, shift);
                                }
                            } else {
                                luna_mat_mul_api = (FC_MAT_MUL_LUNA_API)fc_luna_api_list[0][ou_idx].luna_api;
                                ret = luna_mat_mul_api(p_tmp_in, p_weight, p_tmp_ou, split_M, N, L, shift);
                            }
                        }
                    }
                } else {
                    int32_t split_num = 2;
                    int32_t split_L = L / split_num;
                    while (int8_condition_r > right_limit || (L % split_num != 0)) {
                        split_num++;
                        split_L = L / split_num;
                        int8_condition_r = (luna_ceil(N, 3) << 3) * (luna_ceil(split_L, 2) << 2);
                    }

                    FC_SPLIT_MAT_MUL_LUNA_API luna_mat_mul_api;
                    for (int32_t i = 0; i < s_num; i++) {
                        int8_t *p_tmp_in = p_in + split_left_size * i;
                        int8_t *p_tmp_ou = (int8_t *)p_out + split_out_size * i;
                        int32_t in_oft = split_left_size * i;
                        int32_t ou_oft = split_out_size * i;
                        int32_t tmp_size = 0;

                        if (i == s_num - 1) {
                            split_M = final_s_M;
                            split_left_size = split_M * N;
                            split_out_size = split_M * L;
                        }

                        if (in_is_psram) {
                            p_tmp_in = (int8_t *)tmp->dptr_;
                            tmp_size = (split_left_size > split_out_size) ? split_left_size : split_out_size;
                            memcpy(p_tmp_in, (int8_t *)p_in + in_oft, split_left_size);
                        }

                        if (ou_is_psram) {
                            if (has_bias) {
                                int32_t *p_tmp = (int32_t *)((int8_t *)tmp->dptr_ + tmp_size);
                                p_tmp_ou = p_tmp_in;
                                tmp_size += split_out_size * 4;
                                if (tmp_size > workspace_size) {
                                    return -1;
                                }
                                luna_mat_mul_api = (FC_SPLIT_MAT_MUL_LUNA_API)fc_luna_api_list[1][bias_idx].luna_api;
                                ret = luna_mat_mul_api(p_tmp_in, p_weight, p_tmp, split_num, split_M, N, L, 0);

                                for (int32_t j = 0; j < split_M; j++) {
                                    FC_VEC_ADD_LUNA_API luna_add_api = (FC_VEC_ADD_LUNA_API)fc_luna_api_list[2 + bias_idx][ou_idx].luna_api;
                                    int8_t *tsrc1 = (int8_t *)p_tmp + j * L * (bias->dtype_ & 0xF);
                                    int8_t *tdst = (int8_t *)p_tmp_ou + j * L * (output->dtype_ & 0xF);
                                    ret |= luna_add_api(tsrc1, p_bis, tdst, L, shift);
                                }
                                memcpy(p_out + ou_oft, p_tmp_ou, split_out_size);
                            } else {
                                p_tmp_ou = (int8_t *)tmp->dptr_ + tmp_size;
                                tmp_size += split_out_size;
                                if (tmp_size > workspace_size) {
                                    return -1;
                                }
                                luna_mat_mul_api = (FC_SPLIT_MAT_MUL_LUNA_API)fc_luna_api_list[1][ou_idx].luna_api;
                                ret = luna_mat_mul_api(p_tmp_in, p_weight, p_tmp_ou, split_num, split_M, N, L, shift);
                                memcpy(p_out + ou_oft, p_tmp_ou, split_out_size);
                            }
                        } else {
                            if (has_bias) {
                                int32_t *p_tmp = (int32_t *)((int8_t *)tmp->dptr_ + tmp_size);
                                luna_mat_mul_api = (FC_SPLIT_MAT_MUL_LUNA_API)fc_luna_api_list[1][bias_idx].luna_api;
                                ret = luna_mat_mul_api(p_tmp_in, p_weight, p_tmp, split_num, split_M, N, L, 0);

                                for (int32_t j = 0; j < split_M; j++) {
                                    FC_VEC_ADD_LUNA_API luna_add_api = (FC_VEC_ADD_LUNA_API)fc_luna_api_list[2 + bias_idx][ou_idx].luna_api;
                                    int8_t *tsrc1 = (int8_t *)p_tmp + j * L * (bias->dtype_ & 0xF);
                                    int8_t *tdst = (int8_t *)p_tmp_ou + j * L * (output->dtype_ & 0xF);
                                    ret |= luna_add_api(tsrc1, p_bis, tdst, L, shift);
                                }
                            } else {
                                luna_mat_mul_api = (FC_SPLIT_MAT_MUL_LUNA_API)fc_luna_api_list[1][ou_idx].luna_api;
                                ret = luna_mat_mul_api(p_tmp_in, p_weight, p_tmp_ou, split_num, split_M, N, L, shift);
                            }
                        }
                    }
                }
            } else {
                int8_t *p_tmp_in = p_in;
                int8_t *p_tmp_ou = p_out;
                int32_t tmp_size = 0;

                if (in_is_psram) {
                    p_tmp_in = (int8_t *)tmp->dptr_;
                    tmp_size = M * N;
                    memcpy(p_tmp_in, p_in, M * N);
                }

                if (ou_is_psram) {
                    if (has_bias) {
                        p_tmp_ou = p_tmp_in;
                    } else {
                        p_tmp_ou = (int8_t *)tmp->dptr_ + tmp_size;
                        tmp_size += M * L;
                        if (tmp_size > workspace_size) {
                            return -1;
                        }
                    }
                }

                if (int8_condition_r <= right_limit) {
                    FC_MAT_MUL_LUNA_API luna_mat_mul_api;
                    if (has_bias) {
                        int32_t *p_tmp = (int32_t *)((int8_t *)tmp->dptr_ + tmp_size);
                        tmp_size += M * L * 4;
                        if (tmp_size > workspace_size) {
                            return -1;
                        }
                        luna_mat_mul_api = (FC_MAT_MUL_LUNA_API)fc_luna_api_list[0][bias_idx].luna_api;
                        ret = luna_mat_mul_api(p_tmp_in, p_weight, p_tmp, M, N, L, 0);

                        for (int32_t i = 0; i < M; i++) {
                            FC_VEC_ADD_LUNA_API luna_add_api = (FC_VEC_ADD_LUNA_API)fc_luna_api_list[2 + bias_idx][ou_idx].luna_api;
                            int8_t *tsrc1 = (int8_t *)p_tmp + i * L * (bias->dtype_ & 0xF);
                            int8_t *tdst = (int8_t *)p_tmp_ou + i * L * (output->dtype_ & 0xF);
                            ret |= luna_add_api(tsrc1, p_bis, tdst, L, shift);
                        }
                    } else {
                        luna_mat_mul_api = (FC_MAT_MUL_LUNA_API)fc_luna_api_list[0][ou_idx].luna_api;
                        ret = luna_mat_mul_api(p_tmp_in, p_weight, p_tmp_ou, M, N, L, shift);
                    }
                } else {
                    int32_t split_num = 2;
                    int32_t split_L = L / split_num;
                    while (int8_condition_r > right_limit || (L % split_num != 0)) {
                        split_num++;
                        split_L = L / split_num;
                        int8_condition_r = (luna_ceil(N, 3) << 3) * (luna_ceil(split_L, 2) << 2);
                    }

                    FC_SPLIT_MAT_MUL_LUNA_API luna_mat_mul_api;
                    if (has_bias) {
                        int32_t *p_tmp = (int32_t *)((int8_t *)tmp->dptr_ + tmp_size);
                        tmp_size += M * L * 4;
                        if (tmp_size > workspace_size) {
                            return -1;
                        }
                        luna_mat_mul_api = (FC_SPLIT_MAT_MUL_LUNA_API)fc_luna_api_list[1][bias_idx].luna_api;
                        ret = luna_mat_mul_api(p_tmp_in, p_weight, p_tmp, split_num, M, N, L, 0);

                        for (int32_t i = 0; i < M; i++) {
                            FC_VEC_ADD_LUNA_API luna_add_api = (FC_VEC_ADD_LUNA_API)fc_luna_api_list[2 + bias_idx][ou_idx].luna_api;
                            int8_t *tsrc1 = (int8_t *)p_tmp + i * L * (bias->dtype_ & 0xF);
                            int8_t *tdst = (int8_t *)p_tmp_ou + i * L * (output->dtype_ & 0xF);
                            ret |= luna_add_api(tsrc1, p_bis, tdst, L, shift);
                        }
                    } else {
                        luna_mat_mul_api = (FC_SPLIT_MAT_MUL_LUNA_API)fc_luna_api_list[1][ou_idx].luna_api;
                        ret = luna_mat_mul_api(p_tmp_in, p_weight, p_tmp_ou, split_num, M, N, L, shift);
                    }
                }

                if (ou_is_psram) {
                    memcpy(p_out, p_tmp_ou, M * L);
                }
            }
        } break;

        case Int16: {
            int32_t int8_condition_l = (luna_ceil(M, 2) << 2) * (luna_ceil(N, 3) << 3) * sizeof(int16_t);
            if (int8_condition_l > left_limit) {
                return ret;
            }

            int32_t int8_condition_r = (luna_ceil(N, 3) << 3) * (luna_ceil(L, 2) << 2) * sizeof(int16_t);
            if (int8_condition_r <= right_limit) {
                FC_MAT_MUL_LUNA_INT16_API luna_mat_mul_api;
                if (has_bias) {
                    int32_t *p_tmp = (int32_t *)tmp->dptr_;
                    luna_mat_mul_api = (FC_MAT_MUL_LUNA_INT16_API)fc_luna_int16_api_list[0][bias_idx].luna_api;
                    ret = luna_mat_mul_api((int16_t *)p_in, (int16_t *)p_weight, p_tmp, M, N, L, 0);
                } else {
                    luna_mat_mul_api = (FC_MAT_MUL_LUNA_INT16_API)fc_luna_int16_api_list[0][ou_idx].luna_api;
                    ret = luna_mat_mul_api((int16_t *)p_in, (int16_t *)p_weight, p_out, M, N, L, shift);
                }
            } else {
                int32_t split_num = 2;
                int32_t split_L = L / split_num;
                while (int8_condition_r > right_limit || (L % split_num != 0)) {
                    split_num++;
                    split_L = L / split_num;
                    int8_condition_r = (luna_ceil(N, 3) << 3) * (luna_ceil(split_L, 2) << 2) * sizeof(int16_t);
                }

                FC_SPLIT_MAT_MUL_LUNA_INT16_API luna_mat_mul_api;
                if (has_bias) {
                    int32_t *p_tmp = (int32_t *)tmp->dptr_;
                    luna_mat_mul_api = (FC_SPLIT_MAT_MUL_LUNA_INT16_API)fc_luna_int16_api_list[1][bias_idx].luna_api;
                    ret = luna_mat_mul_api((int16_t *)p_in, (int16_t *)p_weight, p_tmp, split_num, M, N, L, 0);
                } else {
                    luna_mat_mul_api = (FC_SPLIT_MAT_MUL_LUNA_INT16_API)fc_luna_int16_api_list[1][ou_idx].luna_api;
                    ret = luna_mat_mul_api((int16_t *)p_in, (int16_t *)p_weight, p_out, split_num, M, N, L, shift);
                }
            }

            if (has_bias) {
                for (int32_t i = 0; i < M; i++) {
                    int32_t *p_tmp = (int32_t *)tmp->dptr_;
                    FC_VEC_ADD_LUNA_API luna_add_api = (FC_VEC_ADD_LUNA_API)fc_luna_int16_api_list[2 + bias_idx][ou_idx].luna_api;
                    int8_t *tsrc1 = (int8_t *)p_tmp + i * L * (bias->dtype_ & 0xF);
                    int8_t *tdst = (int8_t *)p_out + i * L * (output->dtype_ & 0xF);
                    ret |= luna_add_api(tsrc1, p_bis, tdst, L, shift);
                }
            }

            if (ou_is_psram) {
                memcpy((void *)output->dptr_, p_out, M * L);
            }
        } break;

        default:
            break;
    }

    return ret;
}

/**
 * @brief Main function for linear transformation with integer quantization
 * @param input Input tensor
 * @param weight Weight tensor
 * @param bias Bias tensor (optional)
 * @param attrs Linear transformation attributes
 * @param workspace Temporary workspace tensor
 * @param output Output tensor
 * @return int32_t Operation status
 */
int32_t linearint_luna(tTensor *input, tTensor *weight, tTensor *bias,
                       LinearIntAttrs *attrs, tTensor *workspace,
                       tTensor *output) {
    int32_t ret = T_ERR_FAIL;
    ret = calc_linearint_luna(input, weight, bias, output, workspace);
    return ret;
}

#endif