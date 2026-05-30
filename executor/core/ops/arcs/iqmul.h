#ifndef _MUL_LUNA_H_
#define _MUL_LUNA_H_

#include <math.h>
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

static int32_t iqmul_workspace_bytes(tTensor *Temp) {
    return Temp ? (int32_t)Temp->shape_.dims_[0] : 0;
}

static int32_t iqmul_output_chunk_size(size_t remain_size, int32_t elem_bytes,
                                       int32_t y_in_psram, int32_t workspace_size) {
    if (!y_in_psram) {
        return (int32_t)remain_size;
    }

    int32_t chunk_size = workspace_size / elem_bytes;
    if (chunk_size <= 0) {
        return 0;
    }
    return chunk_size < (int32_t)remain_size ? chunk_size : (int32_t)remain_size;
}

static int32_t iqmul_scalar_i8(tTensor *lhs, tTensor *Y, tTensor *Temp,
                               int8_t scalar, size_t size, int32_t shift) {
    int32_t y_in_psram = (Y->mem_.type_ != 2);
    int32_t workspace_size = iqmul_workspace_bytes(Temp);
    int8_t *workspace = Temp ? (int8_t *)Temp->dptr_ : NULL;
    int8_t *src = (int8_t *)lhs->dptr_;
    int8_t *dst = (int8_t *)Y->dptr_;
    size_t past_size = 0;

    if (!y_in_psram) {
        return API_LIB(scale_i8i8o8)(src, scalar, dst, size, shift);
    }

    while (past_size < size) {
        int32_t cur_size = iqmul_output_chunk_size(size - past_size, sizeof(int8_t),
                                                   y_in_psram, workspace_size);
        if (cur_size <= 0 || workspace == NULL) {
            return T_ERR_NO_WORKSPACE;
        }
        THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(src + past_size, scalar, workspace,
                                                cur_size, shift),
                          "luna_scale_i8i8o8");
        opi_psram_cpy_out(dst + past_size, workspace, cur_size * sizeof(int8_t));
        past_size += cur_size;
    }
    return T_SUCCESS;
}

static int32_t iqmul_scalar_i32(tTensor *lhs, tTensor *Y, tTensor *Temp,
                                int32_t scalar, size_t size, int32_t shift) {
    int32_t y_in_psram = (Y->mem_.type_ != 2);
    int32_t workspace_size = iqmul_workspace_bytes(Temp);
    int32_t *workspace = Temp ? (int32_t *)Temp->dptr_ : NULL;
    int32_t *src = (int32_t *)lhs->dptr_;
    int32_t *dst = (int32_t *)Y->dptr_;
    size_t past_size = 0;

    if (!y_in_psram) {
        return API_LIB(scale_i32i32o32)(src, scalar, dst, size, shift);
    }

    while (past_size < size) {
        int32_t cur_size = iqmul_output_chunk_size(size - past_size, sizeof(int32_t),
                                                   y_in_psram, workspace_size);
        if (cur_size <= 0 || workspace == NULL) {
            return T_ERR_NO_WORKSPACE;
        }
        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(src + past_size, scalar, workspace,
                                                   cur_size, shift),
                          "luna_scale_i32i32o32");
        opi_psram_cpy_out(dst + past_size, workspace, cur_size * sizeof(int32_t));
        past_size += cur_size;
    }
    return T_SUCCESS;
}

static int32_t iqmul_vec_i8(tTensor *lhs, tTensor *rhs, tTensor *Y,
                            tTensor *Temp, size_t size, int32_t shift) {
    int32_t y_in_psram = (Y->mem_.type_ != 2);
    int32_t workspace_size = iqmul_workspace_bytes(Temp);
    int8_t *workspace = Temp ? (int8_t *)Temp->dptr_ : NULL;
    int8_t *src1 = (int8_t *)lhs->dptr_;
    int8_t *src2 = (int8_t *)rhs->dptr_;
    int8_t *dst = (int8_t *)Y->dptr_;
    size_t past_size = 0;

    if (!y_in_psram) {
        return API_LIB(mul_i8i8o8)(src1, src2, dst, size, shift);
    }

    while (past_size < size) {
        int32_t cur_size = iqmul_output_chunk_size(size - past_size, sizeof(int8_t),
                                                   y_in_psram, workspace_size);
        if (cur_size <= 0 || workspace == NULL) {
            return T_ERR_NO_WORKSPACE;
        }
        THINKER_RET_CHECK(API_LIB(mul_i8i8o8)(src1 + past_size, src2 + past_size,
                                              workspace, cur_size, shift),
                          "luna_mul_i8i8o8");
        opi_psram_cpy_out(dst + past_size, workspace, cur_size * sizeof(int8_t));
        past_size += cur_size;
    }
    return T_SUCCESS;
}

static int32_t iqmul_vec_i32(tTensor *lhs, tTensor *rhs, tTensor *Y,
                             tTensor *Temp, size_t size, int32_t shift) {
    int32_t y_in_psram = (Y->mem_.type_ != 2);
    int32_t workspace_size = iqmul_workspace_bytes(Temp);
    int32_t *workspace = Temp ? (int32_t *)Temp->dptr_ : NULL;
    int32_t *src1 = (int32_t *)lhs->dptr_;
    int32_t *src2 = (int32_t *)rhs->dptr_;
    int32_t *dst = (int32_t *)Y->dptr_;
    size_t past_size = 0;

    if (!y_in_psram) {
        return API_LIB(mul_i32i32o32)(src1, src2, dst, size, shift);
    }

    while (past_size < size) {
        int32_t cur_size = iqmul_output_chunk_size(size - past_size, sizeof(int32_t),
                                                   y_in_psram, workspace_size);
        if (cur_size <= 0 || workspace == NULL) {
            return T_ERR_NO_WORKSPACE;
        }
        THINKER_RET_CHECK(API_LIB(mul_i32i32o32)(src1 + past_size, src2 + past_size,
                                                 workspace, cur_size, shift),
                          "luna_mul_i32i32o32");
        opi_psram_cpy_out(dst + past_size, workspace, cur_size * sizeof(int32_t));
        past_size += cur_size;
    }
    return T_SUCCESS;
}

/**
 * @brief Performs vector multiplication with NCHW x NC11 broadcast.
 */
int32_t calc_vec_mul_luna_b2b2_broadcast_h1w1(tTensor *lhs, tTensor *rhs,
                                               tTensor *Y, tTensor *Temp,
                                               int32_t shift) {
    int32_t n = lhs->shape_.dims_[0];
    int32_t c = lhs->shape_.dims_[1];
    int32_t h = lhs->shape_.dims_[2];
    int32_t w = lhs->shape_.dims_[3];
    int32_t rhs_n = rhs->shape_.dims_[0];
    int32_t hw = h * w;
    int32_t y_in_psram = (Y->mem_.type_ != 2);
    int32_t workspace_size = iqmul_workspace_bytes(Temp);
    int32_t ones_bytes = ALIGN4(hw * (int32_t)sizeof(int8_t));
    int32_t factor = y_in_psram ? 2 : 1;

    if (lhs->dtype_ != Int8 || rhs->dtype_ != Int8 || Y->dtype_ != Int8) {
        return T_ERR_INVALID_DATATYPE;
    }
    if (Temp == NULL || Temp->dptr_ == 0 || workspace_size <= ones_bytes) {
        return T_ERR_NO_WORKSPACE;
    }

    int32_t max_c = y_in_psram ? ((workspace_size - ones_bytes) / (hw * factor)) : c;
    if (max_c <= 0 || (!y_in_psram && (workspace_size - ones_bytes) < c * hw)) {
        return T_ERR_NO_WORKSPACE;
    }

    int8_t *workspace = (int8_t *)Temp->dptr_;
    int8_t *ones = workspace;
    int8_t *rhs_broadcast = workspace + ones_bytes;
    THINKER_RET_CHECK(API_LIB(memset_i8o8)(ones, 1, hw), "luna_memset_i8o8");

    for (int32_t b = 0; b < n; b++) {
        int32_t rhs_batch_offset = (rhs_n == 1) ? 0 : b * c;
        for (int32_t ch = 0; ch < c; ch += max_c) {
            int32_t cur_c = MIN(max_c, c - ch);
            int32_t cur_size = cur_c * hw;
            int8_t *lhs_cur = (int8_t *)lhs->dptr_ + (b * c + ch) * hw;
            int8_t *rhs_cur = (int8_t *)rhs->dptr_ + rhs_batch_offset + ch;
            int8_t *dst_cur = (int8_t *)Y->dptr_ + (b * c + ch) * hw;
            int8_t *dst_temp = y_in_psram ? (rhs_broadcast + cur_size) : dst_cur;

            THINKER_RET_CHECK(API_LIB(mat_mul_i8i8o8)(rhs_cur, ones, rhs_broadcast,
                                                       cur_c, 1, hw, 0),
                              "luna_mat_mul_i8i8o8");
            THINKER_RET_CHECK(API_LIB(mul_i8i8o8)(lhs_cur, rhs_broadcast, dst_temp,
                                                  cur_size, shift),
                              "luna_mul_i8i8o8");
            if (y_in_psram) {
                opi_psram_cpy_out(dst_cur, dst_temp, cur_size * sizeof(int8_t));
            }
        }
    }

    return T_SUCCESS;
}

/**
 * @brief Quantized multiplication operation implementation
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param Y Output tensor
 * @param Temp Temporary workspace tensor
 * @param attrs Operation attributes
 * @return int32_t Operation status
 */
int32_t iqmul_luna(tTensor *lhs, tTensor *rhs, tTensor *Y, tTensor *Temp, iqBinaryAttrs *attrs) {
    int32_t x1_q = (int32_t)lhs->scale_;
    int32_t x2_q = (int32_t)rhs->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    int32_t shift = x1_q + x2_q - y_q;
    size_t size = getTensorSize(lhs);

    if (shift < 0) {
        return T_ERR_INVALID_PARA;
    }

    if ((lhs->dtype_ != rhs->dtype_) || (lhs->dtype_ != Y->dtype_)) {
        return T_ERR_INVALID_DATATYPE;
    }

    if (lhs->shape_.ndim_ == 4 && rhs->shape_.ndim_ == 4 &&
        lhs->shape_.dims_[1] == rhs->shape_.dims_[1] &&
        rhs->shape_.dims_[2] == 1 && rhs->shape_.dims_[3] == 1 &&
        (rhs->shape_.dims_[0] == 1 || rhs->shape_.dims_[0] == lhs->shape_.dims_[0])) {
        THINKER_RET_CHECK(calc_vec_mul_luna_b2b2_broadcast_h1w1(lhs, rhs, Y, Temp, shift),
                          "calc_vec_mul_luna_b2b2_broadcast_h1w1");
    } else if (rhs->shape_.ndim_ == 0) {
        if (rhs->dtype_ == Int8) {
            int8_t scalar = *(int8_t *)rhs->dptr_;
            THINKER_RET_CHECK(iqmul_scalar_i8(lhs, Y, Temp, scalar, size, shift),
                              "iqmul_scalar_i8");
        } else if (rhs->dtype_ == Int32) {
            int32_t scalar = *(int32_t *)rhs->dptr_;
            THINKER_RET_CHECK(iqmul_scalar_i32(lhs, Y, Temp, scalar, size, shift),
                              "iqmul_scalar_i32");
        } else {
            return T_ERR_INVALID_DATATYPE;
        }
    } else {
        if (!equalShape(&lhs->shape_, &rhs->shape_)) {
            return T_ERR_INVALID_DATATYPE;
        }
        if (rhs->dtype_ == Int8) {
            THINKER_RET_CHECK(iqmul_vec_i8(lhs, rhs, Y, Temp, size, shift),
                              "iqmul_vec_i8");
        } else if (rhs->dtype_ == Int32) {
            THINKER_RET_CHECK(iqmul_vec_i32(lhs, rhs, Y, Temp, size, shift),
                              "iqmul_vec_i32");
        } else {
            return T_ERR_INVALID_DATATYPE;
        }
    }

    return T_SUCCESS;
}

#endif  // _MUL_LUNA_H_
