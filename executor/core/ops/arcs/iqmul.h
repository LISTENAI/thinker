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
    size_t bytes = Temp ? getTensorDataSize(Temp) : 0;
    return bytes > INT32_MAX ? INT32_MAX : (int32_t)bytes;
}

static int32_t iqmul_workspace_need(int32_t cur_size, int32_t elem_bytes,
                                    int32_t x1_in_psram, int32_t x2_in_psram,
                                    int32_t y_in_psram) {
    int32_t workspace_size = 0;
    if (x1_in_psram) workspace_size += ALIGN4(cur_size * elem_bytes);
    if (x2_in_psram) workspace_size += ALIGN4(cur_size * elem_bytes);
    if (y_in_psram) workspace_size += cur_size * elem_bytes;
    return workspace_size;
}

static int32_t iqmul_split_size(size_t remain_size, int32_t elem_bytes,
                                int32_t x1_in_psram, int32_t x2_in_psram,
                                int32_t y_in_psram, int32_t workspace_size) {
    int32_t bytes_once = (x1_in_psram + x2_in_psram + y_in_psram) * elem_bytes;
    if (bytes_once == 0)
        return remain_size > 0x7fffffff ? 0x7fffffff : (int32_t)remain_size;

    int32_t cur_size = workspace_size / bytes_once;
    if (remain_size < (size_t)cur_size) cur_size = (int32_t)remain_size;
    while (cur_size > 0 && iqmul_workspace_need(cur_size, elem_bytes,
                                                x1_in_psram, x2_in_psram,
                                                y_in_psram) > workspace_size)
        cur_size--;
    return cur_size;
}

static int32_t iqmul_broadcast_workspace_need(int32_t cur_c, int32_t hw,
                                              int32_t x1_in_psram,
                                              int32_t x2_in_psram,
                                              int32_t y_in_psram) {
    int32_t cur_size = cur_c * hw;
    int32_t workspace_size = ALIGN4(hw);
    if (x2_in_psram) workspace_size += ALIGN4(cur_c);
    workspace_size += ALIGN4(cur_size);
    if (x1_in_psram) workspace_size += ALIGN4(cur_size);
    if (y_in_psram) workspace_size += cur_size;
    return workspace_size;
}

static int32_t iqmul_broadcast_split_c(int32_t remain_c, int32_t hw,
                                       int32_t x1_in_psram,
                                       int32_t x2_in_psram,
                                       int32_t y_in_psram,
                                       int32_t workspace_size) {
    int32_t left = 1, right = remain_c, best = 0;
    while (left <= right) {
        int32_t mid = (left + right) >> 1;
        if (iqmul_broadcast_workspace_need(mid, hw, x1_in_psram,
                                           x2_in_psram, y_in_psram) <= workspace_size) {
            best = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return best;
}

static int32_t iqmul_scalar_i8(tTensor *lhs, tTensor *Y, tTensor *Temp,
                               int8_t scalar, size_t size, int32_t shift) {
    int32_t x1_in_psram = (lhs->mem_.type_ != 2);
    int32_t y_in_psram = (Y->mem_.type_ != 2);
    int32_t workspace_size = iqmul_workspace_bytes(Temp);
    int8_t *workspace = Temp ? (int8_t *)Temp->dptr_ : NULL;
    size_t past_size = 0;

    #if THINKER_RUNTIME_CHECK
    if ((x1_in_psram || y_in_psram) && workspace == NULL) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif

    while (past_size < size) {
        int32_t cur_size = iqmul_split_size(size - past_size, sizeof(int8_t),
                                            x1_in_psram, 0, y_in_psram,
                                            workspace_size);
        int8_t *p_tmp = workspace;
        int8_t *src = (int8_t *)lhs->dptr_ + past_size;
        int8_t *dst = (int8_t *)Y->dptr_ + past_size;
        int8_t *src_temp = src;
        int8_t *dst_temp = dst;
        #if THINKER_RUNTIME_CHECK
        if (cur_size <= 0) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
        if (x1_in_psram) {
            src_temp = p_tmp;
            THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(src_temp, src, cur_size),
                              "luna_memcpy_i8o8");
            p_tmp += ALIGN4(cur_size);
        }
        if (y_in_psram) dst_temp = p_tmp;
        THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(src_temp, scalar, dst_temp,
                                                cur_size, shift),
                          "luna_scale_i8i8o8");
        if (y_in_psram) opi_psram_cpy_out(dst, dst_temp, cur_size);
        past_size += cur_size;
    }
    return T_SUCCESS;
}

static int32_t iqmul_scalar_i32(tTensor *lhs, tTensor *Y, tTensor *Temp,
                                int32_t scalar, size_t size, int32_t shift) {
    int32_t x1_in_psram = (lhs->mem_.type_ != 2);
    int32_t y_in_psram = (Y->mem_.type_ != 2);
    int32_t workspace_size = iqmul_workspace_bytes(Temp);
    int8_t *workspace = Temp ? (int8_t *)Temp->dptr_ : NULL;
    size_t past_size = 0;

    #if THINKER_RUNTIME_CHECK
    if ((x1_in_psram || y_in_psram) && workspace == NULL) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif

    while (past_size < size) {
        int32_t cur_size = iqmul_split_size(size - past_size, sizeof(int32_t),
                                            x1_in_psram, 0, y_in_psram,
                                            workspace_size);
        int8_t *p_tmp = workspace;
        int32_t *src = (int32_t *)lhs->dptr_ + past_size;
        int32_t *dst = (int32_t *)Y->dptr_ + past_size;
        int32_t *src_temp = src;
        int32_t *dst_temp = dst;
        #if THINKER_RUNTIME_CHECK
        if (cur_size <= 0) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
        if (x1_in_psram) {
            src_temp = (int32_t *)p_tmp;
            THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)src_temp,
                              (int8_t *)src, cur_size * sizeof(int32_t)),
                              "luna_memcpy_i8o8");
            p_tmp += ALIGN4(cur_size * sizeof(int32_t));
        }
        if (y_in_psram) dst_temp = (int32_t *)p_tmp;
        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(src_temp, scalar, dst_temp,
                                                   cur_size, shift),
                          "luna_scale_i32i32o32");
        if (y_in_psram)
            opi_psram_cpy_out(dst, dst_temp, cur_size * sizeof(int32_t));
        past_size += cur_size;
    }
    return T_SUCCESS;
}

static int32_t iqmul_vec_i8(tTensor *lhs, tTensor *rhs, tTensor *Y,
                            tTensor *Temp, size_t size, int32_t shift) {
    int32_t x1_in_psram = (lhs->mem_.type_ != 2);
    int32_t x2_in_psram = (rhs->mem_.type_ != 2);
    int32_t y_in_psram = (Y->mem_.type_ != 2);
    int32_t workspace_size = iqmul_workspace_bytes(Temp);
    int8_t *workspace = Temp ? (int8_t *)Temp->dptr_ : NULL;
    size_t past_size = 0;

    #if THINKER_RUNTIME_CHECK
    if ((x1_in_psram || x2_in_psram || y_in_psram) &&
                          workspace == NULL) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif

    while (past_size < size) {
        int32_t cur_size = iqmul_split_size(size - past_size, sizeof(int8_t),
                                            x1_in_psram, x2_in_psram,
                                            y_in_psram, workspace_size);
        int8_t *p_tmp = workspace;
        int8_t *src1 = (int8_t *)lhs->dptr_ + past_size;
        int8_t *src2 = (int8_t *)rhs->dptr_ + past_size;
        int8_t *dst = (int8_t *)Y->dptr_ + past_size;
        int8_t *src1_temp = src1;
        int8_t *src2_temp = src2;
        int8_t *dst_temp = dst;
        #if THINKER_RUNTIME_CHECK
        if (cur_size <= 0) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
        if (x1_in_psram) {
            src1_temp = p_tmp;
            THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(src1_temp, src1, cur_size),
                              "luna_memcpy_i8o8");
            p_tmp += ALIGN4(cur_size);
        }
        if (x2_in_psram) {
            src2_temp = p_tmp;
            THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(src2_temp, src2, cur_size),
                              "luna_memcpy_i8o8");
            p_tmp += ALIGN4(cur_size);
        }
        if (y_in_psram) dst_temp = p_tmp;
        THINKER_RET_CHECK(API_LIB(mul_i8i8o8)(src1_temp, src2_temp,
                                              dst_temp, cur_size, shift),
                          "luna_mul_i8i8o8");
        if (y_in_psram) opi_psram_cpy_out(dst, dst_temp, cur_size);
        past_size += cur_size;
    }
    return T_SUCCESS;
}

static int32_t iqmul_vec_i32(tTensor *lhs, tTensor *rhs, tTensor *Y,
                             tTensor *Temp, size_t size, int32_t shift) {
    int32_t x1_in_psram = (lhs->mem_.type_ != 2);
    int32_t x2_in_psram = (rhs->mem_.type_ != 2);
    int32_t y_in_psram = (Y->mem_.type_ != 2);
    int32_t workspace_size = iqmul_workspace_bytes(Temp);
    int8_t *workspace = Temp ? (int8_t *)Temp->dptr_ : NULL;
    size_t past_size = 0;

    #if THINKER_RUNTIME_CHECK
    if ((x1_in_psram || x2_in_psram || y_in_psram) &&
                          workspace == NULL) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif

    while (past_size < size) {
        int32_t cur_size = iqmul_split_size(size - past_size, sizeof(int32_t),
                                            x1_in_psram, x2_in_psram,
                                            y_in_psram, workspace_size);
        int8_t *p_tmp = workspace;
        int32_t *src1 = (int32_t *)lhs->dptr_ + past_size;
        int32_t *src2 = (int32_t *)rhs->dptr_ + past_size;
        int32_t *dst = (int32_t *)Y->dptr_ + past_size;
        int32_t *src1_temp = src1;
        int32_t *src2_temp = src2;
        int32_t *dst_temp = dst;
        #if THINKER_RUNTIME_CHECK
        if (cur_size <= 0) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
        if (x1_in_psram) {
            src1_temp = (int32_t *)p_tmp;
            THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)src1_temp,
                              (int8_t *)src1, cur_size * sizeof(int32_t)),
                              "luna_memcpy_i8o8");
            p_tmp += ALIGN4(cur_size * sizeof(int32_t));
        }
        if (x2_in_psram) {
            src2_temp = (int32_t *)p_tmp;
            THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)src2_temp,
                              (int8_t *)src2, cur_size * sizeof(int32_t)),
                              "luna_memcpy_i8o8");
            p_tmp += ALIGN4(cur_size * sizeof(int32_t));
        }
        if (y_in_psram) dst_temp = (int32_t *)p_tmp;
        THINKER_RET_CHECK(API_LIB(mul_i32i32o32)(src1_temp, src2_temp,
                                                 dst_temp, cur_size, shift),
                          "luna_mul_i32i32o32");
        if (y_in_psram)
            opi_psram_cpy_out(dst, dst_temp, cur_size * sizeof(int32_t));
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
    int32_t x1_in_psram = (lhs->mem_.type_ != 2);
    int32_t x2_in_psram = (rhs->mem_.type_ != 2);
    int32_t y_in_psram = (Y->mem_.type_ != 2);
    int32_t workspace_size = iqmul_workspace_bytes(Temp);
    int32_t max_c;

    #if THINKER_PARAM_CHECK
    if (lhs->dtype_ != Int8 || rhs->dtype_ != Int8 || Y->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (ALIGN4(hw) > 16384) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    #if THINKER_RUNTIME_CHECK
    if (Temp == NULL || Temp->dptr_ == 0) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif
    max_c = iqmul_broadcast_split_c(c, hw, x1_in_psram, x2_in_psram,
                                    y_in_psram, workspace_size);
    #if THINKER_RUNTIME_CHECK
    if (max_c <= 0) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif

    int8_t *workspace = (int8_t *)Temp->dptr_;
    int8_t *ones = workspace;
    THINKER_RET_CHECK(API_LIB(memset_i8o8)(ones, 1, hw), "luna_memset_i8o8");

    for (int32_t b = 0; b < n; b++) {
            int32_t rhs_batch_offset = (rhs_n == 1) ? 0 : b * c;
        for (int32_t ch = 0; ch < c; ch += max_c) {
            int32_t cur_c = MIN(max_c, c - ch);
            int32_t cur_size = cur_c * hw;
            int8_t *p_tmp = workspace + ALIGN4(hw);
            int8_t *lhs_cur = (int8_t *)lhs->dptr_ + (b * c + ch) * hw;
            int8_t *rhs_cur = (int8_t *)rhs->dptr_ + rhs_batch_offset + ch;
            int8_t *dst_cur = (int8_t *)Y->dptr_ + (b * c + ch) * hw;
            int8_t *rhs_temp = rhs_cur;
            int8_t *lhs_temp = lhs_cur;
            int8_t *dst_temp = dst_cur;

            if (x2_in_psram) {
                rhs_temp = p_tmp;
                THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(rhs_temp, rhs_cur, cur_c),
                                  "luna_memcpy_i8o8");
                p_tmp += ALIGN4(cur_c);
            }
            int8_t *rhs_broadcast = p_tmp;
            p_tmp += ALIGN4(cur_size);
            if (x1_in_psram) {
                lhs_temp = p_tmp;
                THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(lhs_temp, lhs_cur, cur_size),
                                  "luna_memcpy_i8o8");
                p_tmp += ALIGN4(cur_size);
            }
            if (y_in_psram) dst_temp = p_tmp;

            THINKER_RET_CHECK(API_LIB(mat_mul_i8i8o8)(rhs_temp, ones, rhs_broadcast,
                                                       cur_c, 1, hw, 0),
                              "luna_mat_mul_i8i8o8");
            THINKER_RET_CHECK(API_LIB(mul_i8i8o8)(lhs_temp, rhs_broadcast, dst_temp,
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
    #if THINKER_PARAM_CHECK
    if (lhs == NULL || rhs == NULL || Y == NULL || attrs == NULL ||
                        lhs->dptr_ == 0 || rhs->dptr_ == 0 || Y->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    int32_t x1_q = (int32_t)lhs->scale_;
    int32_t x2_q = (int32_t)rhs->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    int32_t shift = x1_q + x2_q - y_q;
    size_t size = getTensorSize(lhs);

    #if THINKER_PARAM_CHECK
    if (shift < 0 || shift > 63) {
        return (T_ERR_INVALID_PARA);
    }

    if ((lhs->dtype_ != rhs->dtype_) || (lhs->dtype_ != Y->dtype_)) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (lhs->dtype_ != Int8 && lhs->dtype_ != Int32) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (!equalShape(&lhs->shape_, &Y->shape_)) {
        return (T_ERR_INVALID_PARA);
    }
#endif
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
        } else {
            int32_t scalar = *(int32_t *)rhs->dptr_;
            THINKER_RET_CHECK(iqmul_scalar_i32(lhs, Y, Temp, scalar, size, shift),
                              "iqmul_scalar_i32");
        }
    } else {
        #if THINKER_PARAM_CHECK
        if (!equalShape(&lhs->shape_, &rhs->shape_)) {
            return (T_ERR_INVALID_DATATYPE);
        }
#endif
        if (rhs->dtype_ == Int8) {
            THINKER_RET_CHECK(iqmul_vec_i8(lhs, rhs, Y, Temp, size, shift),
                              "iqmul_vec_i8");
        } else {
            THINKER_RET_CHECK(iqmul_vec_i32(lhs, rhs, Y, Temp, size, shift),
                              "iqmul_vec_i32");
        }
    }

    return T_SUCCESS;
}

#endif  // _MUL_LUNA_H_
