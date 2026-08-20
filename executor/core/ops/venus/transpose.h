#ifndef _TRANSPOSE_LUNA_H_
#define _TRANSPOSE_LUNA_H_

#include <stdio.h>
#include <string.h>
#include "core/comm/utils.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif
static int32_t get_transpose_split_num(int32_t row, int32_t col, int32_t dtype) {
    for (int32_t split_num = 2; split_num <= row; ++split_num) {
        if (row % split_num != 0) continue;
        int32_t split_row = row / split_num;
        if ((dtype == Int8 && ALIGN4(split_row) * ALIGN8(col) <= 65536) ||
            (dtype == Int16 && ALIGN4(split_row) * ALIGN2(col) <= 32768) ||
            (dtype == Int32 && ALIGN2(split_row) * ALIGN2(col) <= 16384))
            return split_num;
    }
    return 0;
}

/**
 * @brief Transpose a matrix of specified data type
 * @param dtype Data type of the matrix (Int8, Int16, Int32)
 * @param dst Output matrix pointer
 * @param src Input matrix pointer
 * @param row Number of rows in the matrix
 * @param col Number of columns in the matrix
 * @return Execution status
 */
int32_t transpose_luna(tTensor *X, tTensor *Y, tTensor * workspace, uint32_t dims, uint32_t *axes, uint32_t *shape) {
    #if THINKER_PARAM_CHECK
    if (X == NULL || Y == NULL || axes == NULL || shape == NULL ||
                        X->dptr_ == 0 || Y->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
    if (dims < 2 || dims > 4 || X->shape_.ndim_ == 0 ||
                        Y->shape_.ndim_ != X->shape_.ndim_ ||
                        getShapeSize(&Y->shape_) != getShapeSize(&X->shape_)) {
        return (T_ERR_INVALID_PARA);
    }
    if (!((X->dtype_ == Int8 && X->byte_ == 1) ||
                          (X->dtype_ == Int16 && X->byte_ == 2) ||
                          ((X->dtype_ == Int32 || X->dtype_ == Float32) && X->byte_ == 4)) ||
                        Y->dtype_ != X->dtype_ || Y->byte_ != X->byte_) {
        return (T_ERR_INVALID_DATATYPE);
    }
    #endif
    for (uint32_t i = 0; i < dims; ++i) {
        #if THINKER_PARAM_CHECK
        if (shape[i] == 0 || axes[i] >= dims) {
            return (T_ERR_INVALID_PARA);
        }
        #endif
        for (uint32_t j = 0; j < i; ++j) {
#if THINKER_PARAM_CHECK
            if (axes[j] == axes[i]) {
                return (T_ERR_INVALID_PARA);
            }
#endif
        }
    }
    #if THINKER_PARAM_CHECK
    if ((dims == 2 && (axes[0] != 1 || axes[1] != 0)) ||
        (dims == 3 && !((axes[0] == 0 && axes[1] == 2 && axes[2] == 1) ||
                        (axes[0] == 2 && axes[1] == 0 && axes[2] == 1) ||
                        (axes[0] == 2 && axes[1] == 1 && axes[2] == 0) ||
                        (axes[0] == 1 && axes[1] == 2 && axes[2] == 0) ||
                        (axes[0] == 1 && axes[1] == 0 && axes[2] == 2))) ||
        (dims == 4 && !(axes[0] == 0 &&
                        ((axes[1] == 1 && axes[2] == 3 && axes[3] == 2) ||
                         (axes[1] == 3 && axes[2] == 1 && axes[3] == 2) ||
                         (axes[1] == 3 && axes[2] == 2 && axes[3] == 1) ||
                         (axes[1] == 2 && axes[2] == 3 && axes[3] == 1) ||
                         (axes[1] == 2 && axes[2] == 1 && axes[3] == 3))))) {
        return (T_ERR_INVALID_PARA);
    }
    #endif
    void *src = (void *)X->dptr_;
    void *dst = (void *)Y->dptr_;
    size_t workspace_size = workspace ? getTensorDataSize(workspace) : 0;
    size_t total_size = getTensorDataSize(X);
    int16_t dtype = X->dtype_ == Float32 ? Int32 : X->dtype_;

    // A 1xN or Nx1 transpose preserves the linear memory layout.
    if (dims == 2 && (shape[0] == 1 || shape[1] == 1)) {
        THINKER_RET_CHECK(API_LIB(memcpy)(dst, src, total_size), "luna_memcpy");
        return T_SUCCESS;
    }

    if ((X->mem_.type_ != 2 || Y->mem_.type_ != 2) &&
        (workspace == NULL || workspace->dptr_ == 0 ||
         workspace->shape_.ndim_ == 0)) return T_ERR_NO_WORKSPACE;

    bool srcInPSRAM = (X->mem_.type_ != 2);
    bool dstInPSRAM = (Y->mem_.type_ != 2);
    if (X->dptr_ == Y->dptr_) {
        bool split_needed = dims >= 3;
        if (dims == 2) {
            split_needed = (dtype == Int8 && ALIGN4(shape[0]) * ALIGN8(shape[1]) > 65536) ||
                           (dtype == Int16 && ALIGN4(shape[0]) * ALIGN2(shape[1]) > 32768) ||
                           (dtype == Int32 && ALIGN2(shape[0]) * ALIGN2(shape[1]) > 16384);
        }
        #if THINKER_PARAM_CHECK
        if (split_needed) {
            return (T_ERR_INVALID_PARA);
        }
        #endif
    }

    if ((!srcInPSRAM) & (!dstInPSRAM)) {
        switch (dims) {
            case 2: {
                uint32_t row = shape[0];
                uint32_t col = shape[1];
                switch (dtype) {
                    case Int8:
                        if (ALIGN4(row) * ALIGN8(col) <= 65536)
                            THINKER_RET_CHECK(luna_mat_trans_q7(src, dst, row, col), "luna_mat_trans_q7");
                        else {
                            int32_t split_num = get_transpose_split_num(row, col, Int8);
                            if (split_num == 0) return T_ERR_NO_IMPLEMENTED;
                            THINKER_RET_CHECK(luna_split_mat_trans_q7(src, dst, row, col, split_num), "luna_split_mat_trans_q7");
                        }
                        break;
                    case Int16:
                        if (ALIGN4(row) * ALIGN2(col) <= 32768)
                            THINKER_RET_CHECK(luna_mat_trans_q15(src, dst, row, col), "luna_mat_trans_q15");
                        else {
                            int32_t split_num = get_transpose_split_num(row, col, Int16);
                            if (split_num == 0) return T_ERR_NO_IMPLEMENTED;
                            THINKER_RET_CHECK(luna_split_mat_trans_q15(src, dst, row, col, split_num), "luna_split_mat_trans_q15");
                        }
                        break;
                    case Int32:
                        if (ALIGN2(row) * ALIGN2(col) <= 16384)
                            THINKER_RET_CHECK(luna_mat_trans_q31(src, dst, row, col), "luna_mat_trans_q31");
                        else {
                            int32_t split_num = get_transpose_split_num(row, col, Int32);
                            if (split_num == 0) return T_ERR_NO_IMPLEMENTED;
                            THINKER_RET_CHECK(luna_split_mat_trans_q31(src, dst, row, col, split_num), "luna_split_mat_trans_q31");
                        }
                        break;
                    default:
                        return T_ERR_INVALID_DATATYPE;
                }
                break;
            }
            case 3: {
                switch (dtype) {
                    case Int8:
                        THINKER_RET_CHECK(luna_trans_axis_q7(src, dst, shape, axes, dims), "luna_trans_axis_q7");
                        break;
                    case Int16:
                        THINKER_RET_CHECK(luna_trans_axis_q15(src, dst, shape, axes, dims), "luna_trans_axis_q15");
                        break;
                    case Int32:
                        THINKER_RET_CHECK(luna_trans_axis_q31(src, dst, shape, axes, dims), "luna_trans_axis_q31");
                        break;
                    default:
                        return T_ERR_INVALID_DATATYPE;
                }
                break;
            }
            case 4:  // only support (0 == new_perm[0]), convert to 3D transpose
            {
                if (0 == axes[0]) {
                    int32_t batch = shape[0];
                    int32_t one_batch_size = shape[1] * shape[2] * shape[3] * X->byte_;

                    uint32_t new_axis[3];
                    uint32_t new_shape[3];
                    for (int32_t n = 0; n < 3; n++) {
                        new_axis[n] = axes[n + 1] - 1;
                        new_shape[n] = shape[n + 1];
                    }

                    switch (dtype) {
                        case Int8:
                            for (int32_t i = 0; i < batch; i++) {
                                void *src_temp = (void *)((int8_t *)src + i * one_batch_size);
                                void *dst_temp = (void *)((int8_t *)dst + i * one_batch_size);
                                THINKER_RET_CHECK(luna_trans_axis_q7(src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_q7");
                            }
                            break;
                        case Int16:
                            for (int32_t i = 0; i < batch; i++) {
                                void *src_temp = (void *)((int8_t *)src + i * one_batch_size);
                                void *dst_temp = (void *)((int8_t *)dst + i * one_batch_size);
                                THINKER_RET_CHECK(luna_trans_axis_q15(src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_q15");
                            }
                            break;
                        case Int32:
                            for (int32_t i = 0; i < batch; i++) {
                                void *src_temp = (void *)((int8_t *)src + i * one_batch_size);
                                void *dst_temp = (void *)((int8_t *)dst + i * one_batch_size);
                                THINKER_RET_CHECK(luna_trans_axis_q31(src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_q31");
                            }
                            break;
                        default:
                            return T_ERR_INVALID_DATATYPE;
                    }
                }
                else {
                    return T_ERR_NO_IMPLEMENTED;
                }
                break;
            }
        }
    }
    else if (srcInPSRAM & (!dstInPSRAM)) {
        switch (dims) {
            case 2: {
                int32_t row = shape[0];
                int32_t col = shape[1];
                switch (dtype) {
                    case Int8:
                        if (ALIGN4(row) * ALIGN8(col) <= 65536 && total_size <= workspace_size) {
                            int8_t *src_temp = (int8_t *)workspace->dptr_;
                            memcpy(src_temp, src, total_size);
                            THINKER_RET_CHECK(luna_mat_trans_q7(src_temp, dst, row, col), "luna_mat_trans_q7");
                        }
                        else if (total_size <= workspace_size) {
                            int8_t *src_temp = (int8_t *)workspace->dptr_;
                            memcpy(src_temp, src, total_size);

                            int32_t split_num = get_transpose_split_num(row, col, Int8);
                            if (split_num == 0) return T_ERR_NO_IMPLEMENTED;
                            THINKER_RET_CHECK(luna_split_mat_trans_q7(src_temp, dst, row, col, split_num), "luna_split_mat_trans_q7");
                        }
                        else
                            return T_ERR_NO_WORKSPACE;
                        break;
                    case Int16:
                        if (ALIGN4(row) * ALIGN2(col) <= 32768 && total_size <= workspace_size) {
                            int16_t *src_temp = (int16_t *)workspace->dptr_;
                            memcpy(src_temp, src, total_size);
                            THINKER_RET_CHECK(luna_mat_trans_q15(src_temp, dst, row, col), "luna_mat_trans_q15");
                        }
                        else if (total_size <= workspace_size) {
                            int16_t *src_temp = (int16_t *)workspace->dptr_;
                            memcpy(src_temp, src, total_size);

                            int32_t split_num = get_transpose_split_num(row, col, Int16);
                            if (split_num == 0) return T_ERR_NO_IMPLEMENTED;
                            THINKER_RET_CHECK(luna_split_mat_trans_q15(src_temp, dst, row, col, split_num), "luna_split_mat_trans_q15");
                        }
                        else
                            return T_ERR_NO_WORKSPACE;
                        break;
                    case Int32:
                        if (ALIGN2(row) * ALIGN2(col) <= 16384 && total_size <= workspace_size) {
                            int32_t *src_temp = (int32_t *)workspace->dptr_;
                            memcpy(src_temp, src, total_size);
                            THINKER_RET_CHECK(luna_mat_trans_q31(src_temp, dst, row, col), "luna_mat_trans_q31");
                        }
                        else if (total_size <= workspace_size) {
                            int32_t *src_temp = (int32_t *)workspace->dptr_;
                            memcpy(src_temp, src, total_size);

                            int32_t split_num = get_transpose_split_num(row, col, Int32);
                            if (split_num == 0) return T_ERR_NO_IMPLEMENTED;
                            THINKER_RET_CHECK(luna_split_mat_trans_q31(src_temp, dst, row, col, split_num), "luna_split_mat_trans_q31");
                        }
                        else
                            return T_ERR_NO_WORKSPACE;
                        break;
                    default:
                        return T_ERR_INVALID_DATATYPE;
                }
                break;
            }
            case 3: {
                if (total_size <= workspace_size) {
                    switch (dtype) {
                        case Int8: {
                            int8_t *src_temp = (int8_t *)workspace->dptr_;
                            memcpy(src_temp, src, total_size);
                            THINKER_RET_CHECK(luna_trans_axis_q7(src_temp, dst, shape, axes, dims), "luna_trans_axis_q7");
                            break;
                        }
                        case Int16: {
                            int16_t *src_temp = (int16_t *)workspace->dptr_;
                            memcpy(src_temp, src, total_size);
                            THINKER_RET_CHECK(luna_trans_axis_q15(src_temp, dst, shape, axes, dims), "luna_trans_axis_q15");
                            break;
                        }
                        case Int32: {
                            int32_t *src_temp = (int32_t *)workspace->dptr_;
                            memcpy(src_temp, src, total_size);
                            THINKER_RET_CHECK(luna_trans_axis_q31(src_temp, dst, shape, axes, dims), "luna_trans_axis_q31");
                            break;
                        }
                        default:
                            return T_ERR_INVALID_DATATYPE;
                    }
                }
                else
                    return T_ERR_NO_WORKSPACE;
                break;
            }
            case 4: {
                if (0 == axes[0]) {
                    int32_t batch = shape[0];
                    int32_t one_batch_size = shape[1] * shape[2] * shape[3] * X->byte_;

                    uint32_t new_axis[3];
                    uint32_t new_shape[3];
                    for (int32_t n = 0; n < 3; n++) {
                        new_axis[n] = axes[n + 1] - 1;
                        new_shape[n] = shape[n + 1];
                    }
                    if (one_batch_size <= workspace_size) {
                        switch (dtype) {
                            case Int8:
                                for (int32_t i = 0; i < batch; i++) {
                                    void *dst_temp = (void *)((int8_t *)dst + i * one_batch_size);
                                    int8_t *src_temp = (int8_t *)workspace->dptr_;
                                    memcpy(src_temp, (int8_t *)src + i * one_batch_size, one_batch_size);
                                    THINKER_RET_CHECK(luna_trans_axis_q7(src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_q7");
                                }
                                break;
                            case Int16:
                                for (int32_t i = 0; i < batch; i++) {
                                    void *dst_temp = (void *)((int8_t *)dst + i * one_batch_size);
                                    int16_t *src_temp = (int16_t *)workspace->dptr_;
                                    memcpy(src_temp, (int8_t *)src + i * one_batch_size, one_batch_size);
                                    THINKER_RET_CHECK(luna_trans_axis_q15(src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_q15");
                                }
                                break;
                            case Int32:
                                for (int32_t i = 0; i < batch; i++) {
                                    void *dst_temp = (void *)((int8_t *)dst + i * one_batch_size);
                                    int32_t *src_temp = (int32_t *)workspace->dptr_;
                                    memcpy(src_temp, (int8_t *)src + i * one_batch_size, one_batch_size);
                                    THINKER_RET_CHECK(luna_trans_axis_q31(src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_q31");
                                }
                                break;
                            default:
                                return T_ERR_INVALID_DATATYPE;
                        }
                    }
                    else {
                        return T_ERR_NO_WORKSPACE;
                    }
                }
                else {
                    return T_ERR_NO_IMPLEMENTED;
                }
                break;
            }
        }
    }
    else {
        switch (dims) {
            case 2: {
                int32_t row = shape[0];
                int32_t col = shape[1];
                switch (dtype) {
                    case Int8: {
                        int8_t *dst_temp = (int8_t *)workspace->dptr_;
                        if ((ALIGN4(row) * ALIGN8(col) <= 65536) && (workspace_size >= total_size)) {
                            int8_t *src_temp = (int8_t *)src;
                            if (srcInPSRAM) {
                                src_temp = (int8_t *)workspace->dptr_;
                                memcpy(src_temp, src, total_size);
                            }
                            THINKER_RET_CHECK(luna_mat_trans_q7(src_temp, dst_temp, row, col), "luna_mat_trans_q7");
                            memcpy(dst, dst_temp, total_size);
                        }
                        else if (ALIGN4(row) * ALIGN8(col) > 65536) {
                            int8_t *src_temp = (int8_t *)src;
                            if (srcInPSRAM) {
                                if (workspace_size < total_size * 2)
                                    return T_ERR_NO_WORKSPACE;
                                else {
                                    src_temp = (int8_t *)workspace->dptr_ + total_size;
                                    memcpy(src_temp, src, total_size);
                                }
                            }

                            int32_t split_num = get_transpose_split_num(row, col, Int8);
                            if (split_num == 0) return T_ERR_NO_IMPLEMENTED;

                            THINKER_RET_CHECK(luna_split_mat_trans_q7(src_temp, dst_temp, row, col, split_num), "luna_split_mat_trans_q7");
                            memcpy(dst, dst_temp, total_size);
                        }
                        else
                            return T_ERR_NO_WORKSPACE;
                        break;
                    }
                    case Int16: {
                        int16_t *dst_temp = (int16_t *)workspace->dptr_;
                        if ((ALIGN4(row) * ALIGN2(col) <= 32768) && (workspace_size >= total_size)) {
                            int8_t *src_temp = (int8_t *)src;
                            if (srcInPSRAM) {
                                src_temp = (int8_t *)workspace->dptr_;
                                memcpy(src_temp, src, total_size);
                            }
                            THINKER_RET_CHECK(luna_mat_trans_q15((int16_t *)src_temp, dst_temp, row, col), "luna_mat_trans_q15");
                            memcpy(dst, dst_temp, total_size);
                        }
                        else if (ALIGN4(row) * ALIGN2(col) > 32768) {
                            int8_t *src_temp = (int8_t *)src;
                            if (srcInPSRAM) {
                                if (workspace_size < total_size * 2)
                                    return T_ERR_NO_WORKSPACE;
                                else {
                                    src_temp = (int8_t *)workspace->dptr_ + total_size;
                                    memcpy(src_temp, src, total_size);
                                }
                            }

                            int32_t split_num = get_transpose_split_num(row, col, Int16);
                            if (split_num == 0) return T_ERR_NO_IMPLEMENTED;

                            THINKER_RET_CHECK(luna_split_mat_trans_q15((int16_t *)src_temp, dst_temp, row, col, split_num), "luna_split_mat_trans_q15");
                            memcpy(dst, dst_temp, total_size);
                        }
                        else
                            return T_ERR_NO_WORKSPACE;
                        break;
                    }
                    case Int32: {
                        int32_t *dst_temp = (int32_t *)workspace->dptr_;
                        if ((ALIGN2(row) * ALIGN2(col) <= 16384) && (workspace_size >= total_size)) {
                            int8_t *src_temp = (int8_t *)src;
                            if (srcInPSRAM) {
                                src_temp = (int8_t *)workspace->dptr_;
                                memcpy(src_temp, src, total_size);
                            }
                            THINKER_RET_CHECK(luna_mat_trans_q31((int32_t *)src_temp, dst_temp, row, col), "luna_mat_trans_q31");
                            memcpy(dst, dst_temp, total_size);
                        }
                        else if (ALIGN2(row) * ALIGN2(col) > 16384) {
                            int8_t *src_temp = (int8_t *)src;
                            if (srcInPSRAM) {
                                if (workspace_size < total_size * 2)
                                    return T_ERR_NO_WORKSPACE;
                                else {
                                    src_temp = (int8_t *)workspace->dptr_ + total_size;
                                    memcpy(src_temp, src, total_size);
                                }
                            }

                            int32_t split_num = get_transpose_split_num(row, col, Int32);
                            if (split_num == 0) return T_ERR_NO_IMPLEMENTED;

                            THINKER_RET_CHECK(luna_split_mat_trans_q31((int32_t *)src_temp, dst_temp, row, col, split_num), "luna_split_mat_trans_q31");
                            memcpy(dst, dst_temp, total_size);
                        }
                        else
                            return T_ERR_NO_WORKSPACE;
                        break;
                    }
                    default:
                        return T_ERR_INVALID_DATATYPE;
                }
                break;
            }
            case 3: {
                switch (dtype) {
                    case Int8: {
                        int8_t *dst_temp = (int8_t *)workspace->dptr_;
                        if ((!srcInPSRAM) && (total_size <= workspace_size)) {
                            THINKER_RET_CHECK(luna_trans_axis_q7(src, dst_temp, shape, axes, dims), "luna_trans_axis_q7");
                            memcpy(dst, dst_temp, total_size);
                        }
                        else if (srcInPSRAM && (total_size * 2 <= workspace_size)) {
                            int8_t *src_temp = (int8_t *)workspace->dptr_ + total_size;
                            memcpy(src_temp, src, total_size);
                            THINKER_RET_CHECK(luna_trans_axis_q7(src_temp, dst_temp, shape, axes, dims), "luna_trans_axis_q7");
                            memcpy(dst, dst_temp, total_size);
                        }
                        else if (0 == axes[0]) {// convert to 2D
                            int32_t batch = shape[0];
                            int32_t one_batch_size = shape[1] * shape[2] * X->byte_;
                            uint32_t new_axis[2];
                            uint32_t new_shape[2];
                            for (int32_t n = 0; n < 2; n++) {
                                new_axis[n] = axes[n + 1] - 1;
                                new_shape[n] = shape[n + 1];
                            }

                            int8_t *dst_temp = (int8_t *)workspace->dptr_;
                            if ((ALIGN4(shape[1]) * ALIGN8(shape[2]) <= 65536) && (workspace_size >= one_batch_size)) {
                                for (int32_t i = 0; i < batch; i++) {
                                    int8_t *src_temp = (int8_t *)src + i * one_batch_size;
                                    if (srcInPSRAM) {
                                        src_temp = (int8_t *)workspace->dptr_;
                                        memcpy(src_temp, (int8_t *)src + i * one_batch_size, one_batch_size);
                                    }
                                    if (shape[1] == 1 || shape[2] == 1)
                                        THINKER_RET_CHECK(API_LIB(memcpy)(dst_temp, src_temp, one_batch_size), "luna_memcpy");
                                    else
                                        THINKER_RET_CHECK(luna_mat_trans_q7(src_temp, dst_temp, shape[1], shape[2]), "luna_mat_trans_q7");
                                    memcpy((int8_t *)dst + i * one_batch_size, dst_temp, one_batch_size);
                                }
                            }
                            else if (ALIGN4(shape[1]) * ALIGN8(shape[2]) > 65536) {
                                for (int32_t i = 0; i < batch; i++) {
                                    int8_t *src_temp = (int8_t *)src + i * one_batch_size;
                                    if (srcInPSRAM) {
                                        if (workspace_size < one_batch_size * 2)
                                            return T_ERR_NO_WORKSPACE;
                                        else {
                                            src_temp = (int8_t *)workspace->dptr_ + one_batch_size;
                                            memcpy(src_temp, src, one_batch_size);
                                        }
                                    }
                                    int32_t row = shape[1];
                                    int32_t col = shape[2];
                                    int32_t split_num = get_transpose_split_num(row, col, Int8);
                                    if (split_num == 0) return T_ERR_NO_IMPLEMENTED;

                                    THINKER_RET_CHECK(luna_split_mat_trans_q7(src_temp, dst_temp, shape[1], shape[2], split_num), "luna_split_mat_trans_q7");
                                    memcpy((int8_t *)dst + i * one_batch_size, dst_temp, one_batch_size);
                                }
                            }
                            else
                                return T_ERR_NO_WORKSPACE;
                        }
                        else
                            return T_ERR_NO_WORKSPACE;
                        break;
                    }
                    case Int16: {
                        int16_t *dst_temp = (int16_t *)workspace->dptr_;
                        if ((!srcInPSRAM) && (total_size <= workspace_size)) {
                            THINKER_RET_CHECK(luna_trans_axis_q15(src, dst_temp, shape, axes, dims), "luna_trans_axis_q15");
                            memcpy(dst, dst_temp, total_size);
                        }
                        else if (srcInPSRAM && (total_size * 2 <= workspace_size)) {
                            int8_t *src_temp = (int8_t *)workspace->dptr_ + total_size;
                            memcpy(src_temp, src, total_size);
                            THINKER_RET_CHECK(luna_trans_axis_q15((int16_t *)src_temp, dst_temp, shape, axes, dims), "luna_trans_axis_q15");
                            memcpy(dst, dst_temp, total_size);
                        }
                        else if (0 == axes[0]) {// convert to 2D
                            int32_t batch = shape[0];
                            int32_t one_batch_size = shape[1] * shape[2] * X->byte_;
                            uint32_t new_axis[2];
                            uint32_t new_shape[2];
                            for (int32_t n = 0; n < 2; n++) {
                                new_axis[n] = axes[n + 1] - 1;
                                new_shape[n] = shape[n + 1];
                            }

                            int16_t *dst_temp = (int16_t *)workspace->dptr_;
                            if ((ALIGN4(shape[1]) * ALIGN2(shape[2]) <= 32768) && (workspace_size >= one_batch_size)) {
                                for (int32_t i = 0; i < batch; i++) {
                                    int8_t *src_temp = (int8_t *)src + i * one_batch_size;
                                    if (srcInPSRAM) {
                                        src_temp = (int8_t *)workspace->dptr_;
                                        memcpy(src_temp, (int8_t *)src + i * one_batch_size, one_batch_size);
                                    }
                                    if (shape[1] == 1 || shape[2] == 1)
                                        THINKER_RET_CHECK(API_LIB(memcpy)(dst_temp, src_temp, one_batch_size), "luna_memcpy");
                                    else
                                        THINKER_RET_CHECK(luna_mat_trans_q15((int16_t *)src_temp, dst_temp, shape[1], shape[2]), "luna_mat_trans_q15");
                                    memcpy((int8_t *)dst + i * one_batch_size, dst_temp, one_batch_size);
                                }
                            }
                            else if (ALIGN4(shape[1]) * ALIGN2(shape[2]) > 32768) {
                                for (int32_t i = 0; i < batch; i++) {
                                    int8_t *src_temp = (int8_t *)src + i * one_batch_size;
                                    if (srcInPSRAM) {
                                        if (workspace_size < one_batch_size * 2)
                                            return T_ERR_NO_WORKSPACE;
                                        else {
                                            src_temp = (int8_t *)workspace->dptr_ + one_batch_size;
                                            memcpy(src_temp, src, one_batch_size);
                                        }
                                    }
                                    int32_t row = shape[1];
                                    int32_t col = shape[2];
                                    int32_t split_num = get_transpose_split_num(row, col, Int16);
                                    if (split_num == 0) return T_ERR_NO_IMPLEMENTED;

                                    THINKER_RET_CHECK(luna_split_mat_trans_q15((int16_t *)src_temp, dst_temp, shape[1], shape[2], split_num), "luna_split_mat_trans_q15");
                                    memcpy((int8_t *)dst + i * one_batch_size, dst_temp, one_batch_size);
                                }
                            }
                            else
                                return T_ERR_NO_WORKSPACE;
                        }
                        else
                            return T_ERR_NO_WORKSPACE;
                        break;
                    }
                    case Int32: {
                        int32_t *dst_temp = (int32_t *)workspace->dptr_;
                        if ((!srcInPSRAM) && (total_size <= workspace_size)) {
                            THINKER_RET_CHECK(luna_trans_axis_q31(src, dst_temp, shape, axes, dims), "luna_trans_axis_q31");
                            memcpy(dst, dst_temp, total_size);
                        }
                        else if (srcInPSRAM && (total_size * 2 <= workspace_size)) {
                            int8_t *src_temp = (int8_t *)workspace->dptr_ + total_size;
                            memcpy(src_temp, src, total_size);
                            THINKER_RET_CHECK(luna_trans_axis_q31((int32_t *)src_temp, dst_temp, shape, axes, dims), "luna_trans_axis_q31");
                            memcpy(dst, dst_temp, total_size);
                        }
                        else if (0 == axes[0]) {// convert to 2D
                            int32_t batch = shape[0];
                            int32_t one_batch_size = shape[1] * shape[2] * X->byte_;
                            uint32_t new_axis[2];
                            uint32_t new_shape[2];
                            for (int32_t n = 0; n < 2; n++) {
                                new_axis[n] = axes[n + 1] - 1;
                                new_shape[n] = shape[n + 1];
                            }

                            int32_t *dst_temp = (int32_t *)workspace->dptr_;
                            if ((ALIGN2(shape[1]) * ALIGN2(shape[2]) <= 16384) && (workspace_size >= one_batch_size)) {
                                for (int32_t i = 0; i < batch; i++) {
                                    int8_t *src_temp = (int8_t *)src + i * one_batch_size;
                                    if (srcInPSRAM) {
                                        src_temp = (int8_t *)workspace->dptr_;
                                        memcpy(src_temp, (int8_t *)src + i * one_batch_size, one_batch_size);
                                    }
                                    if (shape[1] == 1 || shape[2] == 1)
                                        THINKER_RET_CHECK(API_LIB(memcpy)(dst_temp, src_temp, one_batch_size), "luna_memcpy");
                                    else
                                        THINKER_RET_CHECK(luna_mat_trans_q31((int32_t *)src_temp, dst_temp, shape[1], shape[2]), "luna_mat_trans_q31");
                                    memcpy((int8_t *)dst + i * one_batch_size, dst_temp, one_batch_size);
                                }
                            }
                            else if (ALIGN2(shape[1]) * ALIGN2(shape[2]) > 16384) {
                                for (int32_t i = 0; i < batch; i++) {
                                    int8_t *src_temp = (int8_t *)src + i * one_batch_size;
                                    if (srcInPSRAM) {
                                        if (workspace_size < one_batch_size * 2)
                                            return T_ERR_NO_WORKSPACE;
                                        else {
                                            src_temp = (int8_t *)workspace->dptr_ + one_batch_size;
                                            memcpy(src_temp, src, one_batch_size);
                                        }
                                    }
                                    int32_t row = shape[1];
                                    int32_t col = shape[2];
                                    int32_t split_num = get_transpose_split_num(row, col, Int32);
                                    if (split_num == 0) return T_ERR_NO_IMPLEMENTED;

                                    THINKER_RET_CHECK(luna_split_mat_trans_q31((int32_t *)src_temp, dst_temp, shape[1], shape[2], split_num), "luna_split_mat_trans_q31");
                                    memcpy((int8_t *)dst + i * one_batch_size, dst_temp, one_batch_size);
                                }
                            }
                            else
                                return T_ERR_NO_WORKSPACE;
                        }
                        else
                            return T_ERR_NO_WORKSPACE;
                        break;
                    }
                    default:
                        return T_ERR_INVALID_DATATYPE;
                }
                break;
            }
            case 4: {
                if (0 == axes[0]) { // convert to 3D
                    int32_t batch = shape[0];
                    int32_t one_batch_size = shape[1] * shape[2] * shape[3] * X->byte_;

                    uint32_t new_axis[3];
                    uint32_t new_shape[3];
                    for (int32_t n = 0; n < 3; n++) {
                        new_axis[n] = axes[n + 1] - 1;
                        new_shape[n] = shape[n + 1];
                    }
                    switch (dtype) {
                        case Int8:
                            if (srcInPSRAM) {
                                if (one_batch_size * 2 <= workspace_size) {
                                    for (int32_t i = 0; i < batch; i++) {
                                        int8_t *src_temp = (int8_t *)workspace->dptr_ + one_batch_size;
                                        int8_t *dst_temp = (int8_t *)workspace->dptr_;
                                        memcpy(src_temp, (int8_t *)src + i * one_batch_size, one_batch_size);
                                        THINKER_RET_CHECK(luna_trans_axis_q7(src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_q7");
                                        memcpy((int8_t *)dst + i * one_batch_size, dst_temp, one_batch_size);
                                    }
                                }
                                else
                                    return T_ERR_NO_WORKSPACE;
                            }
                            else {
                                if (one_batch_size <= workspace_size) {
                                    for (int32_t i = 0; i < batch; i++) {
                                        int8_t *src_temp = (int8_t *)src + i * one_batch_size;
                                        int8_t *dst_temp = (int8_t *)workspace->dptr_;
                                        THINKER_RET_CHECK(luna_trans_axis_q7(src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_q7");
                                        memcpy((int8_t *)dst + i * one_batch_size, dst_temp, one_batch_size);
                                    }
                                }
                                else
                                    return T_ERR_NO_WORKSPACE;
                            }
                            break;
                        case Int16:
                            if (srcInPSRAM) {
                                if (one_batch_size * 2 <= workspace_size) {
                                    for (int32_t i = 0; i < batch; i++) {
                                        int8_t *src_temp = (int8_t *)workspace->dptr_ + one_batch_size;
                                        int16_t *dst_temp = (int16_t *)workspace->dptr_;
                                        memcpy(src_temp, (int8_t *)src + i * one_batch_size, one_batch_size);
                                        THINKER_RET_CHECK(luna_trans_axis_q15((int16_t *)src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_q15");
                                        memcpy((int8_t *)dst + i * one_batch_size, dst_temp, one_batch_size);
                                    }
                                }
                                else
                                    return T_ERR_NO_WORKSPACE;
                            }
                            else {
                                if (one_batch_size <= workspace_size) {
                                    for (int32_t i = 0; i < batch; i++) {
                                        int16_t *src_temp = (int16_t *)((int8_t *)src + i * one_batch_size);
                                        int16_t *dst_temp = (int16_t *)workspace->dptr_;
                                        THINKER_RET_CHECK(luna_trans_axis_q15(src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_q15");
                                        memcpy((int8_t *)dst + i * one_batch_size, dst_temp, one_batch_size);
                                    }
                                }
                                else
                                    return T_ERR_NO_WORKSPACE;
                            }
                            break;
                        case Int32:
                            if (srcInPSRAM) {
                                if (one_batch_size * 2 <= workspace_size) {
                                    for (int32_t i = 0; i < batch; i++) {
                                        int8_t *src_temp = (int8_t *)workspace->dptr_ + one_batch_size;
                                        int32_t *dst_temp = (int32_t *)workspace->dptr_;
                                        memcpy(src_temp, (int8_t *)src + i * one_batch_size, one_batch_size);
                                        THINKER_RET_CHECK(luna_trans_axis_q31((int32_t *)src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_q31");
                                        memcpy((int8_t *)dst + i * one_batch_size, dst_temp, one_batch_size);
                                    }
                                }
                                else
                                    return T_ERR_NO_WORKSPACE;
                            }
                            else {
                                if (one_batch_size <= workspace_size) {
                                    for (int32_t i = 0; i < batch; i++) {
                                        int32_t *src_temp = (int32_t *)((int8_t *)src + i * one_batch_size);
                                        int32_t *dst_temp = (int32_t *)workspace->dptr_;
                                        THINKER_RET_CHECK(luna_trans_axis_q31(src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_q31");
                                        memcpy((int8_t *)dst + i * one_batch_size, dst_temp, one_batch_size);
                                    }
                                }
                                else
                                    return T_ERR_NO_WORKSPACE;
                            }
                            break;
                        default:
                            return T_ERR_INVALID_DATATYPE;
                    }
                }
                else {
                    return T_ERR_NO_IMPLEMENTED;
                }
                break;
            }
        }
    }
    return T_SUCCESS;
}

/**
 * @brief Split and transpose a matrix with specified dimensions
 * @param Y Output tensor
 * @param X Input tensor
 * @param Temp Temporary buffer for intermediate calculations
 * @param row Number of rows in the matrix
 * @param col Number of columns in the matrix
 * @param split_num Number of splits along the column dimension
 * @return Execution status
 */
int32_t split_transpose_luna(tTensor *Y, tTensor *X, tTensor *Temp, int32_t row, int32_t col, int32_t split_num) {  
    #if THINKER_PARAM_CHECK
    if (X == NULL || Y == NULL || X->dptr_ == 0 || Y->dptr_ == 0 ||
                        row <= 0 || col <= 0 || split_num <= 0 || split_num > col) {
        return (T_ERR_INVALID_PARA);
    }
    if (Y->dtype_ != X->dtype_) {
        return (T_ERR_INVALID_DATATYPE);
    }
    #endif
    if (row == 1 || col == 1) {
        THINKER_RET_CHECK(API_LIB(memcpy)((void *)Y->dptr_, (const void *)X->dptr_,
                                           (size_t)row * col * X->byte_), "luna_memcpy");
        return T_SUCCESS;
    }
    if (X->mem_.type_ != 2) {
        switch (X->dtype_) {
            case Int8: {
                const int8_t *src = (int8_t *)X->dptr_;
                int8_t *dst = (int8_t *)Y->dptr_;
                for (int32_t i = 0; i < split_num - 1; i++) {
                    THINKER_RET_CHECK(luna_mat_trans_inv_q7(src, dst, row, split_num, row, col), "luna_mat_trans_inv_q7");
                }
                THINKER_RET_CHECK(luna_mat_trans_inv_q7(src, dst, row, col - (split_num - 1) * split_num, row, col), "luna_mat_trans_inv_q7");
                break;
            }
            case Int16: {
                const int16_t *src = (int16_t *)X->dptr_;
                int16_t *dst = (int16_t *)Y->dptr_;
                for (int32_t i = 0; i < split_num - 1; i++) {
                    THINKER_RET_CHECK(luna_mat_trans_inv_q15(src, dst, row, split_num, row, col), "luna_mat_trans_inv_q15");
                }
                THINKER_RET_CHECK(luna_mat_trans_inv_q15(src, dst, row, col - (split_num - 1) * split_num, row, col), "luna_mat_trans_inv_q15");
                break;
            }
            case Int32: {
                const int32_t *src = (int32_t *)X->dptr_;
                int32_t *dst = (int32_t *)Y->dptr_;
                for (int32_t i = 0; i < split_num - 1; i++) {
                    THINKER_RET_CHECK(luna_mat_trans_inv_q31(src, dst, row, split_num, row, col), "luna_mat_trans_inv_q31");
                }
                THINKER_RET_CHECK(luna_mat_trans_inv_q31(src, dst, row, col - (split_num - 1) * split_num, row, col), "luna_mat_trans_inv_q31");
                break;
            }
            default:
                return T_ERR_NO_IMPLEMENTED;
        }
    }
    
    return T_SUCCESS;
}

/**
 * @brief Transpose matrix along specified axes
 * @param dtype Data type of the matrix (Int8, Int16, Int32)
 * @param src Input matrix pointer
 * @param dst Output matrix pointer
 * @param in_shape Input matrix dimensions
 * @param axis Axes along which to transpose
 * @param n_dims Number of dimensions
 * @return Execution status
 */
int32_t transpose_axis_luna(int16_t dtype, void *src, void *dst, int32_t *in_shape, int32_t *axis, uint32_t n_dims) {
    #if THINKER_PARAM_CHECK
    if (src == NULL || dst == NULL || in_shape == NULL || axis == NULL) {
        return (T_ERR_INVALID_PARA);
    }
    if (n_dims != 3) {
        return (T_ERR_NO_IMPLEMENTED);
    }
    #endif
    for (uint32_t i = 0; i < n_dims; ++i) {
        #if THINKER_PARAM_CHECK
        if (in_shape[i] <= 0 || axis[i] < 0 ||
                            axis[i] >= (int32_t)n_dims) {
            return (T_ERR_INVALID_PARA);
        }
        #endif
        for (uint32_t j = 0; j < i; ++j) {
#if THINKER_PARAM_CHECK
            if (axis[j] == axis[i]) {
                return (T_ERR_INVALID_PARA);
            }
#endif
        }
    }
    
    switch (dtype) {
        case Int8:
            THINKER_RET_CHECK(API_LIB(trans_axis_q7)((int8_t *)src, (int8_t *)dst, (uint32_t *)in_shape, (uint32_t *)axis, n_dims), "luna_trans_axis_q7");
            break;
        case Int16:
            THINKER_RET_CHECK(API_LIB(trans_axis_q15)((int16_t *)src, (int16_t *)dst, (uint32_t *)in_shape, (uint32_t *)axis, n_dims), "luna_trans_axis_q15");
            break;
        case Int32:
            THINKER_RET_CHECK(API_LIB(trans_axis_q31)((int32_t *)src, (int32_t *)dst, (uint32_t *)in_shape, (uint32_t *)axis, n_dims), "luna_trans_axis_q31");
            break;
        default:
            return T_ERR_NO_IMPLEMENTED;
    }
    
    return T_SUCCESS;
}

#endif
