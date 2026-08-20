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

/**
 * @brief Transpose a matrix of specified data type
 * @param dtype Data type of the matrix (Int8, Int32)
 * @param dst Output matrix pointer
 * @param src Input matrix pointer
 * @param row Number of rows in the matrix
 * @param col Number of columns in the matrix
 * @return Execution status
 */
tStatus transpose_luna(tTensor *X, tTensor *Y, tTensor * workspace, uint32_t dims, uint32_t *axes, uint32_t *shape) {
    void *src = (void *)X->dptr_;
    void *dst = (void *)Y->dptr_;
    size_t workspace_size = workspace ? getTensorDataSize(workspace) : 0;
    size_t total_size = getTensorDataSize(X);
    int16_t dtype = X->byte_ == 1 ? Int8 : X->byte_ == 4 ? Int32 : -1;
    #if THINKER_PARAM_CHECK
    if (dtype == -1 || Y->dtype_ != X->dtype_ ||
                        Y->byte_ != X->byte_ ||
                        (X->dtype_ != Int8 && X->dtype_ != Int32 && X->dtype_ != Float32)) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (dims < 2 || dims > 4) {
        return (T_ERR_INVALID_PARA);
    }

    if (dims == 2 && (axes[0] != 1 || axes[1] != 0)) {
        return (T_ERR_INVALID_PARA);
    }

    if (dims == 3 && !((axes[0] == 0 && axes[1] == 2 && axes[2] == 1) ||
                                       (axes[0] == 2 && axes[1] == 0 && axes[2] == 1) ||
                                       (axes[0] == 2 && axes[1] == 1 && axes[2] == 0) ||
                                       (axes[0] == 1 && axes[1] == 2 && axes[2] == 0) ||
                                       (axes[0] == 1 && axes[1] == 0 && axes[2] == 2))) {
        return (T_ERR_INVALID_PARA);
    }

    if (dims == 4 && !(axes[0] == 0 &&
                                        ((axes[1] == 1 && axes[2] == 3 && axes[3] == 2) ||
                                        (axes[1] == 3 && axes[2] == 1 && axes[3] == 2) ||
                                        (axes[1] == 3 && axes[2] == 2 && axes[3] == 1) ||
                                        (axes[1] == 2 && axes[2] == 3 && axes[3] == 1) ||
                                         (axes[1] == 2 && axes[2] == 1 && axes[3] == 3)))) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    bool srcInPSRAM = (X->mem_.type_ != 2);
    bool dstInPSRAM = (Y->mem_.type_ != 2);

    /* split_mat_trans and trans_axis do not support in-place execution. */
    #if THINKER_PARAM_CHECK
    if (X->dptr_ == Y->dptr_ &&
                        (dims == 3 || dims == 4 ||
                         (dims == 2 && ((dtype == Int8 && ALIGN8(shape[0]) * ALIGN2(shape[1]) > 16384) ||
                                        (dtype == Int32 && ALIGN2(shape[0]) * ALIGN2(shape[1]) > 4096))))) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    if ((!srcInPSRAM) & (!dstInPSRAM)) {
        switch (dims) {
            case 2: {
                uint32_t row = shape[0];
                uint32_t col = shape[1];
                if (row == 1 || col == 1) {
                    THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)dst, (int8_t *)src, total_size),
                                    "luna_memcpy_i8o8");
                    return T_SUCCESS;
                }
                switch (dtype) {
                    case Int8:
                        if (ALIGN8(row) * ALIGN2(col) <= 16384)
                            THINKER_RET_CHECK(API_LIB(mat_trans_i8o8)((int8_t *)src, (int8_t *)dst, row, col), "luna_mat_trans_i8o8");
                        else
                            THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)((int8_t *)src, (int8_t *)dst, row, col), "luna_split_mat_trans_i8o8");
                        break;
                    case Int32:
                        if (ALIGN2(row) * ALIGN2(col) <= 4096)
                            THINKER_RET_CHECK(API_LIB(mat_trans_i32o32)((int32_t *)src, (int32_t *)dst, row, col), "luna_mat_trans_i32o32");
                        else
                            THINKER_RET_CHECK(API_LIB(split_mat_trans_i32o32)((int32_t *)src, (int32_t *)dst, row, col), "luna_split_mat_trans_i32o32");
                        break;
                    default:
                        return T_ERR_INVALID_DATATYPE;
                }
                break;
            }
            case 3: {
                switch (dtype) {
                    case Int8:
                        THINKER_RET_CHECK(API_LIB(trans_axis_i8o8)((int8_t *)src, (int8_t *)dst, shape, axes, dims), "luna_trans_axis_i8o8");
                        break;
                    case Int32:
                        THINKER_RET_CHECK(API_LIB(trans_axis_i32o32)((int32_t *)src, (int32_t *)dst, shape, axes, dims), "luna_trans_axis_i32o32");
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
                                THINKER_RET_CHECK(API_LIB(trans_axis_i8o8)((int8_t *)src_temp, (int8_t *)dst_temp, new_shape, new_axis, 3), "luna_trans_axis_i8o8");
                            }
                            break;
                        case Int32:
                            for (int32_t i = 0; i < batch; i++) {
                                void *src_temp = (void *)((int8_t *)src + i * one_batch_size);
                                void *dst_temp = (void *)((int8_t *)dst + i * one_batch_size);
                                THINKER_RET_CHECK(API_LIB(trans_axis_i32o32)((int32_t *)src_temp, (int32_t *)dst_temp, new_shape, new_axis, 3), "luna_trans_axis_i32o32");
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
                        if (total_size > workspace_size) {
                            return T_ERR_NO_WORKSPACE;
                        }
                        int8_t *src_temp = (int8_t *)workspace->dptr_;
                        THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_size), "luna_memcpy_i8o8");
                        if (ALIGN8(row) * ALIGN2(col) <= 16384) {
                            THINKER_RET_CHECK(API_LIB(mat_trans_i8o8)(src_temp, (int8_t *)dst, row, col), "luna_mat_trans_i8o8");
                        } else {
                            THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(src_temp, (int8_t *)dst, row, col), "luna_split_mat_trans_i8o8");
                        }
                        break;
                    case Int32:
                        if (total_size > workspace_size) {
                            return T_ERR_NO_WORKSPACE;
                        }
                        int32_t *src_temp_i32 = (int32_t *)workspace->dptr_;
                        THINKER_RET_CHECK(luna_memcpy_i8o8((int8_t *)src_temp_i32, (int8_t *)src, total_size), "luna_memcpy_i8o8");
                        if (ALIGN2(row) * ALIGN2(col) <= 4096) {
                            THINKER_RET_CHECK(API_LIB(mat_trans_i32o32)(src_temp_i32, (int32_t *)dst, row, col), "luna_mat_trans_i32o32");
                        } else {
                            THINKER_RET_CHECK(API_LIB(split_mat_trans_i32o32)(src_temp_i32, (int32_t *)dst, row, col), "luna_split_mat_trans_i32o32");
                        }
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
                            THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_size), "luna_memcpy_i8o8");
                            THINKER_RET_CHECK(API_LIB(trans_axis_i8o8)(src_temp, (int8_t *)dst, shape, axes, dims), "luna_trans_axis_i8o8");
                            break;
                        }
                        case Int32: {
                            int32_t *src_temp = (int32_t *)workspace->dptr_;
                            THINKER_RET_CHECK(luna_memcpy_i8o8((int8_t *)src_temp, (int8_t *)src, total_size), "luna_memcpy_i8o8");
                            THINKER_RET_CHECK(API_LIB(trans_axis_i32o32)(src_temp, (int32_t *)dst, shape, axes, dims), "luna_trans_axis_i32o32");
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
                                    void *src_temp = (void *)((int8_t *)src + i * one_batch_size);
                                    void *dst_temp = (void *)((int8_t *)dst + i * one_batch_size);
                                    int8_t *src_temp1 = (int8_t *)workspace->dptr_;
                                    THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp1, (int8_t *)src_temp, one_batch_size), "luna_memcpy_i8o8");
                                    THINKER_RET_CHECK(API_LIB(trans_axis_i8o8)(src_temp1, (int8_t *)dst_temp, new_shape, new_axis, 3), "luna_trans_axis_i8o8");
                                }
                                break;
                            case Int32:
                                for (int32_t i = 0; i < batch; i++) {
                                    void *src_temp = (void *)((int8_t *)src + i * one_batch_size);
                                    void *dst_temp = (void *)((int8_t *)dst + i * one_batch_size);
                                    int32_t *src_temp1 = (int32_t *)workspace->dptr_;
                                    THINKER_RET_CHECK(luna_memcpy_i8o8((int8_t *)src_temp1, (int8_t *)src_temp, one_batch_size), "luna_memcpy_i8o8");
                                    THINKER_RET_CHECK(API_LIB(trans_axis_i32o32)(src_temp1, (int32_t *)dst_temp, new_shape, new_axis, 3), "luna_trans_axis_i32o32");
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
                        if ((ALIGN8(row) * ALIGN2(col) <= 16384) && (workspace_size >= total_size)) {
                            int8_t *src_temp = (int8_t *)src;
                            if (srcInPSRAM) {
                                src_temp = (int8_t *)workspace->dptr_;
                                THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_size), "luna_memcpy_i8o8");
                            }
                            THINKER_RET_CHECK(API_LIB(mat_trans_i8o8)(src_temp, dst_temp, row, col), "luna_mat_trans_i8o8");
                            opi_psram_cpy_out((int8_t *)dst, (int8_t *)dst_temp, total_size);
                        }
                        else if (ALIGN8(row) * ALIGN2(col) > 16384) {
                            int8_t *src_temp = (int8_t *)src;
                            if (srcInPSRAM) {
                                if (workspace_size < total_size * 2)
                                    return T_ERR_NO_WORKSPACE;
                                else {
                                    src_temp = (int8_t *)workspace->dptr_ + total_size;
                                    THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_size), "luna_memcpy_i8o8");
                                }
                            }
                            THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(src_temp, dst_temp, row, col), "luna_split_mat_trans_i8o8");
                            opi_psram_cpy_out((int8_t *)dst, (int8_t *)dst_temp, total_size);
                        }
                        else
                            return T_ERR_NO_WORKSPACE;
                        break;
                    }
                    case Int32: {
                        int32_t *dst_temp = (int32_t *)workspace->dptr_;
                        if ((ALIGN2(row) * ALIGN2(col) <= 4096) && (workspace_size >= total_size)) {
                            int8_t *src_temp = (int8_t *)src;
                            if (srcInPSRAM) {
                                src_temp = (int8_t *)workspace->dptr_;
                                THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_size), "luna_memcpy_i8o8");
                            }
                            THINKER_RET_CHECK(API_LIB(mat_trans_i32o32)((int32_t *)src_temp, dst_temp, row, col), "luna_mat_trans_i32o32");
                            opi_psram_cpy_out((int8_t *)dst, (int8_t *)dst_temp, total_size);
                        }
                        else if (ALIGN2(row) * ALIGN2(col) > 4096) {
                            int8_t *src_temp = (int8_t *)src;
                            if (srcInPSRAM) {
                                if (workspace_size < total_size * 2)
                                    return T_ERR_NO_WORKSPACE;
                                else {
                                    src_temp = (int8_t *)workspace->dptr_ + total_size;
                                    THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_size), "luna_memcpy_i8o8");
                                }
                            }
                            THINKER_RET_CHECK(API_LIB(split_mat_trans_i32o32)((int32_t *)src_temp, dst_temp, row, col), "luna_split_mat_trans_i32o32");
                            opi_psram_cpy_out((int8_t *)dst, (int8_t *)dst_temp, total_size);
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
                            THINKER_RET_CHECK(API_LIB(trans_axis_i8o8)((int8_t *)src, dst_temp, shape, axes, dims), "luna_trans_axis_i8o8");
                            opi_psram_cpy_out((int8_t *)dst, (int8_t *)dst_temp, total_size);
                        }
                        else if (srcInPSRAM && (total_size * 2 <= workspace_size)) {
                            int8_t *src_temp = (int8_t *)workspace->dptr_ + total_size;
                            THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_size), "luna_memcpy_i8o8");
                            THINKER_RET_CHECK(API_LIB(trans_axis_i8o8)(src_temp, dst_temp, shape, axes, dims), "luna_trans_axis_i8o8");
                            opi_psram_cpy_out((int8_t *)dst, (int8_t *)dst_temp, total_size);
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
                            if ((ALIGN8(shape[1]) * ALIGN2(shape[2]) <= 16384) && (workspace_size >= one_batch_size)) {
                                for (int32_t i = 0; i < batch; i++) {
                                    int8_t *src_temp = (int8_t *)src + i * one_batch_size;
                                    if (srcInPSRAM) {
                                        src_temp = (int8_t *)workspace->dptr_;
                                        THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src + i * one_batch_size, one_batch_size), "luna_memcpy_i8o8");
                                    }
                                    THINKER_RET_CHECK(API_LIB(mat_trans_i8o8)(src_temp, dst_temp, shape[1], shape[2]), "luna_mat_trans_i8o8");
                                    opi_psram_cpy_out((int8_t *)dst + i * one_batch_size, (int8_t *)dst_temp, one_batch_size);
                                }
                            }
                            else if (ALIGN8(shape[1]) * ALIGN2(shape[2]) > 16384) {
                                for (int32_t i = 0; i < batch; i++) {
                                    int8_t *src_temp = (int8_t *)src + i * one_batch_size;
                                    if (srcInPSRAM) {
                                        if (workspace_size < one_batch_size * 2)
                                            return T_ERR_NO_WORKSPACE;
                                        else {
                                            src_temp = (int8_t *)workspace->dptr_ + one_batch_size;
                                            THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src + i * one_batch_size, one_batch_size), "luna_memcpy_i8o8");
                                        }
                                    }
                                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(src_temp, dst_temp, shape[1], shape[2]), "luna_split_mat_trans_i8o8");
                                    opi_psram_cpy_out((int8_t *)dst + i * one_batch_size, (int8_t *)dst_temp, one_batch_size);
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
                            THINKER_RET_CHECK(API_LIB(trans_axis_i32o32)((int32_t *)src, dst_temp, shape, axes, dims), "luna_trans_axis_i32o32");
                            opi_psram_cpy_out((int8_t *)dst, (int8_t *)dst_temp, total_size);
                        }
                        else if (srcInPSRAM && (total_size * 2 <= workspace_size)) {
                            int8_t *src_temp = (int8_t *)workspace->dptr_ + total_size;
                            THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_size), "luna_memcpy_i8o8");
                            THINKER_RET_CHECK(API_LIB(trans_axis_i32o32)((int32_t *)src_temp, dst_temp, shape, axes, dims), "luna_trans_axis_i32o32");
                            opi_psram_cpy_out((int8_t *)dst, (int8_t *)dst_temp, total_size);
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
                            if ((ALIGN2(shape[1]) * ALIGN2(shape[2]) <= 4096) && (workspace_size >= one_batch_size)) {
                                for (int32_t i = 0; i < batch; i++) {
                                    int8_t *src_temp = (int8_t *)src + i * one_batch_size;
                                    if (srcInPSRAM) {
                                        src_temp = (int8_t *)workspace->dptr_;
                                        THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src + i * one_batch_size, one_batch_size), "luna_memcpy_i8o8");
                                    }
                                    THINKER_RET_CHECK(API_LIB(mat_trans_i32o32)((int32_t *)src_temp, dst_temp, shape[1], shape[2]), "luna_mat_trans_i32o32");
                                    opi_psram_cpy_out((int8_t *)dst + i * one_batch_size, (int8_t *)dst_temp, one_batch_size);
                                }
                            }
                            else if (ALIGN2(shape[1]) * ALIGN2(shape[2]) > 4096) {
                                for (int32_t i = 0; i < batch; i++) {
                                    int8_t *src_temp = (int8_t *)src + i * one_batch_size;
                                    if (srcInPSRAM) {
                                        if (workspace_size < one_batch_size * 2)
                                            return T_ERR_NO_WORKSPACE;
                                        else {
                                            src_temp = (int8_t *)workspace->dptr_ + one_batch_size;
                                            THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src + i * one_batch_size, one_batch_size), "luna_memcpy_i8o8");
                                        }
                                    }
                                    THINKER_RET_CHECK(API_LIB(split_mat_trans_i32o32)((int32_t *)src_temp, dst_temp, shape[1], shape[2]), "luna_split_mat_trans_i32o32");
                                    opi_psram_cpy_out((int8_t *)dst + i * one_batch_size, (int8_t *)dst_temp, one_batch_size);
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
                                        THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src + i * one_batch_size, one_batch_size), "luna_memcpy_i8o8");
                                        THINKER_RET_CHECK(API_LIB(trans_axis_i8o8)(src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_i8o8");
                                        opi_psram_cpy_out((int8_t *)dst + i * one_batch_size, (int8_t *)dst_temp, one_batch_size);
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
                                        THINKER_RET_CHECK(API_LIB(trans_axis_i8o8)(src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_i8o8");
                                        opi_psram_cpy_out((int8_t *)dst + i * one_batch_size, (int8_t *)dst_temp, one_batch_size);
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
                                        THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src + i * one_batch_size, one_batch_size), "luna_memcpy_i8o8");
                                        THINKER_RET_CHECK(API_LIB(trans_axis_i32o32)((int32_t *)src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_i32o32");
                                        opi_psram_cpy_out((int8_t *)dst + i * one_batch_size, (int8_t *)dst_temp, one_batch_size);
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
                                        THINKER_RET_CHECK(API_LIB(trans_axis_i32o32)(src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_i32o32");
                                        opi_psram_cpy_out((int8_t *)dst + i * one_batch_size, (int8_t *)dst_temp, one_batch_size);
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

#endif
