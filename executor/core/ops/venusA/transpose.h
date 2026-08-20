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


static int32_t transpose_luna_plain_ok(int16_t dtype, uint32_t row, uint32_t col) {
    switch (dtype) {
        case Int8:
            return ALIGN4(row) * ALIGN8(col) <= 65536;
        case Int16:
            return ALIGN4(row) * ALIGN4(col) <= 65536;
        case Int32:
        case Float32:
            return ALIGN2(row) * ALIGN4(col) <= 32768;
        default:
            return 0;
    }
}

static int32_t transpose_luna_axis_perm_ok(const uint32_t *axes) {
    return ((axes[0] == 0 && axes[1] == 2 && axes[2] == 1) ||
            (axes[0] == 2 && axes[1] == 0 && axes[2] == 1) ||
            (axes[0] == 2 && axes[1] == 1 && axes[2] == 0) ||
            (axes[0] == 1 && axes[1] == 2 && axes[2] == 0) ||
            (axes[0] == 1 && axes[1] == 0 && axes[2] == 2));
}

static int32_t transpose_luna_axis_ok(int16_t dtype, const uint32_t *shape, const uint32_t *axes) {
    if (!transpose_luna_axis_perm_ok(axes)) {
        return 0;
    }
    if ((axes[0] == 0 && axes[1] == 2 && axes[2] == 1) ||
        (axes[0] == 2 && axes[1] == 0 && axes[2] == 1) ||
        (axes[0] == 1 && axes[1] == 0 && axes[2] == 2)) {
        return transpose_luna_plain_ok(dtype, shape[1], shape[2]);
    }
    return transpose_luna_plain_ok(dtype, shape[0], shape[2]);
}

static int32_t transpose_luna_check(int16_t dtype, uint32_t dims, uint32_t *axes, uint32_t *shape) {
    if (dims == 2) {
        if (axes[0] != 1 || axes[1] != 0) {
            return T_ERR_INVALID_PARA;
        }
    } else if (dims == 3) {
        if (!transpose_luna_axis_ok(dtype, shape, axes)) {
            return T_ERR_INVALID_PARA;
        }
    } else if (dims == 4) {
        if (axes[0] != 0) {
            return T_ERR_NO_IMPLEMENTED;
        }
        uint32_t new_axis[3];
        uint32_t new_shape[3];
        for (int32_t n = 0; n < 3; n++) {
            if (axes[n + 1] == 0) {
                return T_ERR_INVALID_PARA;
            }
            new_axis[n] = axes[n + 1] - 1;
            new_shape[n] = shape[n + 1];
        }
        if (!transpose_luna_axis_ok(dtype, new_shape, new_axis)) {
            return T_ERR_INVALID_PARA;
        }
    } else {
        return T_ERR_NO_IMPLEMENTED;
    }
    return T_SUCCESS;
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
    void *src = (void *)X->dptr_;
    void *dst = (void *)Y->dptr_;
    size_t workspace_size = workspace ? getTensorDataSize(workspace) : 0;
    size_t total_bytes = getTensorDataSize(X);
    int16_t dtype = X->dtype_;
    if (dtype == Float32) {
        dtype = Int32;
    }

    int32_t check_ret = T_SUCCESS;
    #if THINKER_PARAM_CHECK
    if ((check_ret = transpose_luna_check(dtype, dims, axes, shape)) != T_SUCCESS) {
        return (check_ret);
    }
#endif

    bool srcInPSRAM = (X->mem_.type_ != 2);
    bool dstInPSRAM = (Y->mem_.type_ != 2);
    if (X->dptr_ == Y->dptr_ &&
        (dims >= 3 || (dims == 2 && !transpose_luna_plain_ok(dtype, shape[0], shape[1]))))
        return T_ERR_INVALID_PARA;

    if ((!srcInPSRAM) & (!dstInPSRAM)) {
        switch (dims) {
            case 2: {
                uint32_t row = shape[0];
                uint32_t col = shape[1];
                if (row == 1 || col == 1) {
                    THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)dst, (int8_t *)src, total_bytes),
                                    "luna_memcpy_i8o8");
                    return T_SUCCESS;
                }
                switch (dtype) {
                    case Int8:
                        if (transpose_luna_plain_ok(dtype, row, col))
                            THINKER_RET_CHECK(luna_mat_trans_i8o8((int8_t *)src, (int8_t *)dst, row, col), "luna_mat_trans_i8o8");
                        else
                            THINKER_RET_CHECK(luna_split_mat_trans_i8o8((int8_t *)src, (int8_t *)dst, row, col), "luna_split_mat_trans_i8o8");
                        break;
                    case Int16:
                        if (transpose_luna_plain_ok(dtype, row, col))
                            THINKER_RET_CHECK(luna_mat_trans_i16o16((int16_t *)src, (int16_t *)dst, row, col), "luna_mat_trans_i16o16");
                        else
                            THINKER_RET_CHECK(luna_split_mat_trans_i16o16((int16_t *)src, (int16_t *)dst, row, col), "luna_split_mat_trans_i16o16");
                        break;
                    case Int32:
                    case Float32:
                        if (transpose_luna_plain_ok(dtype, row, col))
                            THINKER_RET_CHECK(luna_mat_trans_i32o32((int32_t *)src, (int32_t *)dst, row, col), "luna_mat_trans_i32o32");
                        else
                            THINKER_RET_CHECK(luna_split_mat_trans_i32o32((int32_t *)src, (int32_t *)dst, row, col), "luna_split_mat_trans_i32o32");
                        break;
                    default:
                        return T_ERR_INVALID_DATATYPE;
                }
                break;
            }
            case 3: {
                switch (dtype) {
                    case Int8:
                        THINKER_RET_CHECK(luna_trans_axis_i8o8((int8_t *)src, (int8_t *)dst, shape, axes, dims), "luna_trans_axis_i8o8");
                        break;
                    case Int16:
                        THINKER_RET_CHECK(luna_trans_axis_i16o16((int16_t *)src, (int16_t *)dst, shape, axes, dims), "luna_trans_axis_i16o16");
                        break;
                    case Int32:
                    case Float32:
                        THINKER_RET_CHECK(luna_trans_axis_i32o32((int32_t *)src, (int32_t *)dst, shape, axes, dims), "luna_trans_axis_i32o32");
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
                                int8_t *p_src = (int8_t *)src + i * one_batch_size;
                                int8_t *p_dst = (int8_t *)dst + i * one_batch_size;
                                THINKER_RET_CHECK(luna_trans_axis_i8o8(p_src, p_dst, new_shape, new_axis, 3), "luna_trans_axis_i8o8");
                            }
                            break;
                        case Int16:
                            for (int32_t i = 0; i < batch; i++) {
                                int16_t *p_src = (int16_t *)((int8_t *)src + i * one_batch_size);
                                int16_t *p_dst = (int16_t *)((int8_t *)dst + i * one_batch_size);
                                THINKER_RET_CHECK(luna_trans_axis_i16o16(p_src, p_dst, new_shape, new_axis, 3), "luna_trans_axis_i16o16");
                            }
                            break;
                        case Int32:
                        case Float32:
                            for (int32_t i = 0; i < batch; i++) {
                                int32_t *p_src = (int32_t *)((int8_t *)src + i * one_batch_size);
                                int32_t *p_dst = (int32_t *)((int8_t *)dst + i * one_batch_size);
                                THINKER_RET_CHECK(luna_trans_axis_i32o32(p_src, p_dst, new_shape, new_axis, 3), "luna_trans_axis_i32o32");
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
                        if (transpose_luna_plain_ok(dtype, row, col)) {
                            int8_t *src_temp = (int8_t *)dst;
                            THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_bytes), "luna_memcpy_i8o8");
                            THINKER_RET_CHECK(luna_mat_trans_i8o8(src_temp, (int8_t *)dst, row, col), "luna_mat_trans_i8o8");
                        }
                        else if (total_bytes <= workspace_size) {
                            int8_t *src_temp = (int8_t *)workspace->dptr_;
                            THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_bytes), "luna_memcpy_i8o8");
                            THINKER_RET_CHECK(luna_split_mat_trans_i8o8(src_temp, (int8_t *)dst, row, col), "luna_split_mat_trans_i8o8");
                        }
                        else
                            return T_ERR_NO_WORKSPACE;
                        break;
                    case Int16:
                        if (transpose_luna_plain_ok(dtype, row, col)) {
                            int8_t *src_temp = (int8_t *)dst;
                            THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_bytes), "luna_memcpy_i8o8");
                            THINKER_RET_CHECK(luna_mat_trans_i16o16((int16_t *)src_temp, (int16_t *)dst, row, col), "luna_mat_trans_i16o16");
                        }
                        else if (total_bytes <= workspace_size) {
                            int8_t *src_temp = (int8_t *)workspace->dptr_;
                            THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_bytes), "luna_memcpy_i8o8");
                            THINKER_RET_CHECK(luna_split_mat_trans_i16o16((int16_t *)src_temp, (int16_t *)dst, row, col), "luna_split_mat_trans_i16o16");
                        }
                        else
                            return T_ERR_NO_WORKSPACE;
                        break;
                    case Int32:
                        if (transpose_luna_plain_ok(dtype, row, col)) {
                            int8_t *src_temp = (int8_t *)dst;
                            THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_bytes), "luna_memcpy_i8o8");
                            THINKER_RET_CHECK(luna_mat_trans_i32o32((int32_t *)src_temp, (int32_t *)dst, row, col), "luna_mat_trans_i32o32");
                        }
                        else if (total_bytes <= workspace_size) {
                            int8_t *src_temp = (int8_t *)workspace->dptr_;
                            THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_bytes), "luna_memcpy_i8o8");
                            THINKER_RET_CHECK(luna_split_mat_trans_i32o32((int32_t *)src_temp, (int32_t *)dst, row, col), "luna_split_mat_trans_i32o32");
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
                if (total_bytes <= workspace_size) {
                    switch (dtype) {
                        case Int8: {
                            int8_t *src_temp = (int8_t *)workspace->dptr_;
                            THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_bytes), "luna_memcpy_i8o8");
                            THINKER_RET_CHECK(luna_trans_axis_i8o8(src_temp, (int8_t *)dst, shape, axes, dims), "luna_trans_axis_i8o8");
                            break;
                        }
                        case Int16: {
                            int8_t *src_temp = (int8_t *)workspace->dptr_;
                            THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_bytes), "luna_memcpy_i8o8");
                            THINKER_RET_CHECK(luna_trans_axis_i16o16((int16_t *)src_temp, (int16_t *)dst, shape, axes, dims), "luna_trans_axis_i16o16");
                            break;
                        }
                        case Int32: {
                            int8_t *src_temp = (int8_t *)workspace->dptr_;
                            THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_bytes), "luna_memcpy_i8o8");
                            THINKER_RET_CHECK(luna_trans_axis_i32o32((int32_t *)src_temp, (int32_t *)dst, shape, axes, dims), "luna_trans_axis_i32o32");
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
                                    int8_t *p_src = (int8_t *)src + i * one_batch_size;
                                    int8_t *p_dst = (int8_t *)dst + i * one_batch_size;
                                    int8_t *src_temp = (int8_t *)workspace->dptr_;
                                    THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)p_src, one_batch_size), "luna_memcpy_i8o8");
                                    THINKER_RET_CHECK(luna_trans_axis_i8o8(src_temp, (int8_t *)p_dst, new_shape, new_axis, 3), "luna_trans_axis_i8o8");
                                }
                                break;
                            case Int16:
                                for (int32_t i = 0; i < batch; i++) {
                                    int16_t *p_src = (int16_t *)((int8_t *)src + i * one_batch_size);
                                    int16_t *p_dst = (int16_t *)((int8_t *)dst + i * one_batch_size);
                                    int8_t *src_temp = (int8_t *)workspace->dptr_;
                                    THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)p_src, one_batch_size), "luna_memcpy_i8o8");
                                    THINKER_RET_CHECK(luna_trans_axis_i16o16((int16_t *)src_temp, (int16_t *)p_dst, new_shape, new_axis, 3), "luna_trans_axis_i16o16");
                                }
                                break;
                            case Int32:
                                for (int32_t i = 0; i < batch; i++) {
                                    int32_t *p_src = (int32_t *)((int8_t *)src + i * one_batch_size);
                                    int32_t *p_dst = (int32_t *)((int8_t *)dst + i * one_batch_size);
                                    int8_t *src_temp = (int8_t *)workspace->dptr_;
                                    THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)p_src, one_batch_size), "luna_memcpy_i8o8");
                                    THINKER_RET_CHECK(luna_trans_axis_i32o32((int32_t *)src_temp, (int32_t *)p_dst, new_shape, new_axis, 3), "luna_trans_axis_i32o32");
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
                        if ((transpose_luna_plain_ok(dtype, row, col)) && (workspace_size >= total_bytes)) {
                            int8_t *src_temp = (int8_t *)src;
                            if (srcInPSRAM) {
                                src_temp = (int8_t *)workspace->dptr_;
                                THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_bytes), "luna_memcpy_i8o8");
                            }
                            THINKER_RET_CHECK(luna_mat_trans_i8o8(src_temp, dst_temp, row, col), "luna_mat_trans_i8o8");
                            opi_psram_cpy_out((void *)dst, (void *)dst_temp, total_bytes);
                        }
                        else if (!transpose_luna_plain_ok(dtype, row, col)) {
                            int8_t *src_temp = (int8_t *)src;
                            if (srcInPSRAM) {
                                if (workspace_size < total_bytes * 2)
                                    return T_ERR_NO_WORKSPACE;
                                else {
                                    src_temp = (int8_t *)workspace->dptr_ + total_bytes;
                                    THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_bytes), "luna_memcpy_i8o8");
                                }
                            }
                            THINKER_RET_CHECK(luna_split_mat_trans_i8o8(src_temp, dst_temp, row, col), "luna_split_mat_trans_i8o8");
                            opi_psram_cpy_out((void *)dst, (void *)dst_temp, total_bytes);
                        }
                        else
                            return T_ERR_NO_WORKSPACE;
                        break;
                    }
                    case Int16: {
                        int16_t *dst_temp = (int16_t *)workspace->dptr_;
                        if ((transpose_luna_plain_ok(dtype, row, col)) && (workspace_size >= total_bytes)) {
                            int8_t *src_temp = (int8_t *)src;
                            if (srcInPSRAM) {
                                src_temp = (int8_t *)workspace->dptr_;
                                THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_bytes), "luna_memcpy_i8o8");
                            }
                            THINKER_RET_CHECK(luna_mat_trans_i16o16((int16_t *)src_temp, dst_temp, row, col), "luna_mat_trans_i16o16");
                            opi_psram_cpy_out((void *)dst, (void *)dst_temp, total_bytes);
                        }
                        else if (!transpose_luna_plain_ok(dtype, row, col)) {
                            int8_t *src_temp = (int8_t *)src;
                            if (srcInPSRAM) {
                                if (workspace_size < total_bytes * 2)
                                    return T_ERR_NO_WORKSPACE;
                                else {
                                    src_temp = (int8_t *)workspace->dptr_ + total_bytes;
                                    THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_bytes), "luna_memcpy_i8o8");
                                }
                            }
                            THINKER_RET_CHECK(luna_split_mat_trans_i16o16((int16_t *)src_temp, dst_temp, row, col), "luna_split_mat_trans_i16o16");
                            opi_psram_cpy_out((void *)dst, (void *)dst_temp, total_bytes);
                        }
                        else
                            return T_ERR_NO_WORKSPACE;
                        break;
                    }
                    case Int32: {
                        int32_t *dst_temp = (int32_t *)workspace->dptr_;
                        if ((transpose_luna_plain_ok(dtype, row, col)) && (workspace_size >= total_bytes)) {
                            int8_t *src_temp = (int8_t *)src;
                            if (srcInPSRAM) {
                                src_temp = (int8_t *)workspace->dptr_;
                                THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_bytes), "luna_memcpy_i8o8");
                            }
                            THINKER_RET_CHECK(luna_mat_trans_i32o32((int32_t *)src_temp, dst_temp, row, col), "luna_mat_trans_i32o32");
                            opi_psram_cpy_out((void *)dst, (void *)dst_temp, total_bytes);
                        }
                        else if (!transpose_luna_plain_ok(dtype, row, col)) {
                            int8_t *src_temp = (int8_t *)src;
                            if (srcInPSRAM) {
                                if (workspace_size < total_bytes * 2)
                                    return T_ERR_NO_WORKSPACE;
                                else {
                                    src_temp = (int8_t *)workspace->dptr_ + total_bytes;
                                    THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_bytes), "luna_memcpy_i8o8");
                                }
                            }
                            THINKER_RET_CHECK(luna_split_mat_trans_i32o32((int32_t *)src_temp, dst_temp, row, col), "luna_split_mat_trans_i32o32");
                            opi_psram_cpy_out((void *)dst, (void *)dst_temp, total_bytes);
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
                        if ((!srcInPSRAM) && (total_bytes <= workspace_size)) {
                            THINKER_RET_CHECK(luna_trans_axis_i8o8((int8_t *)src, dst_temp, shape, axes, dims), "luna_trans_axis_i8o8");
                            opi_psram_cpy_out(dst, dst_temp, total_bytes);
                        }
                        else if (srcInPSRAM && (total_bytes * 2 <= workspace_size)) {
                            int8_t *src_temp = (int8_t *)workspace->dptr_ + total_bytes;
                            THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_bytes), "luna_memcpy_i8o8");
                            THINKER_RET_CHECK(luna_trans_axis_i8o8(src_temp, dst_temp, shape, axes, dims), "luna_trans_axis_i8o8");
                            opi_psram_cpy_out((void *)dst, (void *)dst_temp, total_bytes);
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
                            if ((transpose_luna_plain_ok(dtype, shape[1], shape[2])) && (workspace_size >= one_batch_size)) {
                                for (int32_t i = 0; i < batch; i++) {
                                    int8_t *src_temp = (int8_t *)src + i * one_batch_size;
                                    if (srcInPSRAM) {
                                        src_temp = (int8_t *)workspace->dptr_;
                                        THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src + i * one_batch_size, one_batch_size), "luna_memcpy_i8o8");
                                    }
                                    THINKER_RET_CHECK(luna_mat_trans_i8o8(src_temp, dst_temp, shape[1], shape[2]), "luna_mat_trans_i8o8");
                                    opi_psram_cpy_out(dst + i * one_batch_size, dst_temp, one_batch_size);
                                }
                            }
                            else if (!transpose_luna_plain_ok(dtype, shape[1], shape[2])) {
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
                                    THINKER_RET_CHECK(luna_split_mat_trans_i8o8(src_temp, dst_temp, shape[1], shape[2]), "luna_split_mat_trans_i8o8");
                                    opi_psram_cpy_out(dst + i * one_batch_size, dst_temp, one_batch_size);
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
                        if ((!srcInPSRAM) && (total_bytes <= workspace_size)) {
                            THINKER_RET_CHECK(luna_trans_axis_i16o16((int16_t *)src, dst_temp, shape, axes, dims), "luna_trans_axis_i16o16");
                            opi_psram_cpy_out(dst, dst_temp, total_bytes);
                        }
                        else if (srcInPSRAM && (total_bytes * 2 <= workspace_size)) {
                            int8_t *src_temp = (int8_t *)workspace->dptr_ + total_bytes;
                            THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_bytes), "luna_memcpy_i8o8");
                            THINKER_RET_CHECK(luna_trans_axis_i16o16((int16_t *)src_temp, dst_temp, shape, axes, dims), "luna_trans_axis_i16o16");
                            opi_psram_cpy_out((void *)dst, (void *)dst_temp, total_bytes);
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
                            if ((transpose_luna_plain_ok(dtype, shape[1], shape[2])) && (workspace_size >= one_batch_size)) {
                                for (int32_t i = 0; i < batch; i++) {
                                    int8_t *src_temp = (int8_t *)src + i * one_batch_size;
                                    if (srcInPSRAM) {
                                        src_temp = (int8_t *)workspace->dptr_;
                                        THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src + i * one_batch_size, one_batch_size), "luna_memcpy_i8o8");
                                    }
                                    THINKER_RET_CHECK(luna_mat_trans_i16o16((int16_t *)src_temp, dst_temp, shape[1], shape[2]), "luna_mat_trans_i16o16");
                                    opi_psram_cpy_out(dst + i * one_batch_size, dst_temp, one_batch_size);
                                }
                            }
                            else if (!transpose_luna_plain_ok(dtype, shape[1], shape[2])) {
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
                                    THINKER_RET_CHECK(luna_split_mat_trans_i16o16((int16_t *)src_temp, dst_temp, shape[1], shape[2]), "luna_split_mat_trans_i16o16");
                                    opi_psram_cpy_out(dst + i * one_batch_size, dst_temp, one_batch_size);
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
                        if ((!srcInPSRAM) && (total_bytes <= workspace_size)) {
                            THINKER_RET_CHECK(luna_trans_axis_i32o32((int32_t *)src, dst_temp, shape, axes, dims), "luna_trans_axis_i32o32");
                            opi_psram_cpy_out(dst, dst_temp, total_bytes);
                        }
                        else if (srcInPSRAM && (total_bytes * 2 <= workspace_size)) {
                            int8_t *src_temp = (int8_t *)workspace->dptr_ + total_bytes;
                            THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src, total_bytes), "luna_memcpy_i8o8");
                            THINKER_RET_CHECK(luna_trans_axis_i32o32((int32_t *)src_temp, dst_temp, shape, axes, dims), "luna_trans_axis_i32o32");
                            opi_psram_cpy_out((void *)dst, (void *)dst_temp, total_bytes);
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
                            if ((transpose_luna_plain_ok(dtype, shape[1], shape[2])) && (workspace_size >= one_batch_size)) {
                                for (int32_t i = 0; i < batch; i++) {
                                    int8_t *src_temp = (int8_t *)src + i * one_batch_size;
                                    if (srcInPSRAM) {
                                        src_temp = (int8_t *)workspace->dptr_;
                                        THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src + i * one_batch_size, one_batch_size), "luna_memcpy_i8o8");
                                    }
                                    THINKER_RET_CHECK(luna_mat_trans_i32o32((int32_t *)src_temp, dst_temp, shape[1], shape[2]), "luna_mat_trans_i32o32");
                                    opi_psram_cpy_out(dst + i * one_batch_size, dst_temp, one_batch_size);
                                }
                            }
                            else if (!transpose_luna_plain_ok(dtype, shape[1], shape[2])) {
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
                                    THINKER_RET_CHECK(luna_split_mat_trans_i32o32((int32_t *)src_temp, dst_temp, shape[1], shape[2]), "luna_split_mat_trans_i32o32");
                                    opi_psram_cpy_out(dst + i * one_batch_size, dst_temp, one_batch_size);
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
                                        THINKER_RET_CHECK(luna_trans_axis_i8o8(src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_i8o8");
                                        opi_psram_cpy_out(dst + i * one_batch_size, dst_temp, one_batch_size);
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
                                        THINKER_RET_CHECK(luna_trans_axis_i8o8(src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_i8o8");
                                        opi_psram_cpy_out(dst + i * one_batch_size, dst_temp, one_batch_size);
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
                                        THINKER_RET_CHECK(luna_memcpy_i8o8(src_temp, (int8_t *)src + i * one_batch_size, one_batch_size), "luna_memcpy_i8o8");
                                        THINKER_RET_CHECK(luna_trans_axis_i16o16((int16_t *)src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_i16o16");
                                        opi_psram_cpy_out(dst + i * one_batch_size, dst_temp, one_batch_size);
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
                                        THINKER_RET_CHECK(luna_trans_axis_i16o16(src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_i16o16");
                                        opi_psram_cpy_out(dst + i * one_batch_size, dst_temp, one_batch_size);
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
                                        THINKER_RET_CHECK(luna_trans_axis_i32o32((int32_t *)src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_i32o32");
                                        opi_psram_cpy_out(dst + i * one_batch_size, dst_temp, one_batch_size);
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
                                        THINKER_RET_CHECK(luna_trans_axis_i32o32(src_temp, dst_temp, new_shape, new_axis, 3), "luna_trans_axis_i32o32");
                                        opi_psram_cpy_out(dst + i * one_batch_size, dst_temp, one_batch_size);
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
    if (X->mem_.type_ != 2) {
        switch (X->dtype_) {
            case Int8: {
                const int8_t *src = (int8_t *)X->dptr_;
                int8_t *dst = (int8_t *)Y->dptr_;
                for (int32_t i = 0; i < split_num - 1; i++) {
                    THINKER_RET_CHECK(luna_mat_trans_inv_i8o8(src, dst, row, split_num, row, col), "luna_mat_trans_inv_i8o8");
                }
                THINKER_RET_CHECK(luna_mat_trans_inv_i8o8(src, dst, row, col - (split_num - 1) * split_num, row, col), "luna_mat_trans_inv_i8o8");
                break;
            }
            case Int16: {
                const int16_t *src = (int16_t *)X->dptr_;
                int16_t *dst = (int16_t *)Y->dptr_;
                for (int32_t i = 0; i < split_num - 1; i++) {
                    THINKER_RET_CHECK(luna_mat_trans_inv_i16o16(src, dst, row, split_num, row, col), "luna_mat_trans_inv_i16o16");
                }
                THINKER_RET_CHECK(luna_mat_trans_inv_i16o16(src, dst, row, col - (split_num - 1) * split_num, row, col), "luna_mat_trans_inv_i16o16");
                break;
            }
            case Int32: {
                const int32_t *src = (int32_t *)X->dptr_;
                int32_t *dst = (int32_t *)Y->dptr_;
                for (int32_t i = 0; i < split_num - 1; i++) {
                    THINKER_RET_CHECK(luna_mat_trans_inv_i32o32(src, dst, row, split_num, row, col), "luna_mat_trans_inv_i32o32");
                }
                THINKER_RET_CHECK(luna_mat_trans_inv_i32o32(src, dst, row, col - (split_num - 1) * split_num, row, col), "luna_mat_trans_inv_i32o32");
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
if (n_dims != 3) {
    return (T_ERR_INVALID_PARA);
}
#endif
    
    switch (dtype) {
        case Int8:
            THINKER_RET_CHECK(API_LIB(trans_axis_i8o8)((int8_t *)src, (int8_t *)dst, (uint32_t *)in_shape, (uint32_t *)axis, n_dims), "luna_trans_axis_i8o8");
            break;
        case Int16:
            THINKER_RET_CHECK(API_LIB(trans_axis_i16o16)((int16_t *)src, (int16_t *)dst, (uint32_t *)in_shape, (uint32_t *)axis, n_dims), "luna_trans_axis_i16o16");
            break;
        case Int32:
            THINKER_RET_CHECK(API_LIB(trans_axis_i32o32)((int32_t *)src, (int32_t *)dst, (uint32_t *)in_shape, (uint32_t *)axis, n_dims), "luna_trans_axis_i32o32");
            break;
        default:
            return T_ERR_INVALID_DATATYPE;
    }
    
    return T_SUCCESS;
}

#endif
