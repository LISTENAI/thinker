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

static tStatus mat_transpose_all(void *src, void*dst, int32_t row, int32_t col, uint16_t data_type) {
    if (data_type == Int8)
        return API_LIB(mat_trans_i8o8)((int8_t *)src, (int8_t *)dst, row, col);
    else
        return API_LIB(mat_trans_i32o32)((int32_t *)src, (int32_t *)dst, row, col);
}

static tStatus split_mat_transpose_all(void *src, void*dst, int32_t row, int32_t col, uint16_t data_type) {
    if (data_type == Int8)
        return API_LIB(split_mat_trans_i8o8)((int8_t *)src, (int8_t *)dst, row, col);
    else
        return API_LIB(split_mat_trans_i32o32)((int32_t *)src, (int32_t *)dst, row, col);
}

static tStatus luna_trans_axis_all(void *src, void*dst, uint32_t *shape, uint32_t *axis, uint32_t n_dims,  uint16_t data_type) {
    if (data_type == Int8)
        return API_LIB(trans_axis_i8o8)((int8_t *)src, (int8_t *)dst, shape, axis, n_dims);
    else
        return API_LIB(trans_axis_i32o32)((int32_t *)src, (int32_t *)dst, shape, axis, n_dims);  
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
tStatus transpose_luna(tTensor *X, tTensor *Y, tTensor * workspace, uint32_t dims, uint32_t *axes, uint32_t *shape) {
    tStatus ret = T_ERR_NO_IMPLEMENTED;
    void *src = (void *)X->dptr_;
    void *dst = (void *)Y->dptr_;
    int32_t workspace_size = workspace ? workspace->shape_.dims_[0] : 0;
    int32_t total_size = getShapeSize(&(X->shape_)) * X->byte_;

    bool srcInPSRAM = (X->mem_.type_ != 2);
    bool dstInPSRAM = (Y->mem_.type_ != 2);

    if ((!srcInPSRAM) & (!dstInPSRAM)) {
        switch (dims) {
            case 2: {
                uint32_t row = shape[0];
                uint32_t col = shape[1];
                if (total_size <= 16384)
                    ret = mat_transpose_all(src, dst, row, col, X->dtype_);
                else
                    ret = split_mat_transpose_all(src, dst, row, col, X->dtype_);
                break;
            }
            case 3: {
                ret = luna_trans_axis_all(src, dst, shape, axes, dims, X->dtype_);
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

                    for (int32_t i = 0; i < batch; i++) {
                        void *src_temp = (void *)((int8_t *)src + i * one_batch_size);
                        void *dst_temp = (void *)((int8_t *)dst + i * one_batch_size);
                        ret = luna_trans_axis_all(src_temp, dst_temp, new_shape, new_axis, 3, X->dtype_);
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
                if (total_size <= 16384) {
                    int8_t *src_temp = (int8_t *)dst;
                    ret = luna_memcpy_i8o8(src_temp, src, total_size);
                    ret = mat_transpose_all(src_temp, dst, row, col, X->dtype_);
                }
                else if (total_size <= workspace_size) {
                    int8_t *src_temp = (int8_t *)workspace->dptr_;
                    ret = luna_memcpy_i8o8(src_temp, src, total_size);
                    ret = split_mat_transpose_all(src_temp, dst, row, col, X->dtype_);
                }
                else
                    ret = T_ERR_NO_WORKSPACE;
                break;
            }
            case 3: {
                if (total_size <= workspace_size) {
                    int8_t *src_temp = (int8_t *)workspace->dptr_;
                    ret = luna_memcpy_i8o8(src_temp, src, total_size);
                    ret = luna_trans_axis_all(src, dst, shape, axes, dims, X->dtype_);
                }
                else
                    ret = T_ERR_NO_WORKSPACE;
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
                        for (int32_t i = 0; i < batch; i++) {
                            void *src = (void *)((int8_t *)src + i * one_batch_size);
                            void *dst = (void *)((int8_t *)dst + i * one_batch_size);
                            int8_t *src_temp = (int8_t *)workspace->dptr_;
                            ret = luna_memcpy_i8o8(src_temp, src, one_batch_size);
                            ret = luna_trans_axis_all(src_temp, dst, new_shape, new_axis, 3, X->dtype_);
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
                int8_t *dst_temp = (int8_t *)workspace->dptr_;
                if ((total_size <= 16384) && (workspace_size >= total_size)) {
                    int8_t *src_temp = (int8_t *)src;
                    if (srcInPSRAM) {
                        src_temp = (int8_t *)workspace->dptr_;
                        ret = luna_memcpy_i8o8(src_temp, src, total_size);
                    }
                    ret = mat_transpose_all(src_temp, dst_temp, row, col, X->dtype_);
                    opi_psram_cpy_out(dst, dst_temp, total_size);
                }
                else if (total_size > 16384) {
                    int8_t *src_temp = (int8_t *)src;
                    if (srcInPSRAM) {
                        if (workspace_size < total_size * 2)
                            return T_ERR_NO_WORKSPACE;
                        else {
                            src_temp = (int8_t *)workspace->dptr_ + total_size;
                            ret = luna_memcpy_i8o8(src_temp, src, total_size);
                        }
                    }
                    ret = split_mat_transpose_all(src_temp, dst_temp, row, col, X->dtype_);
                    opi_psram_cpy_out(dst, dst_temp, total_size);
                }
                else
                    ret = T_ERR_NO_WORKSPACE;
                break;
            }
            case 3: {
                int8_t *dst_temp = (int8_t *)workspace->dptr_;
                if ((!srcInPSRAM) && (total_size <= workspace_size)) {
                    ret = luna_trans_axis_all(src, dst_temp, shape, axes, dims, X->dtype_);
                    opi_psram_cpy_out(dst, dst_temp, total_size);
                }
                else if (srcInPSRAM && (total_size * 2 <= workspace_size)) {
                    int8_t *src_temp = (int8_t *)workspace->dptr_ + total_size;
                    ret = luna_memcpy_i8o8(src_temp, src, total_size);
                    ret = luna_trans_axis_all(src_temp, dst_temp, shape, axes, dims, X->dtype_);
                    opi_psram_cpy_out(dst, dst_temp, total_size);
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
                    if ((one_batch_size <= 16384) && (workspace_size >= one_batch_size)) {
                        for (int32_t i = 0; i < batch; i++) {
                            int8_t *src_temp = (int8_t *)(src + i * one_batch_size);
                            if (srcInPSRAM) {
                                src_temp = (int8_t *)workspace->dptr_;
                                ret = luna_memcpy_i8o8(src_temp, src + i * one_batch_size, one_batch_size);
                            }
                            ret = mat_transpose_all(src_temp, dst_temp, shape[1], shape[2], X->dtype_);
                            opi_psram_cpy_out(dst + i * one_batch_size, dst_temp, one_batch_size);
                        }
                    }
                    else if (one_batch_size > 16384) {
                        for (int32_t i = 0; i < batch; i++) {
                            int8_t *src_temp = (int8_t *)(src + i * one_batch_size);
                            if (srcInPSRAM) {
                                if (workspace_size < one_batch_size * 2)
                                    return T_ERR_NO_WORKSPACE;
                                else {
                                    src_temp = (int8_t *)workspace->dptr_ + one_batch_size;
                                    ret = luna_memcpy_i8o8(src_temp, src, one_batch_size);
                                }
                            }
                            ret = split_mat_transpose_all(src_temp, dst_temp, shape[1], shape[2], X->dtype_);
                            opi_psram_cpy_out(dst + i * one_batch_size, dst_temp, one_batch_size);
                        }
                    }
                    else
                        ret = T_ERR_NO_WORKSPACE;
                }
                else
                    ret = T_ERR_NO_WORKSPACE;
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
                    if (srcInPSRAM) {
                        if (one_batch_size * 2 <= workspace_size) {
                            for (int32_t i = 0; i < batch; i++) {
                                int8_t *src_temp = (int8_t *)workspace->dptr_ + one_batch_size;
                                int8_t *dst_temp = (int8_t *)workspace->dptr_;
                                ret = luna_memcpy_i8o8(src_temp, src + i * one_batch_size, one_batch_size);
                                ret = luna_trans_axis_all(src_temp, dst_temp, new_shape, new_axis, 3, X->dtype_);
                                opi_psram_cpy_out(dst + i * one_batch_size, dst_temp, total_size);
                            }
                        }
                        else
                            ret = T_ERR_NO_WORKSPACE;
                    }
                    else {
                        if (one_batch_size <= workspace_size) {
                            for (int32_t i = 0; i < batch; i++) {
                                int8_t *src_temp = (int8_t *)src + one_batch_size;
                                int8_t *dst_temp = (int8_t *)workspace->dptr_;
                                ret = luna_trans_axis_all(src_temp, dst_temp, new_shape, new_axis, 3, X->dtype_);
                                opi_psram_cpy_out(dst + i * one_batch_size, dst_temp, total_size);
                            }
                        }
                        else 
                            ret = T_ERR_NO_WORKSPACE;
                    }
                }
                else {
                    return T_ERR_NO_IMPLEMENTED;
                }
                break;
            }
        }
    }
    return ret;
}

#endif