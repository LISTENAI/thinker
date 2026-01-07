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
 * @brief Calculate the ceiling of x divided by 2^shift
 * @param x Input value
 * @param shift Number of bits to shift
 * @return int32_t Ceiling value
 */
static int32_t luna_ceil(int32_t x, int32_t shift) {
    if (x & ~(0xFFFFFFFF << shift)) {
        return (x >> shift) + 1;
    } else {
        return (x >> shift);
    }
}

/**
 * @brief Check if the matrix dimensions are within the allowed size limits
 * @param row Number of rows
 * @param col Number of columns
 * @param dtype Data type (e.g., Int8, Int16, Int32)
 * @return int32_t 0 if size is valid, 1 otherwise
 */
static int32_t check_trans_size_legal(int32_t row, int32_t col, int32_t dtype) {
    const int32_t max_size = 64 * 1024;
    int32_t d_byte = (dtype & 0xF);
    int32_t row_bit = 4 - (d_byte >> 1);
    int32_t col_bit = 2;
    int32_t size = (luna_ceil(row, row_bit) << row_bit) * (luna_ceil(col, col_bit) << col_bit) * d_byte;
    return size > max_size ? 1 : 0;
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
    tStatus ret = T_ERR_NO_IMPLEMENTED;
    void *src = (void *)X->dptr_;
    void *dst = (void *)Y->dptr_;
    int32_t workspace_size = workspace ? workspace->shape_.dims_[0] : 0;
    int32_t total_size = getShapeSize(&(X->shape_));

    bool srcInPSRAM = (X->mem_.type_ != 2);
    bool dstInPSRAM = (Y->mem_.type_ != 2);

    if ((!srcInPSRAM) & (!dstInPSRAM)) {
        switch (dims) {
            case 2: {
                uint32_t row = shape[0];
                uint32_t col = shape[1];
                if (total_size <= 65536)
                    ret = luna_mat_trans_q7(src, dst, row, col);
                else {
                    int32_t split_num = 2;
                    int32_t split_row = row / split_num;
                    while (check_trans_size_legal(split_row, col, Int8)) {
                        split_num++;
                        split_row = row / split_num;
                    }
                    if (row % split_num != 0) {
                        return T_ERR_NO_IMPLEMENTED;
                    }

                    ret = luna_split_mat_trans_q7(src, dst, row, col, split_num);
                }
                break;
            }
            case 3: {
                ret = luna_trans_axis_q7(src, dst, shape, axes, dims);
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
                        ret = luna_trans_axis_q7(src_temp, dst_temp, new_shape, new_axis, 3);
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
                if (total_size <= 65536) {
                    int8_t *src_temp = (int8_t *)dst;
                    memcpy(src_temp, src, total_size);
                    ret = luna_mat_trans_q7(src_temp, dst, row, col);
                }
                else if (total_size <= workspace_size) {
                    int8_t *src_temp = (int8_t *)workspace->dptr_;
                    memcpy(src_temp, src, total_size);

                    int32_t split_num = 2;
                    int32_t split_row = row / split_num;
                    while (check_trans_size_legal(split_row, col, Int8)) {
                        split_num++;
                        split_row = row / split_num;
                    }
                    if (row % split_num != 0) {
                        return T_ERR_NO_IMPLEMENTED;
                    }

                    ret = luna_split_mat_trans_q7(src_temp, dst, row, col, split_num);
                }
                else
                    ret = T_ERR_NO_WORKSPACE;
                break;
            }
            case 3: {
                if (total_size <= workspace_size) {
                    int8_t *src_temp = (int8_t *)workspace->dptr_;
                    memcpy(src_temp, src, total_size);
                    ret = luna_trans_axis_q7(src, dst, shape, axes, dims);
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
                            void *dst_temp = (void *)((int8_t *)dst + i * one_batch_size);
                            int8_t *src_temp = (int8_t *)workspace->dptr_;
                            memcpy(src_temp, src + i * one_batch_size, one_batch_size);
                            ret = luna_trans_axis_q7(src_temp, dst_temp, new_shape, new_axis, 3);
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
                if ((total_size <= 65536) && (workspace_size >= total_size)) {
                    int8_t *src_temp = (int8_t *)src;
                    if (srcInPSRAM) {
                        src_temp = (int8_t *)workspace->dptr_;
                        memcpy(src_temp, src, total_size);
                    }
                    ret = luna_mat_trans_q7(src_temp, dst_temp, row, col);
                    memcpy(dst, dst_temp, total_size);
                }
                else if (total_size > 65536) {
                    int8_t *src_temp = (int8_t *)src;
                    if (srcInPSRAM) {
                        if (workspace_size < total_size * 2)
                            return T_ERR_NO_WORKSPACE;
                        else {
                            src_temp = (int8_t *)workspace->dptr_ + total_size;
                            memcpy(src_temp, src, total_size);
                        }
                    }

                    int32_t split_num = 2;
                    int32_t split_row = row / split_num;
                    while (check_trans_size_legal(split_row, col, Int8)) {
                        split_num++;
                        split_row = row / split_num;
                    }
                    if (row % split_num != 0) {
                        return T_ERR_NO_IMPLEMENTED;
                    }

                    ret = luna_split_mat_trans_q7(src_temp, dst_temp, row, col, split_num);
                    memcpy(dst, dst_temp, total_size);
                }
                else
                    ret = T_ERR_NO_WORKSPACE;
                break;
            }
            case 3: {
                int8_t *dst_temp = (int8_t *)workspace->dptr_;
                if ((!srcInPSRAM) && (total_size <= workspace_size)) {
                    ret = luna_trans_axis_q7(src, dst_temp, shape, axes, dims);
                    memcpy(dst, dst_temp, total_size);
                }
                else if (srcInPSRAM && (total_size * 2 <= workspace_size)) {
                    int8_t *src_temp = (int8_t *)workspace->dptr_ + total_size;
                    memcpy(src_temp, src, total_size);
                    ret = luna_trans_axis_q7(src_temp, dst_temp, shape, axes, dims);
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
                    if ((one_batch_size <= 65536) && (workspace_size >= one_batch_size)) {
                        for (int32_t i = 0; i < batch; i++) {
                            int8_t *src_temp = (int8_t *)(src + i * one_batch_size);
                            if (srcInPSRAM) {
                                src_temp = (int8_t *)workspace->dptr_;
                                memcpy(src_temp, src + i * one_batch_size, one_batch_size);
                            }
                            ret = luna_mat_trans_q7(src_temp, dst_temp, shape[1], shape[2]);
                            memcpy(dst + i * one_batch_size, dst_temp, one_batch_size);
                        }
                    }
                    else if (one_batch_size > 65536) {
                        for (int32_t i = 0; i < batch; i++) {
                            int8_t *src_temp = (int8_t *)(src + i * one_batch_size);
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
                            int32_t split_num = 2;
                            int32_t split_row = row / split_num;
                            while (check_trans_size_legal(split_row, col, Int8)) {
                                split_num++;
                                split_row = row / split_num;
                            }
                            if (row % split_num != 0) {
                                return T_ERR_NO_IMPLEMENTED;
                            }

                            ret = luna_split_mat_trans_q7(src_temp, dst_temp, shape[1], shape[2], split_num);
                            memcpy(dst + i * one_batch_size, dst_temp, one_batch_size);
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
                                memcpy(src_temp, src + i * one_batch_size, one_batch_size);
                                ret = luna_trans_axis_q7(src_temp, dst_temp, new_shape, new_axis, 3);
                                memcpy(dst + i * one_batch_size, dst_temp, total_size);
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
                                ret = luna_trans_axis_q7(src_temp, dst_temp, new_shape, new_axis, 3);
                                memcpy(dst + i * one_batch_size, dst_temp, total_size);
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
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
    if (X->mem_.type_ != 2) {
        switch (X->dtype_) {
            case Int8: {
                const int8_t *src = (int8_t *)X->dptr_;
                int8_t *dst = (int8_t *)Y->dptr_;
                for (int32_t i = 0; i < split_num - 1; i++) {
                    ret = luna_mat_trans_inv_q7(src, dst, row, split_num, row, col);
                }
                ret = luna_mat_trans_inv_q7(src, dst, row, col - (split_num - 1) * split_num, row, col);
                break;
            }
            case Int16: {
                const int16_t *src = (int16_t *)X->dptr_;
                int16_t *dst = (int16_t *)Y->dptr_;
                for (int32_t i = 0; i < split_num - 1; i++) {
                    ret = luna_mat_trans_inv_q15(src, dst, row, split_num, row, col);
                }
                ret = luna_mat_trans_inv_q15(src, dst, row, col - (split_num - 1) * split_num, row, col);
                break;
            }
            case Int32: {
                const int32_t *src = (int32_t *)X->dptr_;
                int32_t *dst = (int32_t *)Y->dptr_;
                for (int32_t i = 0; i < split_num - 1; i++) {
                    ret = luna_mat_trans_inv_q31(src, dst, row, split_num, row, col);
                }
                ret = luna_mat_trans_inv_q31(src, dst, row, col - (split_num - 1) * split_num, row, col);
                break;
            }
            default:
                return T_ERR_NO_IMPLEMENTED;
        }
    }
    
    return ret;
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
    int32_t ret = T_ERR_FAIL;
    
    if (n_dims != 3) {
        return ret;
    }
    
    switch (dtype) {
        case Int8:
            ret = API_LIB(trans_axis_q7)((int8_t *)src, (int8_t *)dst, (uint32_t *)in_shape, (uint32_t *)axis, n_dims);
            break;
        case Int16:
            ret = API_LIB(trans_axis_q15)((int16_t *)src, (int16_t *)dst, (uint32_t *)in_shape, (uint32_t *)axis, n_dims);
            break;
        case Int32:
            ret = API_LIB(trans_axis_q31)((int32_t *)src, (int32_t *)dst, (uint32_t *)in_shape, (uint32_t *)axis, n_dims);
            break;
        default:
            return T_ERR_NO_IMPLEMENTED;
    }
    
    return ret;
}

#endif