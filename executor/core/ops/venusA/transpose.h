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
 * @param dtype Data type of the matrix (Int8, Int16, Int32)
 * @param dst Output matrix pointer
 * @param src Input matrix pointer
 * @param row Number of rows in the matrix
 * @param col Number of columns in the matrix
 * @return Execution status
 */
int32_t transpose_luna(int16_t dtype, void *dst, const void *src, int32_t row, int32_t col) {
    switch (dtype) {
        case Int8:
            return API_LIB(mat_trans_i8o8)((int8_t *)src, (int8_t *)dst, row, col);
        case Int16:
            return API_LIB(mat_trans_i16o16)((int16_t *)src, (int16_t *)dst, row, col);
        case Int32:
            return API_LIB(mat_trans_i32o32)((int32_t *)src, (int32_t *)dst, row, col);
        default:
            return T_ERR_NO_IMPLEMENTED;
    }
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
                    ret = luna_mat_trans_inv_i8o8(src, dst, row, split_num, row, col);
                }
                ret = luna_mat_trans_inv_i8o8(src, dst, row, col - (split_num - 1) * split_num, row, col);
                break;
            }
            case Int16: {
                const int16_t *src = (int16_t *)X->dptr_;
                int16_t *dst = (int16_t *)Y->dptr_;
                for (int32_t i = 0; i < split_num - 1; i++) {
                    ret = luna_mat_trans_inv_i16o16(src, dst, row, split_num, row, col);
                }
                ret = luna_mat_trans_inv_i16o16(src, dst, row, col - (split_num - 1) * split_num, row, col);
                break;
            }
            case Int32: {
                const int32_t *src = (int32_t *)X->dptr_;
                int32_t *dst = (int32_t *)Y->dptr_;
                for (int32_t i = 0; i < split_num - 1; i++) {
                    ret = luna_mat_trans_inv_i32o32(src, dst, row, split_num, row, col);
                }
                ret = luna_mat_trans_inv_i32o32(src, dst, row, col - (split_num - 1) * split_num, row, col);
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
            ret = API_LIB(trans_axis_i8o8)((int8_t *)src, (int8_t *)dst, (uint32_t *)in_shape, (uint32_t *)axis, n_dims);
            break;
        case Int16:
            ret = API_LIB(trans_axis_i16o16)((int16_t *)src, (int16_t *)dst, (uint32_t *)in_shape, (uint32_t *)axis, n_dims);
            break;
        case Int32:
            ret = API_LIB(trans_axis_i32o32)((int32_t *)src, (int32_t *)dst, (uint32_t *)in_shape, (uint32_t *)axis, n_dims);
            break;
        default:
            return T_ERR_NO_IMPLEMENTED;
    }
    
    return ret;
}

#endif