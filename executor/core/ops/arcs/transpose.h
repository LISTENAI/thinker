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
 * @brief Transpose matrix based on data type
 * @param dtype Data type (Int8 or Int32)
 * @param dst Destination buffer
 * @param src Source buffer
 * @param row Number of rows
 * @param col Number of columns
 * @return Operation status
 */
static int32_t transpose_luna(int16_t dtype, void *dst, const void *src, int32_t row, int32_t col) 
{
    if (dtype == Int8)
        return API_LIB(split_mat_trans_i8o8)(src, dst, row, col);
    else if (dtype == Int32)
        return API_LIB(split_mat_trans_i32o32)(src, dst, row, col);
    else
        return T_ERR_NO_IMPLEMENTED;
}

/**
 * @brief Split transpose operation for large matrices
 * @param Y Output tensor
 * @param X Input tensor
 * @param Temp Temporary buffer
 * @param row Number of rows
 * @param col Number of columns
 * @param split_num Split factor
 * @return Operation status
 */
static int32_t split_transpose_luna(tTensor *Y, tTensor *X, tTensor *Temp, int32_t row, int32_t col, int32_t split_num)
{
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
    if (X->dtype_ == Int8) {
        const int8_t *src = (int8_t *)X->dptr_;
        int8_t *dst = (int8_t *)Y->dptr_;
        
        if (X->mem_.type_ != 2) {
            // Process all splits except the last one
            for (int32_t i = 0; i < split_num - 1; i++) {
                ret = luna_mat_trans_inv_i8o8(src, dst, row, split_num, row, col);
            }
            // Process the last split
            ret = luna_mat_trans_inv_i8o8(src, dst, row, col - (split_num - 1) * split_num, row, col);
        }
    }
    else if (X->dtype_ == Int32) {
        // TODO: Implement Int32 case
    }
    else {
        return T_ERR_NO_IMPLEMENTED;
    }
    
    return ret;
}

/**
 * @brief Transpose tensor along specified axes
 * @param dtype Data type (Int8 or Int32)
 * @param src Source buffer
 * @param dst Destination buffer
 * @param in_shape Input shape array
 * @param axis Axis permutation array
 * @param n_dims Number of dimensions
 * @return Operation status
 */
int32_t transpose_axis_luna(int16_t dtype, void *src, void *dst, int32_t *in_shape, int32_t *axis, uint32_t n_dims) 
{
    int32_t ret = T_ERR_FAIL;

    // Only support 3D tensors
    if (3 != n_dims) {
        return ret;
    }

    if (dtype == Int8)
        ret = API_LIB(trans_axis_i8o8)(src, dst, (uint32_t *)in_shape, (uint32_t *)axis, n_dims);
    else if (dtype == Int32)
        ret = API_LIB(trans_axis_i32o32)(src, dst, (uint32_t *)in_shape, (uint32_t *)axis, n_dims);
    else
        return T_ERR_NO_IMPLEMENTED;
        
    return ret;
}

#endif