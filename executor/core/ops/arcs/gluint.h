#ifndef _GLUINT_LUNA_H_
#define _GLUINT_LUNA_H_

#include <math.h>
#include "core/operator_attrs.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "thinker_status.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#include "luna/luna_matrix_math.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief GluInt operation implementation
 * @param X Input tensor
 * @param Y Output tensor
 * @param workspace Workspace tensor
 * @param attr Operation attributes
 * @return int32_t Operation status
 */
int32_t gluint_luna(tTensor *X, tTensor *Y, tTensor *workspace, GluIntAttrs *attr) 
{
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    const int32_t Q_SIGMOID_IN = 27;
    const int32_t Q_SIGMOID_OU = 15;

    // Calculate tensor dimensions
    int32_t axis = attr->axis;
    axis = (axis < 0) ? (X->shape_.ndim_ + axis) : axis;
    
    // Compute matrix dimensions
    uint32_t M = 1;
    for (int32_t i = 0; i < axis; i++) {
        M *= X->shape_.dims_[i];
    }
    
    uint32_t N = X->shape_.dims_[axis];  // Use direct access instead of loop
    
    // Memory alignment and size calculations
    uint32_t align_size = ALIGN8(M) * ALIGN2(N);
    uint32_t size = M * N;
    uint32_t half_size = size / 2;

    // Pointer declarations
    int8_t *input = (int8_t *)(X->dptr_);
    int8_t *output = (int8_t *)(Y->dptr_);
    int8_t *transposed_input = (int8_t *)(workspace->dptr_);
    int8_t *temp_output = (int8_t *)(workspace->dptr_ + size);

    // Quantization parameters
    int32_t x_q = (int32_t)X->scale_;
    int32_t y_q = (int32_t)Y->scale_;

    // Matrix transpose based on size
    if (align_size <= 16384)
        ret = API_LIB(mat_trans_i8o8)(input, transposed_input, M, N);
    else if (align_size <= 32768)
        ret = API_LIB(split_mat_trans_i8o8)(input, transposed_input, M, N);
    else
        return T_ERR_FAIL;

    // Pointer to input data
    int32_t *input_data = (int32_t *)(transposed_input + size);
    int32_t *a = input_data;
    int32_t *b = input_data + half_size;

    // Data processing steps
    ret |= API_LIB(scale_i8i8o32)(transposed_input, 1, input_data, size, 0);
    ret |= API_LIB(scale_i32i32o32)(b, 1UL << (Q_SIGMOID_IN - x_q), b, half_size, 0);
    ret |= API_LIB(sigmoid_i32o32)(b, b, half_size);
    ret |= API_LIB(scale_i32i32o32)(b, 1, b, half_size, Q_SIGMOID_OU + x_q - y_q);
    
    // Final output transformation
    if (align_size <= 32768)
        ret |= API_LIB(mat_trans_i8o8)(temp_output, output, N/2, M);
    else if (align_size <= 65536)
        ret |= API_LIB(split_mat_trans_i8o8)(temp_output, output, N/2, M);
    else
        ret = T_ERR_FAIL;

    return ret;
}

#endif  //_GLUINT_LUNA_H_