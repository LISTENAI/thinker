#ifndef _GLU_LUNA_H_
#define _GLU_LUNA_H_

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/type_switch.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "thinker_status.h"
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_basic_math.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/// Gated Linear Unit (Glu) operation implementation
int32_t gluint_luna(tTensor *X, tTensor *Y, tTensor *workspace, GluIntAttrs *attr) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    const int32_t Q_SIGMOID_IN = 27;
    const int32_t Q_SIGMOID_OU = 15;

    int32_t axis = attr->axis;
    axis = (axis < 0) ? (X->shape_.ndim_ + axis) : axis;
    int32_t in_dims = X->shape_.ndim_;

    // Calculate dimensions
    uint32_t M = 1;
    for (int32_t i = 0; i < axis; i++) {
        M *= X->shape_.dims_[i];
    }

    uint32_t N = X->shape_.dims_[axis]; // Corrected N to be the size at 'axis'

    // Calculate aligned sizes
    uint32_t align_size = ALIGN8(M) * ALIGN2(N);
    uint32_t size = M * N;
    uint32_t half_size = size / 2;

    // Pointers to tensor data
    int8_t *input = (int8_t *)(X->dptr_);
    int8_t *output = (int8_t *)(Y->dptr_);
    int8_t *input_trans = (int8_t *)(workspace->dptr_);
    int8_t *output_temp = input_trans + size;

    // Matrix transpose based on size
    if (align_size <= 16384) {
        ret = API_LIB(mat_trans_i8o8)(input, input_trans, M, N);
    } else if (align_size <= 32768) {
        ret = API_LIB(split_mat_trans_i8o8)(input, input_trans, M, N);
    } else {
        return T_ERR_FAIL;
    }

    // Data type conversion and scaling
    uint32_t shift = Q_SIGMOID_IN - X->scale_;
    int16_t *input_i16 = (int16_t *)(input_trans + size);
    int32_t *input_i32 = (int32_t *)(input_i16 + size);
    int32_t *p_a = input_i32;
    int32_t *p_b = input_i32 + half_size;

    ret |= API_LIB(scale_i8i8o16)(input_trans, 1, input_i16, size, 0);
    ret |= API_LIB(scale_i16i16o32)(input_i16, 1, input_i32, size, 0);
    ret |= API_LIB(scale_i32i32o32)(p_b, 1UL << shift, p_b, half_size, 0);
    ret |= API_LIB(sigmoid_i32o32)(p_b, p_b, half_size);
    ret |= API_LIB(scale_i32i32o32)(p_b, 1, p_b, half_size, 16); // Convert to Q15
    shift = Q_SIGMOID_OU + X->scale_ - Y->scale_;
    ret |= API_LIB(mul_i32i32o8)(p_a, p_b, output_temp, half_size, shift);

    // Final matrix transpose for output
    if (align_size <= 32768) {
        ret |= API_LIB(mat_trans_i8o8)(output_temp, output, N / 2, M);
    } else if (align_size <= 65536) {
        ret |= API_LIB(split_mat_trans_i8o8)(output_temp, output, N / 2, M);
    } else {
        ret = T_ERR_FAIL;
    }

    return ret;
}

#endif  //_GLU_LUNA_H_