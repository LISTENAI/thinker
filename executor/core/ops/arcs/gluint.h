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
int32_t gluint_luna(tTensor *X, tTensor *Y, tTensor *workspace, GluIntAttrs *attr) {
    const int32_t Q_SIGMOID_IN = 27;
    const int32_t Q_SIGMOID_OU = 15;

    // Calculate tensor dimensions
    int32_t axis = attr->axis;
    axis = (axis < 0) ? (X->shape_.ndim_ + axis) : axis;

    #if THINKER_PARAM_CHECK
    if (axis != X->shape_.ndim_ - 1) {
        return (T_ERR_INVALID_PARA);
    }

    if (X->dtype_ != Int8 || Y->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }
#endif
    uint32_t axis_size = X->shape_.dims_[axis];
    #if THINKER_PARAM_CHECK
    if ((axis_size & 1) != 0) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    size_t output_elements = getTensorSize(Y);
    #if THINKER_PARAM_CHECK
    if (output_elements == 0 || output_elements > UINT32_MAX) {
        return (T_ERR_INVALID_DATA);
    }
#endif
    uint32_t output_size = (uint32_t)output_elements;
    uint32_t half_axis = axis_size / 2;
    #if THINKER_PARAM_CHECK
    if (X->mem_.type_ != 2 || Y->mem_.type_ != 2) {
        return (T_ERR_INVALID_PLATFROM);
    }
#endif
    #if THINKER_RUNTIME_CHECK
    if (workspace == NULL || workspace->dptr_ == 0 ||
                           workspace->mem_.type_ != 2 || workspace->dtype_ != Int8 ||
                           workspace->byte_ != 1 ||
                           getTensorDataSize(workspace) < (size_t)half_axis * 8) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif

    int32_t *a_tmp = (int32_t *)workspace->dptr_;
    int32_t *b_tmp = a_tmp + half_axis;

    // Quantization parameters
    int32_t x_q = (int32_t)X->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    #if THINKER_PARAM_CHECK
    if (!isfinite(X->scale_) || !isfinite(Y->scale_) ||
                        floorf(X->scale_) != X->scale_ ||
                        floorf(Y->scale_) != Y->scale_) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    int32_t sigmoid_shift = Q_SIGMOID_IN - x_q;
    int32_t mul_shift = 31 + x_q - y_q;
    #if THINKER_PARAM_CHECK
    if (sigmoid_shift < 0 || sigmoid_shift > 30 || mul_shift < 0 || mul_shift > 63) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    uint32_t rows = output_size / half_axis;
    for (uint32_t row = 0; row < rows; ++row) {
        int8_t *a = (int8_t *)X->dptr_ + row * axis_size;
        int8_t *b = a + half_axis;
        int8_t *out = (int8_t *)Y->dptr_ + row * half_axis;
        THINKER_RET_CHECK(API_LIB(scale_i8i8o32)(a, 1, a_tmp, half_axis, 0), "luna_scale_i8i8o32");
        THINKER_RET_CHECK(API_LIB(scale_i8i8o32)(b, 1, b_tmp, half_axis, 0), "luna_scale_i8i8o32");
        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(b_tmp, 1UL << sigmoid_shift, b_tmp, half_axis, 0), "luna_scale_i32i32o32");
        THINKER_RET_CHECK(API_LIB(sigmoid_i32o32)(b_tmp, b_tmp, half_axis), "luna_sigmoid_i32o32");
        THINKER_RET_CHECK(API_LIB(mul_i32i32o8)(a_tmp, b_tmp, out, half_axis, mul_shift), "luna_mul_i32i32o8");
    }

    return T_SUCCESS;
}

#endif  //_GLUINT_LUNA_H_
