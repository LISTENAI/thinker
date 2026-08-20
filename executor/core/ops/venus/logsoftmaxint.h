#ifndef _LOGSOFTMAXINT_LUNA_H_
#define _LOGSOFTMAXINT_LUNA_H_

#include <math.h>
#include <stdio.h>
#include <string.h>
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "hifi/NatureDSP_Signal_math.h"
#include "hifi/NatureDSP_Signal_vector.h"
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief Compute LogSoftmax for quantized integer tensors
 * @param data Input tensor
 * @param out Output tensor
 * @param Workspace Temporary workspace tensor
 * @param attrs LogSoftmax attributes
 * @return int32_t Operation status
 */
int32_t logsoftmaxint_luna(tTensor *data, tTensor *out, tTensor *Workspace, LogSoftmaxIntAttrs *attrs) {
    const int32_t LOG_Q_IN = 25;   // Input quantization factor
    const int32_t LOG_Q_OUT = 25;  // Output quantization factor

    #if THINKER_PARAM_CHECK
    if (data == NULL || out == NULL || Workspace == NULL || attrs == NULL ||
                        data->dptr_ == 0 || out->dptr_ == 0 || Workspace->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
    #endif
    int32_t leading = 1, stride = 1;
    int32_t axis = attrs->axis < 0 ? data->shape_.ndim_ + attrs->axis : attrs->axis;

    #if THINKER_PARAM_CHECK
    if (axis < 0 || axis >= data->shape_.ndim_ ||
                        axis != data->shape_.ndim_ - 1) {
        return (T_ERR_INVALID_PARA);
    }
    if (data->dtype_ != Int8 || out->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }
    if (data->mem_.type_ != 2 || out->mem_.type_ != 2) {
        return (T_ERR_INVALID_PLATFROM);
    }
    #endif

    // Calculate leading dimensions and stride
    for (int32_t i = 0; i < axis; ++i) {
        leading *= data->shape_.dims_[i];
    }
    for (int32_t i = axis; i < data->shape_.ndim_; ++i) {
        stride *= data->shape_.dims_[i];
    }
    #if THINKER_PARAM_CHECK
    if (stride <= 0 || stride > 2048) {
        return (T_ERR_INVALID_PARA);
    }
    #endif
    #if THINKER_RUNTIME_CHECK
    if (Workspace == NULL || Workspace->dptr_ == 0) {
        return (T_ERR_NO_WORKSPACE);
    }
    #endif
    size_t workspace_size = getTensorDataSize(Workspace);
    #if THINKER_RUNTIME_CHECK
    if (workspace_size < stride * (int32_t)sizeof(int32_t) * 2) {
        return (T_ERR_NO_WORKSPACE);
    }
    #endif

    int32_t input_shift = LOG_Q_IN - (int32_t)data->scale_;
    int32_t output_shift = LOG_Q_OUT - (int32_t)out->scale_;
    #if THINKER_PARAM_CHECK
    if (input_shift < 0 || input_shift > 30 ||
                        output_shift < 0 || output_shift > 63) {
        return (T_ERR_INVALID_PARA);
    }
    #endif

    {
        int8_t *src = (int8_t *)(data->dptr_);
        int8_t *dst = (int8_t *)(out->dptr_);
        int32_t *tmp1 = (int32_t *)(Workspace->dptr_);
        int32_t *tmp2 = tmp1 + stride;
        // Process each leading dimension
        for (int32_t l = 0; l < leading; ++l) {
            int8_t *lsrc = src + l * stride;
            int8_t *ldst = dst + l * stride;

            // Scale input to Q25 format
            THINKER_RET_CHECK(API_LIB(scale_q7_int32)(lsrc, 1, tmp1, stride, 0), "luna_scale_q7_int32");
            // Apply quantization factor and scale to Q25
            THINKER_RET_CHECK(API_LIB(scale_q31_int32)(tmp1, (1 << input_shift), tmp2, stride, 0), "luna_scale_q31_int32");
            // Compute Softmax in Q25 format
            vec_softmax32x32((int32_t *)tmp1, (int32_t *)tmp2, stride);
            // Compute natural logarithm in Q25 format
            vec_logn_32x32((int32_t *)tmp2, (int32_t *)tmp1, stride);
            // Scale output to Q8 format
            THINKER_RET_CHECK(API_LIB(scale_q31_int8)(tmp2, 1, ldst, stride, output_shift), "luna_scale_q31_int8");
        }
    }

    return T_SUCCESS;
}

#endif
