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

    int32_t leading = 1, stride = 1;
    int32_t i = 0;
    int32_t axis = 1;

    // Adjust axis if it's set to -1
    if (-1 == attrs->axis) {
        axis = data->shape_.ndim_ - 1;
    }

    // Calculate leading dimensions and stride
    for (; i < axis; ++i) {
        leading *= data->shape_.dims_[i];
    }
    for (; i < data->shape_.ndim_; ++i) {
        stride *= data->shape_.dims_[i];
    }

    tStatus ret = T_ERR_NO_IMPLEMENTED;

    if (Int8 == data->dtype_) {
        int8_t *src = (int8_t *)(data->dptr_);
        int8_t *dst = (int8_t *)(out->dptr_);
        int32_t *tmp1 = (int32_t *)(Workspace->dptr_);
        int32_t *tmp2 = tmp1 + stride;
        int32_t x_scale = (int32_t)data->scale_;
        int32_t y_scale = (int32_t)out->scale_;

        // Process each leading dimension
        for (int32_t l = 0; l < leading; ++l) {
            int8_t *lsrc = src + l * stride;
            int8_t *ldst = dst + l * stride;

            // Scale input to Q25 format
            ret = API_LIB(scale_q7_int32)(lsrc, 1, tmp1, stride, 0);
            // Apply quantization factor and scale to Q25
            ret |= API_LIB(scale_q31_int32)(tmp1, (1 << (LOG_Q_IN - x_scale)), tmp2, stride, 0);
            // Compute Softmax in Q25 format
            vec_softmax32x32((int32_t *)tmp1, (int32_t *)tmp2, stride);
            // Compute natural logarithm in Q25 format
            vec_logn_32x32((int32_t *)tmp2, (int32_t *)tmp1, stride);
            // Scale output to Q8 format
            ret |= API_LIB(scale_q31_int8)(tmp2, 1, ldst, stride, (LOG_Q_OUT - y_scale));
        }
    } else {
        THINKER_LOG_FATAL("LogSoftmaxInt support int8 data type only.");
        ret = T_ERR_INVALID_DATATYPE;
    }

    return ret;
}

#endif