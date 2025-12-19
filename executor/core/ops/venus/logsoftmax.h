#ifndef _LOGSOFTMAX_LUNA_H_
#define _LOGSOFTMAX_LUNA_H_

#include <math.h>
#include <stdio.h>
#include <string.h>
#include "core/comm/utils.h"
#include "core/comm/thinker_log.h"
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif
#include "core/operator_attrs.h"
#include "hifi/NatureDSP_Signal_math.h"
#include "hifi/NatureDSP_Signal_vector.h"

/**
 * @brief Compute LogSoftmax for a tensor
 * @param data Input tensor
 * @param out Output tensor
 * @param attrs LogSoftmax attributes
 * @return int32_t Operation status
 */
int32_t logsoftmax_luna(tTensor *data, tTensor *out, LogSoftmaxAttrs *attrs) {
    // Check if input and output tensors have the same data type
    CHECK_EQ(data->dtype_, out->dtype_);

    // Adjust negative axis to positive
    if (attrs->axis < 0) {
        attrs->axis += data->shape_.ndim_;
    }

    // Ensure axis is within valid range
    CHECK_LT(attrs->axis, data->shape_.ndim_);

    // Calculate leading dimensions and stride
    int leading = 1;
    int stride = 1;
    int i = 0;

    // Compute leading dimensions up to the specified axis
    for (; i < attrs->axis; ++i) {
        leading *= data->shape_.dims_[i];
    }

    // Compute stride for the specified axis
    for (; i < data->shape_.ndim_; ++i) {
        stride *= data->shape_.dims_[i];
    }

    // Handle Float32 data type
    if (Float32 == data->dtype_) {
        float *src = (float *)(data->dptr_);
        float *dst = (float *)(out->dptr_);

        // Process each leading dimension
        for (int l = 0; l < leading; ++l) {
            float *lsrc = src + l * stride;
            float *ldst = dst + l * stride;

            // Compute Softmax
            vec_softmaxf(ldst, lsrc, stride);

            // Compute natural logarithm
            vec_lognf(lsrc, ldst, stride);

            // Copy results back to output
            memcpy(ldst, lsrc, stride * sizeof(float));
        }
    } else {
        // Log error for unsupported data types
        THINKER_LOG_FATAL("LogSoftmax support float data type only.");
        return T_ERR_FAIL;
    }

    return T_SUCCESS;
}

#endif