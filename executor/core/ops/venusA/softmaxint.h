#ifndef _SOFTMAXINT_LUNA_H_
#define _SOFTMAXINT_LUNA_H_

#include <math.h>
#include <stdio.h>
#include <string.h>
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "thinker_status.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief Perform integer softmax operation
 * @param data Input tensor
 * @param out Output tensor
 * @param Workspace Temporary workspace for calculations
 * @param attrs Softmax attributes containing axis and scaling parameters
 * @return Execution status
 */
int32_t softmaxint_luna(tTensor *data, tTensor *out, tTensor *Workspace, SoftmaxIntAttrs *attrs) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    const int32_t SOFTMAX_Q_IN = 25;
    const int32_t SOFTMAX_Q_OUT = 15;

    int32_t leading = 1, stride = 1;
    int32_t axis = attrs->axis < 0 ? (data->shape_.ndim_ + attrs->axis) : attrs->axis;

    // Calculate leading and stride dimensions based on axis
    for (int32_t i = 0; i < axis; ++i) {
        leading *= data->shape_.dims_[i];
    }
    for (int32_t i = axis; i < data->shape_.ndim_; ++i) {
        stride *= data->shape_.dims_[i];
    }
    int32_t data_size = leading * stride;

    if (Int8 != data->dtype_ && Int16 != data->dtype_ && Int32 != data->dtype_) {
        return T_ERR_INVALID_DATATYPE;
    }
    if (Int8 != out->dtype_ && Int16 != out->dtype_ && Int32 != out->dtype_) {
        return T_ERR_INVALID_DATATYPE;
    }

    // Check if output is in PSRAM
    if (out->mem_.type_ != 2) {
        return T_ERR_INVALID_PLATFROM;
    }

    int32_t x_scale = (int32_t)data->scale_;
    int32_t y_scale = (int32_t)out->scale_;

    // Process based on input data type
    if (data->dtype_ == Int8) {
        int16_t *p_tmp0 = (int16_t *)Workspace->dptr_;
        int32_t *p_tmp1 = (int32_t *)(p_tmp0 + data_size);
        int32_t *dst_tmp = p_tmp1 + 4 * data_size;

        // Scale from Int8 to Int16
        ret = API_LIB(scale_i8i8o16)((int8_t *)data->dptr_, 1, p_tmp0, data_size, 0);
        // Scale from Int16 to Int32
        ret |= API_LIB(scale_i16i16o32)(p_tmp0, 1, p_tmp1, data_size, 0);
        // Apply input scaling
        ret |= API_LIB(scale_i32i32o32)(p_tmp1, (1 << (SOFTMAX_Q_IN - x_scale)), p_tmp1, data_size, 0);

        // Compute softmax
        for (int32_t l = 0; l < leading; ++l) {
            int32_t offset = l * stride;
            ret |= API_LIB(softmax_i32o32)(p_tmp1 + offset, (int32_t *)dst_tmp + offset, stride);
        }

        // Scale output based on output data type
        if (out->dtype_ == Int8) {
            ret |= API_LIB(scale_i32i32o8)((int32_t *)dst_tmp, 1, (int8_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale));
        } else if (out->dtype_ == Int16) {
            ret |= API_LIB(scale_i32i32o16)((int32_t *)dst_tmp, 1, (int16_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale));
        } else {
            ret |= API_LIB(scale_i32i32o32)((int32_t *)dst_tmp, 1, (int32_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale));
        }
    } else if (data->dtype_ == Int16) {
        int32_t *p_tmp = (int32_t *)Workspace->dptr_;
        int32_t *dst_tmp = p_tmp + 4 * data_size;

        // Scale from Int16 to Int32
        ret = API_LIB(scale_i16i16o32)((int16_t *)data->dptr_, 1, p_tmp, data_size, 0);
        // Apply input scaling
        ret |= API_LIB(scale_i32i32o32)(p_tmp, (1 << (SOFTMAX_Q_IN - x_scale)), p_tmp, data_size, 0);

        // Compute softmax
        for (int32_t l = 0; l < leading; ++l) {
            int32_t offset = l * stride;
            ret |= API_LIB(softmax_i32o32)(p_tmp + offset, (int32_t *)dst_tmp + offset, stride);
        }

        // Scale output based on output data type
        if (out->dtype_ == Int8) {
            ret |= API_LIB(scale_i32i32o8)((int32_t *)dst_tmp, 1, (int8_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale));
        } else if (out->dtype_ == Int16) {
            ret |= API_LIB(scale_i32i32o16)((int32_t *)dst_tmp, 1, (int16_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale));
        } else {
            ret |= API_LIB(scale_i32i32o32)((int32_t *)dst_tmp, 1, (int32_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale));
        }
    } else if (data->dtype_ == Int32) {
        int32_t *p_tmp = (int32_t *)Workspace->dptr_;
        int32_t *dst_tmp = p_tmp + 4 * stride;

        // Apply input scaling
        ret = API_LIB(scale_i32i32o32)((int32_t *)data->dptr_, (1 << (SOFTMAX_Q_IN - x_scale)), p_tmp, data_size, 0);

        // Compute softmax
        for (int32_t l = 0; l < leading; ++l) {
            int32_t offset = l * stride;
            ret |= API_LIB(softmax_i32o32)(p_tmp + offset, (int32_t *)dst_tmp + offset, stride);
        }

        // Scale output based on output data type
        if (out->dtype_ == Int8) {
            ret |= API_LIB(scale_i32i32o8)((int32_t *)dst_tmp, 1, (int8_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale));
        } else if (out->dtype_ == Int16) {
            ret |= API_LIB(scale_i32i32o16)((int32_t *)dst_tmp, 1, (int16_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale));
        } else {
            ret |= API_LIB(scale_i32i32o32)((int32_t *)dst_tmp, 1, (int32_t *)out->dptr_, data_size, (SOFTMAX_Q_OUT - y_scale));
        }
    }

    return ret;
}

#endif