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
 * @brief Softmax operation for integer tensors
 * @param data Input tensor
 * @param out Output tensor
 * @param Workspace Workspace buffer
 * @param attrs Softmax attributes
 * @return Operation result status
 */
int32_t softmaxint_luna(tTensor *data, tTensor *out, tTensor *Workspace, SoftmaxIntAttrs *attrs) {
    const int32_t SOFTMAX_Q_IN = 25;
    const int32_t SOFTMAX_Q_OUT = 15;
    int32_t leading = 1;
    int32_t axis = attrs->axis < 0 ? ((int32_t)data->shape_.ndim_ + attrs->axis) : attrs->axis;

    #if THINKER_PARAM_CHECK
    if (axis < 0 || axis >= (int32_t)data->shape_.ndim_) {
        return (T_ERR_INVALID_PARA);
    }

    if (axis != (int32_t)data->shape_.ndim_ - 1) {
        return (T_ERR_INVALID_PARA);
    }

    if (data->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (out->dtype_ != Int8 && out->dtype_ != Int32) {
        return (T_ERR_INVALID_DATATYPE);
    }
#endif

    for (int32_t i = 0; i < axis; ++i) {
        leading *= data->shape_.dims_[i];
    }
    int32_t stride = data->shape_.dims_[axis];
    int32_t data_size = leading * stride;

    #if THINKER_PARAM_CHECK
    if (stride <= 0 || stride > 2048) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    #if THINKER_RUNTIME_CHECK
    if (Workspace == NULL || Workspace->dptr_ == 0 ||
                          Workspace->mem_.type_ != 2) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif

    int32_t input_is_psram = (data->mem_.type_ != 2);
    int32_t output_is_psram = (out->mem_.type_ != 2);
    int32_t need_softmax_tmp = (out->dtype_ == Int8) || output_is_psram;
    int32_t need_i8_tmp = input_is_psram || (out->dtype_ == Int8 && output_is_psram);
    int32_t required_workspace = data_size * (int32_t)sizeof(int32_t);
    if (need_softmax_tmp) {
        required_workspace += stride * (int32_t)sizeof(int32_t);
    }
    if (need_i8_tmp) {
        required_workspace += stride * (int32_t)sizeof(int8_t);
    }

    int32_t workspace_size = (int32_t)getTensorSize(Workspace) * Workspace->byte_;
    #if THINKER_RUNTIME_CHECK
    if (workspace_size < required_workspace) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif

    int32_t x_scale = (int32_t)data->scale_;
    int32_t y_scale = (int32_t)out->scale_;
    int32_t input_shift = SOFTMAX_Q_IN - x_scale;
    int32_t output_shift = SOFTMAX_Q_OUT - y_scale;
    #if THINKER_PARAM_CHECK
    if (input_shift < 0 || input_shift > 30) {
        return (T_ERR_INVALID_PARA);
    }

    if (out->dtype_ == Int8 && (output_shift < 0 || output_shift > 63)) {
        return (T_ERR_INVALID_PARA);
    }

    if (out->dtype_ == Int32 && y_scale != SOFTMAX_Q_OUT) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    int32_t *data_temp = (int32_t *)(Workspace->dptr_);
    int32_t *softmax_tmp = data_temp + data_size;
    int8_t *i8_tmp = (int8_t *)(softmax_tmp + (need_softmax_tmp ? stride : 0));

    for (int32_t l = 0; l < leading; ++l) {
        int32_t offset = l * stride;
        int8_t *src = (int8_t *)data->dptr_ + offset;
        int32_t *data_vec = data_temp + offset;
        if (input_is_psram) {
            THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(i8_tmp, src, stride), "luna_memcpy_i8o8");
            src = i8_tmp;
        }
        THINKER_RET_CHECK(API_LIB(scale_i8i8o32)(src, 1, data_vec, stride, 0),
                          "luna_scale_i8i8o32");
        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(data_vec, (1 << input_shift), data_vec, stride, 0),
                          "luna_scale_i32i32o32");

        if (out->dtype_ == Int8) {
            int8_t *dst = (int8_t *)out->dptr_ + offset;
            THINKER_RET_CHECK(API_LIB(softmax_i32o32)(data_vec, softmax_tmp, stride),
                              "luna_softmax_i32o32");
            if (output_is_psram) {
                THINKER_RET_CHECK(API_LIB(scale_i32i32o8)(softmax_tmp, 1, i8_tmp, stride, output_shift),
                                  "luna_scale_i32i32o8");
                opi_psram_cpy_out(dst, i8_tmp, stride * sizeof(int8_t));
            } else {
                THINKER_RET_CHECK(API_LIB(scale_i32i32o8)(softmax_tmp, 1, dst, stride, output_shift),
                                  "luna_scale_i32i32o8");
            }
        } else {
            int32_t *dst = (int32_t *)out->dptr_ + offset;
            int32_t *dst_tmp = output_is_psram ? softmax_tmp : dst;
            THINKER_RET_CHECK(API_LIB(softmax_i32o32)(data_vec, dst_tmp, stride),
                              "luna_softmax_i32o32");
            if (output_is_psram) {
                opi_psram_cpy_out((int8_t *)dst, (int8_t *)dst_tmp, stride * sizeof(int32_t));
            }
        }
    }

    return T_SUCCESS;
}
#endif
