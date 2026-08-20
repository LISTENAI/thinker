#ifndef _SOFTMAXINT_LUNA_H_
#define _SOFTMAXINT_LUNA_H_

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
 * @brief Compute softmax for quantized integer data
 * @param data Input tensor
 * @param out Output tensor
 * @param Workspace Temporary workspace tensor
 * @param attrs Softmax attributes
 * @return int32_t Operation status
 */
int32_t softmaxint_luna(tTensor *data, tTensor *out, tTensor *Workspace, SoftmaxIntAttrs *attrs) {
    const int32_t SOFTMAX_Q_IN = 25;
    const int32_t SOFTMAX_Q_OUT = 15;

    #if THINKER_PARAM_CHECK
    if (data == NULL || out == NULL || Workspace == NULL || attrs == NULL ||
                        data->dptr_ == 0 || out->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
    #endif
    int32_t leading = 1, stride = 1;
    int32_t axis = attrs->axis < 0 ? (int32_t)data->shape_.ndim_ + attrs->axis : attrs->axis;
    #if THINKER_PARAM_CHECK
    if (axis < 0 || axis >= (int32_t)data->shape_.ndim_ ||
                        axis != (int32_t)data->shape_.ndim_ - 1) {
        return (T_ERR_INVALID_PARA);
    }
    if (data->dtype_ != Int8 || out->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }
    #endif
    #if THINKER_RUNTIME_CHECK
    if (Workspace == NULL || Workspace->dptr_ == 0) {
        return (T_ERR_NO_WORKSPACE);
    }
    #endif

    // Calculate leading and stride dimensions
    for (int32_t i = 0; i < axis; ++i) {
        leading *= data->shape_.dims_[i];
    }
    for (int32_t i = axis; i < data->shape_.ndim_; ++i) {
        stride *= data->shape_.dims_[i];
    }
    int32_t data_size = leading * stride;
    #if THINKER_PARAM_CHECK
    if (stride <= 0 || stride > 2048) {
        return (T_ERR_INVALID_PARA);
    }
    #endif
    int32_t input_shift = SOFTMAX_Q_IN - (int32_t)data->scale_;
    int32_t output_shift = SOFTMAX_Q_OUT - (int32_t)out->scale_;
    #if THINKER_PARAM_CHECK
    if (input_shift < 0 || input_shift > 30 ||
                        output_shift < 0 || output_shift > 63) {
        return (T_ERR_INVALID_PARA);
    }
    #endif

    if (data->dtype_ == Int8) {
        int8_t *src = (int8_t *)data->dptr_;
        int8_t *dst = (int8_t *)out->dptr_;
        int32_t *tmp1 = (int32_t *)Workspace->dptr_;
        int32_t *tmp2 = tmp1 + stride;
        int32_t workspace_size = (int32_t)getTensorDataSize(Workspace);
        #if THINKER_RUNTIME_CHECK
        if (workspace_size < stride * 8) {
            return (T_ERR_NO_WORKSPACE);
        }
        #endif

        int32_t bulk_size = data_size * 4;
        if (data->mem_.type_ != 2 || out->mem_.type_ != 2) {
            bulk_size += data_size;
        }
        if (workspace_size >= bulk_size) {
            int8_t *src_tmp = src;
            int8_t *dst_tmp = dst;
            tmp2 = tmp1 + data_size;

            if (data->mem_.type_ != 2) {
                src_tmp = (int8_t *)(tmp1 + data_size);
                memcpy(src_tmp, src, data_size);
            }

            if (out->mem_.type_ != 2) {
                dst_tmp = (int8_t *)(tmp1 + data_size);
            }

            // Scale input to Q25
            THINKER_RET_CHECK(API_LIB(scale_q7_int32)(src_tmp, 1, tmp1, data_size, 0), "luna_scale_q7_int32");
            THINKER_RET_CHECK(API_LIB(scale_q31_int32)(tmp1, (1 << input_shift), tmp1, data_size, 0), "luna_scale_q31_int32");

            // Compute Softmax
            for (int32_t l = 0; l < leading; ++l) {
                int32_t offset = l * stride;
                vec_softmax32x32(tmp1 + offset, tmp1 + offset, stride);
            }

            // Scale output to Q15
            THINKER_RET_CHECK(API_LIB(scale_q31_int8)(tmp1, 1, dst_tmp, data_size, output_shift), "luna_scale_q31_int8");

            if (out->mem_.type_ != 2) {
                memcpy(dst, dst_tmp, data_size);
            }
        } else {
            for (int32_t l = 0; l < leading; ++l) {
                int8_t *lsrc = src + l * stride;
                int8_t *ldst = dst + l * stride;

                if (data->mem_.type_ != 2) {
                    lsrc = (int8_t *)(tmp1 + stride);
                    memcpy(lsrc, src + l * stride, stride);
                }

                if (out->mem_.type_ != 2) {
                    ldst = (int8_t *)(tmp1 + stride);
                }

                // Scale input to Q25
                THINKER_RET_CHECK(API_LIB(scale_q7_int32)(lsrc, 1, tmp1, stride, 0), "luna_scale_q7_int32");
                THINKER_RET_CHECK(API_LIB(scale_q31_int32)(tmp1, (1 << input_shift), tmp2, stride, 0), "luna_scale_q31_int32");

                // Compute Softmax
                vec_softmax32x32(tmp1, tmp2, stride);

                // Scale output to Q15
                THINKER_RET_CHECK(API_LIB(scale_q31_int8)(tmp1, 1, ldst, stride, output_shift), "luna_scale_q31_int8");

                if (out->mem_.type_ != 2) {
                    memcpy(dst + l * stride, ldst, stride);
                }
            }
        }
    } else {
        printf("SoftmaxInt supports Int8 data type only.");
        #if THINKER_PARAM_CHECK
        if (1) {
            return (T_ERR_INVALID_DATATYPE);
        }
        #endif
    }

    return T_SUCCESS;
}

#endif
