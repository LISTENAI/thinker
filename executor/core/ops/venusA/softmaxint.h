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


static int32_t softmaxint_dtype_size(uint16_t dtype) {
    if (dtype == Int8) {
        return 1;
    } else if (dtype == Int16) {
        return 2;
    } else if (dtype == Int32) {
        return 4;
    }
    return 0;
}

static int32_t softmaxint_scale_i32_to_dtype(const int32_t *src, void *dst,
                                             uint16_t dtype, int32_t size,
                                             int32_t shift) {
    int32_t scalar = 1;
    uint32_t rshift = 0;
    if (shift >= 0) {
#if THINKER_PARAM_CHECK
if (shift > 63) {
    return (T_ERR_INVALID_PARA);
}
#endif
        rshift = (uint32_t)shift;
    } else {
#if THINKER_PARAM_CHECK
if ((-shift) > 30) {
    return (T_ERR_INVALID_PARA);
}
#endif
        scalar = 1 << (uint32_t)(-shift);
    }

    if (dtype == Int8) {
        THINKER_RET_CHECK(API_LIB(scale_i32i32o8)(src, scalar, (int8_t *)dst, size, rshift),
                          "luna_scale_i32i32o8");
    } else if (dtype == Int16) {
        THINKER_RET_CHECK(API_LIB(scale_i32i32o16)(src, scalar, (int16_t *)dst, size, rshift),
                          "luna_scale_i32i32o16");
    } else if (dtype == Int32) {
        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(src, scalar, (int32_t *)dst, size, rshift),
                          "luna_scale_i32i32o32");
    } else {
        return T_ERR_INVALID_DATATYPE;
    }
    return T_SUCCESS;
}

static int32_t softmaxint_scale_i32_to_q25(int32_t *src, int32_t size, int32_t shift) {
    int32_t scalar = 1;
    uint32_t rshift = 0;
    if (shift >= 0) {
#if THINKER_PARAM_CHECK
if (shift > 30) {
    return (T_ERR_INVALID_PARA);
}
#endif
        scalar = 1 << (uint32_t)shift;
    } else {
#if THINKER_PARAM_CHECK
if ((-shift) > 63) {
    return (T_ERR_INVALID_PARA);
}
#endif
        rshift = (uint32_t)(-shift);
    }
    THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(src, scalar, src, size, rshift),
                      "luna_scale_i32i32o32");
    return T_SUCCESS;
}

static int32_t softmaxint_workspace_bytes(uint16_t dtype, int32_t rows,
                                          int32_t stride) {
    int32_t cur_size = rows * stride;
    if (dtype == Int8) {
        return ALIGN4(cur_size * (int32_t)sizeof(int16_t)) +
               cur_size * (int32_t)sizeof(int32_t) +
               stride * (int32_t)sizeof(int32_t);
    }
    return cur_size * (int32_t)sizeof(int32_t) +
           stride * (int32_t)sizeof(int32_t);
}

static int32_t softmaxint_split_leading(uint16_t dtype, int32_t leading,
                                        int32_t stride,
                                        int32_t workspace_size) {
    int32_t per_row = stride * (dtype == Int8 ? 6 : 4);
    int32_t fixed = stride * (int32_t)sizeof(int32_t);
    int32_t split = (workspace_size > fixed && per_row > 0)
                        ? (workspace_size - fixed) / per_row
                        : 0;
    if (split > leading) {
        split = leading;
    }
    while (split > 0 &&
           softmaxint_workspace_bytes(dtype, split, stride) > workspace_size) {
        split--;
    }
    return split;
}

/**
 * @brief Perform integer softmax operation
 * @param data Input tensor
 * @param out Output tensor
 * @param Workspace Temporary workspace for calculations
 * @param attrs Softmax attributes containing axis and scaling parameters
 * @return Execution status
 */
int32_t softmaxint_luna(tTensor *data, tTensor *out, tTensor *Workspace, SoftmaxIntAttrs *attrs) {
    const int32_t SOFTMAX_Q_IN = 25;
    const int32_t SOFTMAX_Q_OUT = 15;

    if (Workspace == NULL || Workspace->dptr_ == 0) {
        return T_ERR_NO_WORKSPACE;
    }

    int32_t axis = attrs->axis < 0 ? ((int32_t)data->shape_.ndim_ + attrs->axis) : attrs->axis;
#if THINKER_PARAM_CHECK
if (axis < 0 || axis >= (int32_t)data->shape_.ndim_) {
    return (T_ERR_INVALID_PARA);
}

if (axis != (int32_t)data->shape_.ndim_ - 1) {
    return (T_ERR_NO_IMPLEMENTED);
}
#endif

    int32_t leading = 1;
    int32_t stride = 1;
    for (int32_t i = 0; i < axis; ++i) {
        leading *= data->shape_.dims_[i];
    }
    for (int32_t i = axis; i < (int32_t)data->shape_.ndim_; ++i) {
        stride *= data->shape_.dims_[i];
    }
#if THINKER_PARAM_CHECK
if (stride <= 0 || stride > 2048) {
    return (T_ERR_INVALID_PARA);
}

if (softmaxint_dtype_size(data->dtype_) == 0 ||
        softmaxint_dtype_size(out->dtype_) == 0) {
    return (T_ERR_INVALID_DATATYPE);
}
#endif

    int32_t out_bytes = softmaxint_dtype_size(out->dtype_);
    int32_t workspace_size = (int32_t)getTensorSize(Workspace) * Workspace->byte_;
    int32_t split_leading = softmaxint_split_leading(data->dtype_, leading, stride,
                                                     workspace_size);
    if (split_leading <= 0) {
        return T_ERR_NO_WORKSPACE;
    }

    int32_t x_scale = (int32_t)data->scale_;
    int32_t y_scale = (int32_t)out->scale_;
    int32_t input_shift = SOFTMAX_Q_IN - x_scale;
    int32_t output_shift = SOFTMAX_Q_OUT - y_scale;
    int32_t y_in_psram = (out->mem_.type_ != 2);
    int32_t paste_leading = 0;

    while (paste_leading < leading) {
        int32_t remain_leading = leading - paste_leading;
        int32_t cur_leading = remain_leading > split_leading ? split_leading : remain_leading;
        int32_t cur_size = cur_leading * stride;
        int8_t *workspace = (int8_t *)Workspace->dptr_;
        int32_t *src_i32 = NULL;
        int32_t *softmax_tmp = NULL;

        if (data->dtype_ == Int8) {
            int16_t *src_i16 = (int16_t *)workspace;
            src_i32 = (int32_t *)(workspace + ALIGN4(cur_size * (int32_t)sizeof(int16_t)));
            softmax_tmp = src_i32 + cur_size;
            THINKER_RET_CHECK(API_LIB(scale_i8i8o16)((int8_t *)data->dptr_ + paste_leading * stride,
                                                     1, src_i16, cur_size, 0),
                              "luna_scale_i8i8o16");
            THINKER_RET_CHECK(API_LIB(scale_i16i16o32)(src_i16, 1, src_i32, cur_size, 0),
                              "luna_scale_i16i16o32");
        } else if (data->dtype_ == Int16) {
            src_i32 = (int32_t *)workspace;
            softmax_tmp = src_i32 + cur_size;
            THINKER_RET_CHECK(API_LIB(scale_i16i16o32)((int16_t *)data->dptr_ + paste_leading * stride,
                                                       1, src_i32, cur_size, 0),
                              "luna_scale_i16i16o32");
        } else {
            src_i32 = (int32_t *)workspace;
            softmax_tmp = src_i32 + cur_size;
            THINKER_RET_CHECK(API_LIB(scale_i32i32o32)((int32_t *)data->dptr_ + paste_leading * stride,
                                                       1, src_i32, cur_size, 0),
                              "luna_scale_i32i32o32");
        }

        THINKER_RET_CHECK(softmaxint_scale_i32_to_q25(src_i32, cur_size, input_shift),
                          "softmaxint_scale_i32_to_q25");

        for (int32_t l = 0; l < cur_leading; ++l) {
            int32_t offset = l * stride;
            int8_t *out_ptr = (int8_t *)out->dptr_ + (paste_leading + l) * stride * out_bytes;
            void *dst = (void *)out_ptr;

            THINKER_RET_CHECK(API_LIB(softmax_i32o32)(src_i32 + offset, softmax_tmp, stride),
                              "luna_softmax_i32o32");

            if (y_in_psram) {
                dst = (out->dtype_ == Int32) ? (void *)softmax_tmp : (void *)workspace;
            }

            THINKER_RET_CHECK(softmaxint_scale_i32_to_dtype(softmax_tmp, dst, out->dtype_,
                                                            stride, output_shift),
                              "softmaxint_scale_i32_to_dtype");

            if (y_in_psram) {
                opi_psram_cpy_out(out_ptr, dst, stride * out_bytes);
            }
        }

        paste_leading += cur_leading;
    }

    return T_SUCCESS;
}

#endif
