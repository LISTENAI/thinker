#ifndef _PAD_LUNA_H_
#define _PAD_LUNA_H_

#include <string.h>
#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#include "luna/luna_cnn_tools.h"
#define API_LIB(api) luna_##api
#endif
#include "thinker_status.h"


static int32_t iqpad_luna_generic(tTensor *X, tTensor *P, tTensor *data, tTensor *workspace, tTensor *Y, iqPadAttrs *attrs) {
    int32_t ndim = X->shape_.ndim_;
#if THINKER_PARAM_CHECK
if (ndim <= 0 || ndim > 4 || X->dtype_ != Int8 || Y->dtype_ != Int8) {
    return (T_ERR_INVALID_PARA);
}
#endif
    int8_t fill_data = *((int8_t *)data->dptr_);
#if THINKER_PARAM_CHECK
if (fill_data != 0) {
    return (T_ERR_INVALID_PARA);
}
#endif

    int32_t pads_len = P->shape_.dims_[0];
    int32_t before[4] = {0};
    int32_t after[4] = {0};
    int64_t *pads = (int64_t *)P->dptr_;
    if (pads_len == 2 * ndim) {
        for (int32_t i = 0; i < ndim; ++i) {
            before[i] = (int32_t)pads[i];
            after[i] = (int32_t)pads[i + ndim];
        }
    } else if (pads_len <= 8 && (pads_len & 1) == 0) {
        int32_t pad_dims = pads_len >> 1;
#if THINKER_PARAM_CHECK
if (pad_dims > ndim) {
    return (T_ERR_INVALID_PARA);
}
#endif
        int32_t base = ndim - pad_dims;
        for (int32_t i = 0; i < pad_dims; ++i) {
            before[base + i] = (int32_t)pads[i];
            after[base + i] = (int32_t)pads[i + pad_dims];
        }
    } else {
#if THINKER_PARAM_CHECK
if (1) {
    return (T_ERR_INVALID_PARA);
}
#endif
    }

    int32_t in_stride[4] = {1, 1, 1, 1};
    int32_t out_dims[4] = {1, 1, 1, 1};
    int32_t out_coord[4] = {0, 0, 0, 0};
    int32_t total_out = 1;
    for (int32_t i = ndim - 1; i >= 0; --i) {
#if THINKER_PARAM_CHECK
if (before[i] < 0 || after[i] < 0) {
    return (T_ERR_INVALID_PARA);
}

if (attrs->mode == 2 && (before[i] >= X->shape_.dims_[i] || after[i] >= X->shape_.dims_[i])) {
    return (T_ERR_INVALID_PARA);
}
#endif
        out_dims[i] = X->shape_.dims_[i] + before[i] + after[i];
#if THINKER_PARAM_CHECK
if (out_dims[i] != Y->shape_.dims_[i]) {
    return (T_ERR_INVALID_PARA);
}
#endif
        total_out *= out_dims[i];
        if (i + 1 < ndim) {
            in_stride[i] = in_stride[i + 1] * X->shape_.dims_[i + 1];
        }
    }

    int8_t *dst = (int8_t *)Y->dptr_;
    int32_t dst_in_psram = (Y->mem_.type_ != 2);
    if (dst_in_psram) {
        size_t workspace_size = workspace ? getTensorDataSize(workspace) : 0;
        if (workspace == NULL || workspace_size < total_out) {
            return T_ERR_NO_WORKSPACE;
        }
        dst = (int8_t *)workspace->dptr_;
    }

    int8_t *src = (int8_t *)X->dptr_;
    for (int32_t out_idx = 0; out_idx < total_out; ++out_idx) {
        int32_t src_offset = 0;
        int32_t use_pad = 0;
        for (int32_t d = 0; d < ndim; ++d) {
            int32_t pos = out_coord[d] - before[d];
            if (pos < 0 || pos >= X->shape_.dims_[d]) {
                if (attrs->mode == 0) {
                    use_pad = 1;
                    break;
                } else if (attrs->mode == 1) {
                    pos = pos < 0 ? 0 : X->shape_.dims_[d] - 1;
                } else if (attrs->mode == 2) {
                    pos = pos < 0 ? -pos : (2 * X->shape_.dims_[d] - 2 - pos);
#if THINKER_PARAM_CHECK
if (pos < 0 || pos >= X->shape_.dims_[d]) {
    return (T_ERR_INVALID_PARA);
}
#endif
                } else {
#if THINKER_PARAM_CHECK
if (1) {
    return (T_ERR_INVALID_PARA);
}
#endif
                }
            }
            src_offset += pos * in_stride[d];
        }
        dst[out_idx] = use_pad ? fill_data : src[src_offset];

        for (int32_t d = ndim - 1; d >= 0; --d) {
            out_coord[d]++;
            if (out_coord[d] < out_dims[d]) {
                break;
            }
            out_coord[d] = 0;
        }
    }

    if (dst_in_psram) {
        opi_psram_cpy_out((void *)Y->dptr_, dst, total_out);
    }
    return T_SUCCESS;
}

/**
 * @brief Tensor padding operation implementation
 * @param X Input tensor
 * @param P Padding parameters tensor
 * @param data Fill data tensor
 * @param workspace Temporary workspace tensor
 * @param Y Output tensor
 * @param attrs Padding attributes
 * @return int32_t Operation status
 */
int32_t iqpad_luna(tTensor *X, tTensor *P, tTensor *data, tTensor *workspace, tTensor *Y, iqPadAttrs *attrs) {
    #if THINKER_PARAM_CHECK
    if (X == NULL || P == NULL || data == NULL || Y == NULL ||
                        attrs == NULL || X->dptr_ == 0 || P->dptr_ == 0 ||
                        data->dptr_ == 0 || Y->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }

    if (X->dtype_ != Int8 || Y->dtype_ != Int8 ||
                        P->dtype_ != Int64 || data->dtype_ != Int8 ||
                        P->shape_.ndim_ != 1 || getTensorSize(data) != 1) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (X->mem_.type_ != 2 ||
                        (Y->mem_.type_ != 1 && Y->mem_.type_ != 2)) {
        return (T_ERR_NO_SUPPORT_OP);
    }
#endif
    if (X->shape_.ndim_ != 4) {
        return iqpad_luna_generic(X, P, data, workspace, Y, attrs);
    }
    #if THINKER_PARAM_CHECK
    if (X->shape_.dims_[0] != 1 || Y->shape_.ndim_ != 4 ||
                        Y->shape_.dims_[0] != 1 ||
                        Y->shape_.dims_[1] != X->shape_.dims_[1]) {
        return (T_ERR_INVALID_DATA);
    }
#endif

    // Get input and output dimensions
    int32_t c_in = X->shape_.dims_[1];
    int32_t h_in = X->shape_.dims_[2];
    int32_t w_in = X->shape_.dims_[3];
    int32_t h_out = Y->shape_.dims_[2];
    int32_t w_out = Y->shape_.dims_[3];
    int32_t in_size = c_in * h_in * w_in;
    int32_t out_size = c_in * h_out * w_out;

    // Parse padding parameters
    int64_t pads[8] = {0};
    int32_t pads_h_up = 0, pads_h_down = 0, pads_w_left = 0, pads_w_right = 0;
    switch (P->shape_.dims_[0]) {
        case 4:
            pads_h_up = pads[0] = *((int64_t *)P->dptr_);
            pads_w_left = pads[1] = *((int64_t *)P->dptr_ + 1);
            pads_h_down = pads[2] = *((int64_t *)P->dptr_ + 2);
            pads_w_right = pads[3] = *((int64_t *)P->dptr_ + 3);
            break;
        case 6:
            pads[0] = *((int64_t *)P->dptr_);
            pads_h_up = pads[1] = *((int64_t *)P->dptr_ + 1);
            pads_w_left = pads[2] = *((int64_t *)P->dptr_ + 2);
            pads_h_down = pads[4] = *((int64_t *)P->dptr_ + 4);
            pads_w_right = pads[5] = *((int64_t *)P->dptr_ + 5);
            pads[3] = *((int64_t *)P->dptr_ + 3);
#if THINKER_PARAM_CHECK
if (pads[0] != 0 || pads[3] != 0) {
    return (T_ERR_INVALID_PARA);
}
#endif
            break;
        case 8:
            pads[0] = *((int64_t *)P->dptr_);
            pads[1] = *((int64_t *)P->dptr_ + 1);
            pads_h_up = pads[2] = *((int64_t *)P->dptr_ + 2);
            pads_w_left = pads[3] = *((int64_t *)P->dptr_ + 3);
            pads_h_down = pads[6] = *((int64_t *)P->dptr_ + 6);
            pads_w_right = pads[7] = *((int64_t *)P->dptr_ + 7);
            pads[4] = *((int64_t *)P->dptr_ + 4);
            pads[5] = *((int64_t *)P->dptr_ + 5);
#if THINKER_PARAM_CHECK
if (pads[0] != 0 || pads[1] != 0 || pads[4] != 0 || pads[5] != 0) {
    return (T_ERR_INVALID_PARA);
}
#endif
            break;
        default:
#if THINKER_PARAM_CHECK
if (1) {
    return (T_ERR_INVALID_PARA);
}
#endif
            break;
    }

    // Get input and output pointers
    int8_t *src = (int8_t *)X->dptr_;
    int8_t *dst = (int8_t *)Y->dptr_;
    int8_t mode = attrs->mode;
    bool srcInPSRAM = (X->mem_.type_ != 2);
    bool dstInPSRAM = (Y->mem_.type_ != 2);

    // Check mode validity
#if THINKER_PARAM_CHECK
if (mode < 0 || mode > 2) {
    return (T_ERR_INVALID_PARA);
}
#endif

    // Get fill data
    int8_t fill_data = *((int8_t *)data->dptr_);
#if THINKER_PARAM_CHECK
if (fill_data != 0) {
    return (T_ERR_INVALID_PARA);
}

if (pads_h_up < 0 || pads_h_down < 0 || pads_w_left < 0 || pads_w_right < 0) {
    return (T_ERR_INVALID_PARA);
}

    if (h_out != h_in + pads_h_up + pads_h_down ||
                        w_out != w_in + pads_w_left + pads_w_right) {
        return (T_ERR_INVALID_DATA);
    }

if (attrs->mode == 2 &&
        (pads_h_up >= h_in || pads_h_down >= h_in || pads_w_left >= w_in || pads_w_right >= w_in)) {
    return (T_ERR_INVALID_PARA);
}
#endif
    size_t workspace_size = workspace ? getTensorDataSize(workspace) : 0;
    int32_t required_workspace = out_size + (dstInPSRAM ? out_size : in_size);
    if (workspace == NULL || workspace_size < required_workspace) {
        return T_ERR_NO_WORKSPACE;
    }

    // Temporary workspace
    int8_t *temp = (int8_t *)workspace->dptr_;
    int8_t *src_temp = temp + c_in * h_out * w_out;
    int8_t *dst_temp = temp;

    if (srcInPSRAM) {
        int8_t *src_temp1 = temp;
        THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(src_temp1, src, in_size), "luna_memcpy_i8o8");
        THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(src_temp1, src_temp, c_in, h_in * w_in), "luna_split_mat_trans_i8o8");
    }
    else
        THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(src, src_temp, c_in, h_in * w_in), "luna_split_mat_trans_i8o8");

    // Perform padding based on mode
    switch (mode) {
        case 0: {  // Constant padding (fill with zero)
            THINKER_RET_CHECK(API_LIB(memset_i8o8)(dst_temp, fill_data, out_size), "luna_memset_i8o8");
            for (int32_t i = 0; i < h_in; i++) {
                for (int32_t j = 0; j < w_in; j++) {
                    THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(dst_temp + ((i + pads_h_up) * w_out + j + pads_w_left) * c_in, 
                                                src_temp + (i * w_in + j) * c_in, c_in), "luna_memcpy_i8o8");
                }
            }
            break;
        }

        case 1: {  // Replicate padding (fill with last data)
            for (int32_t i = 0; i < h_in; i++) {
                for (int32_t j = 0; j < w_in; j++) {
                    THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(dst_temp + ((i + pads_h_up) * w_out + j + pads_w_left) * c_in, 
                                                src_temp + (i * w_in + j) * c_in, c_in), "luna_memcpy_i8o8");
                }
            }
            if (pads_h_up > 0) {
                for (int32_t i = 0; i < pads_h_up; i++) {
                    THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(dst_temp + (i * w_out + pads_w_left) * c_in, src_temp, w_in * c_in), "luna_memcpy_i8o8");
                }
            }
            if (pads_w_left > 0) {
                for (int32_t i = 0; i < h_in + pads_h_up; i++) {
                    for (int32_t j = 0; j < pads_w_left; j++) {
                        for (int32_t k = 0; k < c_in; k++) {
                            dst_temp[(i * w_out + j) * c_in + k] = dst_temp[(i * w_out + pads_w_left) * c_in + k];
                        }
                    }
                }
            }
            if (pads_w_right > 0) {
                for (int32_t i = 0; i < h_in + pads_h_up; i++) {
                    for (int32_t j = w_in + pads_w_left; j < w_out; j++) {
                        for (int32_t k = 0; k < c_in; k++) {
                            dst_temp[(i * w_out + j) * c_in + k] = dst_temp[(i * w_out + w_in + pads_w_left - 1) * c_in + k];
                        }
                    }
                }
            }
            if (pads_h_down > 0) {
                for (int32_t i = h_in + pads_h_up; i < h_out; i++) {
                    THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(dst_temp + i * w_out * c_in, 
                                                dst_temp + (h_in + pads_h_up - 1) * w_out * c_in, w_out * c_in),
                                                "luna_memcpy_i8o8");
                }
            }
            break;
        }

        case 2: {  // Reflect padding
            for (int32_t i = 0; i < h_in; i++) {
                for (int32_t j = 0; j < w_in; j++) {
                    THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(dst_temp + ((i + pads_h_up) * w_out + j + pads_w_left) * c_in, 
                                                src_temp + (i * w_in + j) * c_in, c_in), "luna_memcpy_i8o8");
                }
            }
            if (pads_h_up > 0) {
                for (int32_t i = 0; i < pads_h_up; i++) {
                    THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(dst_temp + (i * w_out + pads_w_left) * c_in, 
                                                src_temp + (pads_h_up - i) * w_in * c_in, w_in * c_in),
                                                "luna_memcpy_i8o8");
                }
            }
            if (pads_w_left > 0) {
                for (int32_t i = 0; i < h_in + pads_h_up; i++) {
                    for (int32_t j = 0; j < pads_w_left; j++) {
                        for (int32_t k = 0; k < c_in; k++) {
                            dst_temp[(i * w_out + j) * c_in + k] = dst_temp[(i * w_out + 2 * pads_w_left - j) * c_in + k];
                        }
                    }
                }
            }
            if (pads_w_right > 0) {
                for (int32_t i = 0; i < h_in + pads_h_up; i++) {
                    for (int32_t j = w_in + pads_w_left; j < w_out; j++) {
                        for (int32_t k = 0; k < c_in; k++) {
                            dst_temp[(i * w_out + j) * c_in + k] = dst_temp[(i * w_out + 2 * (w_in + pads_w_left) - j - 2) * c_in + k];
                        }
                    }
                }
            }
            if (pads_h_down > 0) {
                for (int32_t i = h_in + pads_h_up; i < h_out; i++) {
                    THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(dst_temp + i * w_out * c_in, 
                                                dst_temp + (2 * (h_in + pads_h_up) - i - 2) * w_out * c_in, w_out * c_in),
                                                "luna_memcpy_i8o8");
                }
            }
            break;
        }

        default:
            memset(dst_temp, fill_data, out_size);
            for (int32_t i = 0; i < h_in; i++) {
                for (int32_t j = 0; j < w_in; j++) {
                    opi_psram_cpy_out(dst_temp + ((i + pads_h_up) * w_out + j + pads_w_left) * c_in, 
                                    src_temp + (i * w_in + j) * c_in, c_in);
                }
            }
            break;
    }
    if (dstInPSRAM) {
        int8_t *dst_temp1 = temp + c_in * h_out * w_out;
        THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(dst_temp, dst_temp1, h_out * w_out, c_in), "luna_split_mat_trans_i8o8");
        opi_psram_cpy_out(dst, dst_temp1, out_size);
    }
    else
        THINKER_RET_CHECK(API_LIB(split_mat_trans_i8o8)(dst_temp, dst, h_out * w_out, c_in), "luna_split_mat_trans_i8o8");
    return T_SUCCESS;
}

#endif  // _PAD_LUNA_H_
