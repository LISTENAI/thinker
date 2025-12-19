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
#define API_LIB(api) luna_##api
#endif
#include "thinker_status.h"

/**
 * @brief Integer quantized padding operation
 * @param X Input tensor
 * @param P Padding parameters tensor
 * @param data Fill data tensor
 * @param workspace Workspace tensor
 * @param Y Output tensor
 * @param attrs Padding attributes
 * @return int32_t Operation status
 */
int32_t iqpad_luna(tTensor *X, tTensor *P, tTensor *data, tTensor *workspace, tTensor *Y, iqPadAttrs *attrs) {
    // Check if input tensor is 4-dimensional
    if (X->shape_.ndim_ != 4) {
        printf("Only 4D tensors are supported!");
        return T_ERR_INVALID_DATA;
    }

    int32_t ret = T_SUCCESS;
    int32_t c_in = X->shape_.dims_[1];
    int32_t h_in = X->shape_.dims_[2];
    int32_t w_in = X->shape_.dims_[3];
    int32_t c_out = Y->shape_.dims_[1];
    int32_t h_out = Y->shape_.dims_[2];
    int32_t w_out = Y->shape_.dims_[3];
    int32_t size = h_in * w_in;
    int32_t out_size = c_in * h_out * w_out;

    int64_t pads[8] = {};
    int8_t pads_h_up, pads_h_down;
    int8_t pads_w_left, pads_w_right;

    // Parse padding parameters based on the length of padding tensor
    if (P->shape_.dims_[0] == 4) {
        pads[0] = *((int64_t *)P->dptr_);
        pads[1] = *((int64_t *)P->dptr_ + 1);
        pads[2] = *((int64_t *)P->dptr_ + 2);
        pads[3] = *((int64_t *)P->dptr_ + 3);
        pads_h_up = pads[0];
        pads_w_left = pads[1];
        pads_h_down = pads[2];
        pads_w_right = pads[3];
    } else if (P->shape_.dims_[0] == 6) {
        pads[0] = *((int64_t *)P->dptr_);
        pads[1] = *((int64_t *)P->dptr_ + 1);
        pads[2] = *((int64_t *)P->dptr_ + 2);
        pads[3] = *((int64_t *)P->dptr_ + 3);
        pads[4] = *((int64_t *)P->dptr_ + 4);
        pads[5] = *((int64_t *)P->dptr_ + 5);
        pads_h_up = pads[1];
        pads_w_left = pads[2];
        pads_h_down = pads[4];
        pads_w_right = pads[5];
        if (pads[0] != 0 || pads[3] != 0) {
            printf("Unsupported padding configuration!");
            return T_ERR_INVALID_DATA;
        }
    } else if (P->shape_.dims_[0] == 8) {
        pads[0] = *((int64_t *)P->dptr_);
        pads[1] = *((int64_t *)P->dptr_ + 1);
        pads[2] = *((int64_t *)P->dptr_ + 2);
        pads[3] = *((int64_t *)P->dptr_ + 3);
        pads[4] = *((int64_t *)P->dptr_ + 4);
        pads[5] = *((int64_t *)P->dptr_ + 5);
        pads[6] = *((int64_t *)P->dptr_ + 6);
        pads[7] = *((int64_t *)P->dptr_ + 7);
        pads_h_up = pads[2];
        pads_w_left = pads[3];
        pads_h_down = pads[6];
        pads_w_right = pads[7];
        if (pads[0] != 0 || pads[1] != 0 || pads[4] != 0 || pads[5] != 0) {
            printf("Unsupported padding configuration!");
            return T_ERR_INVALID_DATA;
        }
    } else {
        printf("Invalid padding parameters!");
        return T_ERR_INVALID_DATA;
    }

    int8_t *src = (int8_t *)X->dptr_;
    int8_t *dst = (int8_t *)Y->dptr_;
    int8_t mode = attrs->mode;

    // Validate padding mode
    if (mode < 0 || mode > 2) {
        printf("Invalid padding mode!");
        return T_ERR_INVALID_DATA;
    }

    int8_t fill_data = *((int8_t *)data->dptr_);
    if (fill_data != 0) {
        printf("Only zero padding is supported!");
        return T_ERR_INVALID_DATA;
    }

    int8_t *temp = (int8_t *)workspace->dptr_;
    API_LIB(mat_trans_q7)(src, temp, c_in, h_in * w_in);

    switch (mode) {
        case 0: {  // Constant padding (zero)
            memset(dst, fill_data, out_size);
            for (int32_t i = 0; i < h_in; ++i) {
                for (int32_t j = 0; j < w_in; ++j) {
                    memcpy(dst + ((i + pads_h_up) * w_out + j + pads_w_left) * c_in, 
                           temp + (i * w_in + j) * c_in, c_in);
                }
            }
            break;
        }
        case 1: {  // Replicate last data
            for (int32_t i = 0; i < h_in; ++i) {
                for (int32_t j = 0; j < w_in; ++j) {
                    memcpy(dst + ((i + pads_h_up) * w_out + j + pads_w_left) * c_in, 
                           temp + (i * w_in + j) * c_in, c_in);
                }
            }
            if (pads_h_up > 0) {
                for (int32_t i = 0; i < pads_h_up; ++i) {
                    memcpy(dst + (i * w_out + pads_w_left) * c_in, temp, w_in * c_in);
                }
            }
            if (pads_w_left > 0) {
                for (int32_t i = 0; i < h_in + pads_h_up; ++i) {
                    for (int32_t j = 0; j < pads_w_left; ++j) {
                        for (int32_t k = 0; k < c_in; ++k) {
                            dst[(i * w_out + j) * c_in + k] = dst[(i * w_out + pads_w_left) * c_in + k];
                        }
                    }
                }
            }
            if (pads_h_down > 0) {
                for (int32_t i = h_in + pads_h_up; i < h_out; ++i) {
                    memcpy(dst + i * w_out * c_in, 
                           dst + (h_in + pads_h_up - 1) * w_out * c_in, 
                           (w_in + pads_w_left) * c_in);
                }
            }
            if (pads_w_right > 0) {
                for (int32_t i = 0; i < h_out; ++i) {
                    for (int32_t j = w_in + pads_w_left; j < w_out; ++j) {
                        for (int32_t k = 0; k < c_in; ++k) {
                            dst[(i * w_out + j) * c_in + k] = dst[(i * w_out + w_in + pads_w_left - 1) * c_in + k];
                        }
                    }
                }
            }
            break;
        }
        case 2: {  // Reflect padding
            for (int32_t i = 0; i < h_in; ++i) {
                for (int32_t j = 0; j < w_in; ++j) {
                    memcpy(dst + ((i + pads_h_up) * w_out + j + pads_w_left) * c_in, 
                           temp + (i * w_in + j) * c_in, c_in);
                }
            }
            if (pads_h_up > 0) {
                for (int32_t i = 0; i < pads_h_up; ++i) {
                    memcpy(dst + (i * w_out + pads_w_left) * c_in, 
                           temp + (pads_h_up - i) * w_in * c_in, w_in * c_in);
                }
            }
            if (pads_w_left > 0) {
                for (int32_t i = 0; i < h_in + pads_h_up; ++i) {
                    for (int32_t j = 0; j < pads_w_left; ++j) {
                        for (int32_t k = 0; k < c_in; ++k) {
                            dst[(i * w_out + j) * c_in + k] = dst[(i * w_out + 2 * pads_w_left - j) * c_in + k];
                        }
                    }
                }
            }
            if (pads_h_down > 0) {
                for (int32_t i = h_in + pads_h_up; i < h_out; ++i) {
                    memcpy(dst + i * w_out * c_in, 
                           dst + (2 * (h_in + pads_h_up) - i - 2) * w_out * c_in, 
                           (w_in + pads_w_left) * c_in);
                }
            }
            if (pads_w_right > 0) {
                for (int32_t i = 0; i < h_out; ++i) {
                    for (int32_t j = w_in + pads_w_left; j < w_out; ++j) {
                        for (int32_t k = 0; k < c_in; ++k) {
                            dst[(i * w_out + j) * c_in + k] = dst[(i * w_out + 2 * (w_in + pads_w_left) - j - 2) * c_in + k];
                        }
                    }
                }
            }
            break;
        }
        default: {
            memset(dst, fill_data, out_size);
            for (int32_t i = 0; i < h_in; ++i) {
                for (int32_t j = 0; j < w_in; ++j) {
                    memcpy(dst + ((i + pads_h_up) * w_out + j + pads_w_left) * c_in, 
                           temp + (i * w_in + j) * c_in, c_in);
                }
            }
            break;
        }
    }

    API_LIB(mat_trans_q7)(dst, dst, h_out * w_out, c_out);
    return ret;
}

#endif