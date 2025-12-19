#ifndef _PAD_LUNA_H_
#define _PAD_LUNA_H_

#include <string.h>
#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "thinker_status.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#include "luna/luna_cnn_tools.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief Integer Quantized Padding operation
 * @param X Input tensor
 * @param P Padding parameters tensor
 * @param data Fill data tensor
 * @param workspace Temporary workspace tensor
 * @param Y Output tensor
 * @param attrs Padding attributes
 * @return int32_t Operation status
 */
int32_t iqpad_luna(tTensor *X, tTensor *P, tTensor *data, tTensor *workspace, tTensor *Y, iqPadAttrs *attrs) 
{
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    if (X->shape_.ndim_ != 4) {
        printf("do not support this type!\n");
        return T_ERR_INVALID_DATA;
    }

    int32_t c_in = X->shape_.dims_[1];
    int32_t h_in = X->shape_.dims_[2];
    int32_t w_in = X->shape_.dims_[3];
    int32_t size = h_in * w_in;

    int32_t c_out = Y->shape_.dims_[1];
    int32_t h_out = Y->shape_.dims_[2];
    int32_t w_out = Y->shape_.dims_[3];
    int32_t out_size = c_in * h_out * w_out;

    int64_t pads[8] = {};
    int8_t pads_h_up, pads_h_down;
    int8_t pads_w_left, pads_w_right;

    // Parse padding parameters
    if (P->shape_.dims_[0] == 4) {
        pads_h_up = *((int64_t *)P->dptr_);
        pads_w_left = *((int64_t *)P->dptr_ + 1);
        pads_h_down = *((int64_t *)P->dptr_ + 2);
        pads_w_right = *((int64_t *)P->dptr_ + 3);
    } else if (P->shape_.dims_[0] == 6) {
        pads_h_up = *((int64_t *)P->dptr_ + 1);
        pads_w_left = *((int64_t *)P->dptr_ + 2);
        pads_h_down = *((int64_t *)P->dptr_ + 4);
        pads_w_right = *((int64_t *)P->dptr_ + 5);
    } else if (P->shape_.dims_[0] == 8) {
        pads_h_up = *((int64_t *)P->dptr_ + 2);
        pads_w_left = *((int64_t *)P->dptr_ + 3);
        pads_h_down = *((int64_t *)P->dptr_ + 6);
        pads_w_right = *((int64_t *)P->dptr_ + 7);
    } else {
        printf("pads error!\n");
        return -1;
    }

    int8_t *src = (int8_t *)X->dptr_;
    int8_t *dst = (int8_t *)Y->dptr_;
    int8_t mode = attrs->mode;

    if (mode < 0 || mode > 2) {
        printf("do not support this mode!\n");
        return -1;
    }

    int8_t fill_data = *((int8_t *)data->dptr_);
    if (fill_data != 0) {
        printf("do not support!\n");
        return -1;
    }

    int8_t *temp = (int8_t *)workspace->dptr_;
    ret = API_LIB(split_mat_trans_i8o8)(src, temp, c_in, h_in * w_in);

    switch (mode) {
        case 0: { // Constant padding
            if (Y->mem_.type_ == 2) {
                ret = API_LIB(memset_i8o8)(dst, fill_data, out_size);
                for (int32_t i = 0; i < h_in; ++i) {
                    for (int32_t j = 0; j < w_in; ++j) {
                        ret = API_LIB(memcpy_i8o8)(dst + ((i + pads_h_up) * w_out + j + pads_w_left) * c_in, 
                                                  temp + (i * w_in + j) * c_in, c_in);
                    }
                }
            } else {
                memset(dst, fill_data, out_size);
                for (int32_t i = 0; i < h_in; ++i) {
                    for (int32_t j = 0; j < w_in; ++j) {
                        opi_psram_cpy_out(dst + ((i + pads_h_up) * w_out + j + pads_w_left) * c_in, 
                                         temp + (i * w_in + j) * c_in, c_in);
                    }
                }
            }
        } break;

        case 1: { // Replicate padding
            if (Y->mem_.type_ == 2) {
                for (int32_t i = 0; i < h_in; ++i) {
                    for (int32_t j = 0; j < w_in; ++j) {
                        ret = API_LIB(memcpy_i8o8)(dst + ((i + pads_h_up) * w_out + j + pads_w_left) * c_in, 
                                                  temp + (i * w_in + j) * c_in, c_in);
                    }
                }
                if (pads_h_up != 0) {
                    for (int32_t i = 0; i < pads_h_up; ++i) {
                        ret = API_LIB(memcpy_i8o8)(dst + (i * w_out + pads_w_left) * c_in, temp, w_in * c_in);
                    }
                }
                if (pads_w_left != 0) {
                    for (int32_t i = 0; i < h_in + pads_h_up; ++i) {
                        for (int32_t j = 0; j < pads_w_left; ++j) {
                            for (int32_t k = 0; k < c_in; ++k) {
                                dst[(i * w_out + j) * c_in + k] = dst[(i * w_out + pads_w_left) * c_in + k];
                            }
                        }
                    }
                }
                if (pads_h_down != 0) {
                    for (int32_t i = h_in + pads_h_up; i < h_out; ++i) {
                        ret = API_LIB(memcpy_i8o8)(dst + i * w_out * c_in, 
                                                  dst + (h_in + pads_h_up - 1) * w_out * c_in, 
                                                  (w_in + pads_w_left) * c_in);
                    }
                }
            } else {
                for (int32_t i = 0; i < h_in; ++i) {
                    for (int32_t j = 0; j < w_in; ++j) {
                        opi_psram_cpy_out(dst + ((i + pads_h_up) * w_out + j + pads_w_left) * c_in, 
                                         temp + (i * w_in + j) * c_in, c_in);
                    }
                }
                if (pads_h_up != 0) {
                    for (int32_t i = 0; i < pads_h_up; ++i) {
                        opi_psram_cpy_out(dst + (i * w_out + pads_w_left) * c_in, temp, w_in * c_in);
                    }
                }
                if (pads_w_left != 0) {
                    for (int32_t i = 0; i < h_in + pads_h_up; ++i) {
                        for (int32_t j = 0; j < pads_w_left; ++j) {
                            for (int32_t k = 0; k < c_in; ++k) {
                                dst[(i * w_out + j) * c_in + k] = dst[(i * w_out + pads_w_left) * c_in + k];
                            }
                        }
                    }
                }
                if (pads_h_down != 0) {
                    for (int32_t i = h_in + pads_h_up; i < h_out; ++i) {
                        opi_psram_cpy_out(dst + i * w_out * c_in, 
                                         dst + (h_in + pads_h_up - 1) * w_out * c_in, 
                                         (w_in + pads_w_left) * c_in);
                    }
                }
            }
            if (pads_w_right != 0) {
                for (int32_t i = 0; i < h_out; ++i) {
                    for (int32_t j = w_in + pads_w_left; j < w_out; ++j) {
                        for (int32_t k = 0; k < c_in; ++k) {
                            dst[(i * w_out + j) * c_in + k] = dst[(i * w_out + w_in + pads_w_left - 1) * c_in + k];
                        }
                    }
                }
            }
        } break;

        case 2: { // Reflect padding
            if (Y->mem_.type_ == 2) {
                for (int32_t i = 0; i < h_in; ++i) {
                    for (int32_t j = 0; j < w_in; ++j) {
                        ret = API_LIB(memcpy_i8o8)(dst + ((i + pads_h_up) * w_out + j + pads_w_left) * c_in, 
                                                  temp + (i * w_in + j) * c_in, c_in);
                    }
                }
                if (pads_h_up != 0) {
                    for (int32_t i = 0; i < pads_h_up; ++i) {
                        ret = API_LIB(memcpy_i8o8)(dst + (i * w_out + pads_w_left) * c_in, 
                                                  temp + (pads_h_up - i) * w_in * c_in, w_in * c_in);
                    }
                }
                if (pads_w_left != 0) {
                    for (int32_t i = 0; i < h_in + pads_h_up; ++i) {
                        for (int32_t j = 0; j < pads_w_left; ++j) {
                            for (int32_t k = 0; k < c_in; ++k) {
                                dst[(i * w_out + j) * c_in + k] = dst[(i * w_out + 2 * pads_w_left - j) * c_in + k];
                            }
                        }
                    }
                }
                if (pads_h_down != 0) {
                    for (int32_t i = h_in + pads_h_up; i < h_out; ++i) {
                        ret = API_LIB(memcpy_i8o8)(dst + i * w_out * c_in, 
                                                  dst + (2 * (h_in + pads_h_up) - i - 2) * w_out * c_in, 
                                                  (w_in + pads_w_left) * c_in);
                    }
                }
            } else {
                for (int32_t i = 0; i < h_in; ++i) {
                    for (int32_t j = 0; j < w_in; ++j) {
                        opi_psram_cpy_out(dst + ((i + pads_h_up) * w_out + j + pads_w_left) * c_in, 
                                         temp + (i * w_in + j) * c_in, c_in);
                    }
                }
                if (pads_h_up != 0) {
                    for (int32_t i = 0; i < pads_h_up; ++i) {
                        opi_psram_cpy_out(dst + (i * w_out + pads_w_left) * c_in, 
                                         temp + (pads_h_up - i) * w_in * c_in, w_in * c_in);
                    }
                }
                if (pads_w_left != 0) {
                    for (int32_t i = 0; i < h_in + pads_h_up; ++i) {
                        for (int32_t j = 0; j < pads_w_left; ++j) {
                            for (int32_t k = 0; k < c_in; ++k) {
                                dst[(i * w_out + j) * c_in + k] = dst[(i * w_out + 2 * pads_w_left - j) * c_in + k];
                            }
                        }
                    }
                }
                if (pads_h_down != 0) {
                    for (int32_t i = h_in + pads_h_up; i < h_out; ++i) {
                        opi_psram_cpy_out(dst + i * w_out * c_in, 
                                         dst + (2 * (h_in + pads_h_up) - i - 2) * w_out * c_in, 
                                         (w_in + pads_w_left) * c_in);
                    }
                }
            }
            if (pads_w_right != 0) {
                for (int32_t i = 0; i < h_out; ++i) {
                    for (int32_t j = w_in + pads_w_left; j < w_out; ++j) {
                        for (int32_t k = 0; k < c_in; ++k) {
                            dst[(i * w_out + j) * c_in + k] = dst[(i * w_out + 2 * (w_in + pads_w_left) - j - 2) * c_in + k];
                        }
                    }
                }
            }
        } break;

        default:
            memset(dst, fill_data, out_size);
            for (int32_t i = 0; i < h_in; ++i) {
                for (int32_t j = 0; j < w_in; ++j) {
                    opi_psram_cpy_out(dst + ((i + pads_h_up) * w_out + j + pads_w_left) * c_in, 
                                     temp + (i * w_in + j) * c_in, c_in);
                }
            }
            break;
    }

    ret = API_LIB(split_mat_trans_i8o8)(dst, dst, h_out * w_out, c_out);
    return ret;
}

#endif