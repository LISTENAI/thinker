#ifndef _IQPAD_LUNA_H_
#define _IQPAD_LUNA_H_

#include <string.h>
#include "c_api/thinker_define.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "thinker_status.h"

int32_t iqpad_luna(tTensor *X, tTensor *P, tTensor *data, tTensor *workspace,
                   tTensor *Y, iqPadAttrs *attrs) {
    (void)workspace;
    #if THINKER_PARAM_CHECK
    if (X == NULL || P == NULL || data == NULL || Y == NULL || attrs == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if (X->dptr_ == 0 || P->dptr_ == 0 || data->dptr_ == 0 ||
                        Y->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    int32_t ndim = X->shape_.ndim_;
    #if THINKER_PARAM_CHECK
    if ((ndim != 3 && ndim != 4) || Y->shape_.ndim_ != ndim ||
                        (ndim == 4 && (X->shape_.dims_[0] != 1 || Y->shape_.dims_[0] != 1))) {
        return (T_ERR_INVALID_DATA);
    }

    if (X->dtype_ != Int8 || Y->dtype_ != Int8 || P->dtype_ != Int64 ||
                        data->dtype_ != Int8 || getTensorSize(data) != 1) {
        return (T_ERR_INVALID_DATATYPE);
    }
#endif

    int32_t pad_count = P->shape_.dims_[0];
    #if THINKER_PARAM_CHECK
    if (P->shape_.ndim_ != 1 ||
                        (pad_count != 4 && pad_count != 6 && pad_count != 8)) {
        return (T_ERR_INVALID_DATA);
    }
#endif
    int64_t *pads = (int64_t *)P->dptr_;
    int32_t top = pad_count == 4 ? pads[0] : pad_count == 6 ? pads[1] : pads[2];
    int32_t left = pad_count == 4 ? pads[1] : pad_count == 6 ? pads[2] : pads[3];
    int32_t bottom = pad_count == 4 ? pads[2] : pad_count == 6 ? pads[4] : pads[6];
    int32_t right = pad_count == 4 ? pads[3] : pad_count == 6 ? pads[5] : pads[7];
    if (pad_count == 6) {
        #if THINKER_PARAM_CHECK
        if (pads[0] != 0 || pads[3] != 0) {
            return (T_ERR_INVALID_DATA);
        }
#endif
    } else if (pad_count == 8) {
        #if THINKER_PARAM_CHECK
        if (pads[0] != 0 || pads[1] != 0 || pads[4] != 0 || pads[5] != 0) {
            return (T_ERR_INVALID_DATA);
        }
#endif
    }
    #if THINKER_PARAM_CHECK
    if (top < 0 || bottom < 0 || left < 0 || right < 0) {
        return (T_ERR_INVALID_DATA);
    }
#endif

    int32_t offset = ndim - 3;
    int32_t c = X->shape_.dims_[offset];
    int32_t h = X->shape_.dims_[offset + 1];
    int32_t w = X->shape_.dims_[offset + 2];
    int32_t out_h = Y->shape_.dims_[offset + 1];
    int32_t out_w = Y->shape_.dims_[offset + 2];
    #if THINKER_PARAM_CHECK
    if (Y->shape_.dims_[offset] != c || out_h != h + top + bottom ||
                        out_w != w + left + right) {
        return (T_ERR_INVALID_DATA);
    }

    if (attrs->mode < 0 || attrs->mode > 2) {
        return (T_ERR_INVALID_PARA);
    }

    if (attrs->mode == 2 &&
                        (top >= h || bottom >= h || left >= w || right >= w)) {
        return (T_ERR_INVALID_DATA);
    }
#endif

    int8_t fill = *(int8_t *)data->dptr_;
    #if THINKER_PARAM_CHECK
    if (attrs->mode != 0 && fill != 0) {
        return (T_ERR_INVALID_DATA);
    }
#endif
    int8_t *src = (int8_t *)X->dptr_;
    int8_t *dst = (int8_t *)Y->dptr_;
    for (int32_t ch = 0; ch < c; ++ch) {
        for (int32_t oh = 0; oh < out_h; ++oh) {
            int32_t ih = oh - top;
            for (int32_t ow = 0; ow < out_w; ++ow) {
                int32_t iw = ow - left;
                int32_t src_h = ih;
                int32_t src_w = iw;
                int8_t value = fill;
                if (attrs->mode == 1) {
                    src_h = src_h < 0 ? 0 : src_h >= h ? h - 1 : src_h;
                    src_w = src_w < 0 ? 0 : src_w >= w ? w - 1 : src_w;
                } else if (attrs->mode == 2) {
                    src_h = src_h < 0 ? -src_h : src_h >= h ? 2 * h - src_h - 2 : src_h;
                    src_w = src_w < 0 ? -src_w : src_w >= w ? 2 * w - src_w - 2 : src_w;
                }
                if ((uint32_t)src_h < (uint32_t)h && (uint32_t)src_w < (uint32_t)w) {
                    value = src[(ch * h + src_h) * w + src_w];
                }
                dst[(ch * out_h + oh) * out_w + ow] = value;
            }
        }
    }
    if (Y->mem_.type_ == 1) {
        thinker_psram_write_complete((void *)Y->dptr_, getTensorDataSize(Y));
    }
    return T_SUCCESS;
}

#endif
