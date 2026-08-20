#ifndef _DIV_LUNA_H_
#define _DIV_LUNA_H_

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif
#include "thinker_status.h"

static int32_t iqdiv_has_zero_i32(const int32_t *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (data[i] == 0) {
            return 1;
        }
    }
    return 0;
}


/**
 * @brief Quantized division operation implementation
 * @param lhs Left-hand side tensor
 * @param rhs Right-hand side tensor
 * @param Y Output tensor
 * @return int32_t Operation status
 */
int32_t iqdiv_luna(tTensor *lhs, tTensor *rhs, tTensor *Y) {
    #if THINKER_PARAM_CHECK
    if (lhs == NULL || rhs == NULL || Y == NULL ||
                        lhs->dptr_ == 0 || rhs->dptr_ == 0 || Y->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }

    if (lhs->dtype_ != Int32 || rhs->dtype_ != Int32 ||
                        Y->dtype_ != Int32) {
        return (T_ERR_INVALID_DATATYPE);
    }
#endif
    #if THINKER_RUNTIME_CHECK
    if (lhs->mem_.type_ != 2 || rhs->mem_.type_ != 2 ||
                          Y->mem_.type_ != 2) {
        return (T_ERR_NO_SUPPORT_OP);
    }

    if (!equalShape(&lhs->shape_, &Y->shape_) ||
                        (rhs->shape_.ndim_ != 0 &&
                         !equalShape(&lhs->shape_, &rhs->shape_)) ||
                        lhs->zero_ != 0 || rhs->zero_ != 0 || Y->zero_ != 0) {
        return (T_ERR_INVALID_DATA);
    }
#endif

    size_t size = getTensorSize(lhs);
    size_t rhs_size = getTensorSize(rhs);

    // Calculate quantization shift
    int32_t lhs_scale = (int32_t)lhs->scale_;
    int32_t rhs_scale = (int32_t)rhs->scale_;
    int32_t output_scale = (int32_t)Y->scale_;
    int32_t shift = output_scale - (lhs_scale - rhs_scale);
#if THINKER_PARAM_CHECK
if (shift < 0 || shift > 63) {
    return (T_ERR_INVALID_PARA);
}
#endif

    // Check if right-hand side is a scalar
    if (rhs->shape_.ndim_ == 0) {
        int32_t scalar = (int32_t)(*(int32_t *)rhs->dptr_);
        #if THINKER_PARAM_CHECK
        if (scalar <= 0 || (scalar & (scalar - 1)) != 0) {
            return (T_ERR_INVALID_PARA);
        }
#endif
        return luna_div_scalar_i32i32o32((const int32_t *)lhs->dptr_, scalar, (int32_t *)Y->dptr_, size, shift);
    } 
    else {
#if THINKER_PARAM_CHECK
if (rhs_size != size || lhs->mem_.type_ != rhs->mem_.type_) {
    return (T_ERR_INVALID_PARA);
}
#endif
        const int32_t *rhs_data = (const int32_t *)rhs->dptr_;
        #if THINKER_PARAM_CHECK
        if (iqdiv_has_zero_i32(rhs_data, rhs_size)) {
            return (T_ERR_INVALID_PARA);
        }
#endif
        return API_LIB(div_i32i32o32)((const int32_t *)lhs->dptr_, rhs_data, (int32_t *)Y->dptr_, size, shift);
    }

    return T_SUCCESS;
}

#endif  // _DIV_LUNA_H_
