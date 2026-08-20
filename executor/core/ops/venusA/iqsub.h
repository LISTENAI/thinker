#ifndef _SUB_LUNA_H_
#define _SUB_LUNA_H_

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

/**
 * @brief Quantized subtraction operation implementation
 * @param X1 First input tensor
 * @param X2 Second input tensor
 * @param Temp Temporary workspace tensor
 * @param Y Output tensor
 * @return int32_t Operation status
 */
int32_t iqsub_luna(tTensor *X1, tTensor *X2, tTensor *Temp, tTensor *Y) {
#if THINKER_PARAM_CHECK
if (!equalShape(&X1->shape_, &X2->shape_) || X1->dtype_ != X2->dtype_ ||
        X1->dtype_ != Y->dtype_ || X1->dtype_ != Int8) {
    return (T_ERR_INVALID_DATATYPE);
}
#endif

    // Quantization scales
    int32_t x1_q = (int32_t)X1->scale_;
    int32_t x2_q = (int32_t)X2->scale_;
    int32_t y_q = (int32_t)Y->scale_;

    // Memory type checks
    int32_t x1InPSram = (X1->mem_.type_ == 1 || X1->mem_.type_ == 3) ? 1 : 0;
    int32_t x2InPSram = (X2->mem_.type_ == 1 || X2->mem_.type_ == 3) ? 1 : 0;
    int32_t yInPSram = (Y->mem_.type_ == 1 || Y->mem_.type_ == 3) ? 1 : 0;

    // Total data size
    size_t size = getTensorSize(X1);

#if THINKER_PARAM_CHECK
if (x1_q < y_q || x2_q < y_q || (x1_q - y_q) > 63 || (x2_q - y_q) > 63) {
    return (T_ERR_INVALID_PARA);
}
#endif

    int32_t x1_need_workspace = x1InPSram || (x1_q != y_q);
    int32_t x2_need_workspace = x2InPSram || (x2_q != y_q);
    int32_t required_workspace = 0;
    if (x1_need_workspace) {
        required_workspace += ALIGN4(size);
    }
    if (x2_need_workspace) {
        required_workspace += ALIGN4(size);
    }
    if (yInPSram && required_workspace < (int32_t)size) {
        required_workspace = ALIGN4(size);
    }
    if (required_workspace > 0 &&
        (Temp == NULL || Temp->shape_.dims_[0] < required_workspace)) {
        return T_ERR_NO_WORKSPACE;
    }

    // Pointers to tensor data
    int8_t *src1 = (int8_t *)X1->dptr_;
    int8_t *src2 = (int8_t *)X2->dptr_;
    int8_t *dst = (int8_t *)Y->dptr_;

    // Quantization shifts
    int32_t shift1 = x1_q - y_q;
    int32_t shift2 = x2_q - y_q;

    // Process each tensor based on memory type and quantization
    if (x1InPSram) {
        src1 = (int8_t *)Temp->dptr_;
        THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(src1, (int8_t *)X1->dptr_, size * sizeof(int8_t)), "luna_memcpy_i8o8");
    }
    if (x1_q != y_q) {
        THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(src1, 1, (int8_t *)Temp->dptr_, size, shift1), "luna_scale_i8i8o8");
        src1 = (int8_t *)Temp->dptr_;
    }

    if (x2InPSram) {
        src2 = (int8_t *)Temp->dptr_ + ((x1InPSram || x1_q != y_q) ? ALIGN4(size) : 0);
        THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(src2, (int8_t *)X2->dptr_, size * sizeof(int8_t)), "luna_memcpy_i8o8");
    }
    if (x2_q != y_q) {
        THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(src2, 1, (int8_t *)((uint8_t *)Temp->dptr_ + ((x1InPSram || x1_q != y_q) ? ALIGN4(size) : 0)), size, shift2), "luna_scale_i8i8o8");
        src2 = (int8_t *)Temp->dptr_ + ((x1InPSram || x1_q != y_q) ? ALIGN4(size) : 0);
    }

    // Perform subtraction
    if (yInPSram) {
        dst = (int8_t *)Temp->dptr_;
    }
    THINKER_RET_CHECK(API_LIB(sub_i8i8o8)(src1, src2, dst, size, 0), "luna_sub_i8i8o8");

    // Copy result to PSram if necessary
    if (yInPSram) {
        opi_psram_cpy_out((void *)Y->dptr_, dst, size * sizeof(int8_t));
    }

    return T_SUCCESS;
}

#endif  // _SUB_LUNA_H_
