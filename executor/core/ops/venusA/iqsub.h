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
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    // Check if input tensors have the same shape and data type
    if (!equalShape(&X1->shape_, &X2->shape_) || X1->dtype_ != X2->dtype_) {
        return T_ERR_INVALID_DATATYPE;
    }

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

    // Check if quantization scales are valid
    if (x1_q < y_q || x2_q < y_q) {
        return T_ERR_INVALID_PARA;
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
        ret = API_LIB(memcpy_i8o8)(src1, (int8_t *)X1->dptr_, size * sizeof(int8_t));
    }
    if (x1_q != y_q) {
        ret = API_LIB(scale_i8i8o8)(src1, 1, (int8_t *)Temp->dptr_, size, shift1);
        src1 = (int8_t *)Temp->dptr_;
    }

    if (x2InPSram) {
        src2 = (int8_t *)Temp->dptr_ + ((x1InPSram || x1_q != y_q) ? size : 0);
        ret = API_LIB(memcpy_i8o8)(src2, (int8_t *)X2->dptr_, size * sizeof(int8_t));
    }
    if (x2_q != y_q) {
        ret = API_LIB(scale_i8i8o8)(src2, 1, (int8_t *)(Temp->dptr_ + ((x1InPSram || x1_q != y_q) ? size : 0)), size, shift2);
        src2 = (int8_t *)Temp->dptr_ + ((x1InPSram || x1_q != y_q) ? size : 0);
    }

    // Perform subtraction
    if (yInPSram) {
        dst = (int8_t *)Temp->dptr_;
    }
    ret = API_LIB(sub_i8i8o8)(dst, src2, dst, size, 0);

    // Copy result to PSram if necessary
    if (yInPSram) {
        opi_psram_cpy_out((void *)Y->dptr_, dst, size * sizeof(int8_t));
    }

    return ret;
}

#endif  // _SUB_LUNA_H_