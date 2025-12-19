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
 * @brief Integer quantized subtraction operation
 * @param X1 First input tensor
 * @param X2 Second input tensor
 * @param Temp Temporary tensor (if needed)
 * @param Y Output tensor
 * @return int32_t Operation status
 */
int32_t iqsub_luna(tTensor *X1, tTensor *X2, tTensor *Temp, tTensor *Y) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    int32_t x1_q = (int32_t)X1->scale_;
    int32_t x2_q = (int32_t)X2->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    void *src1 = (void *)X1->dptr_;
    void *src2 = (void *)X2->dptr_;
    void *dst = (void *)Y->dptr_;
    size_t size = getTensorSize(X1);

    // Check if input scales are valid relative to output scale
    if ((x1_q < y_q) || x2_q < y_q) {
        return ret;
    }

    // Check if tensors are in PSRAM
    int32_t x1_is_psram = ((1 == X1->mem_.type_ || 3 == X1->mem_.type_) && Temp);
    int32_t x2_is_psram = ((1 == X2->mem_.type_ || 3 == X2->mem_.type_) && Temp);
    int32_t y_is_psram = ((1 == Y->mem_.type_ || 3 == Y->mem_.type_) && Temp);

    // Handle Int8 data type
    if (equalShape(&X1->shape_, &X2->shape_) && (X1->dtype_ == X2->dtype_)) {
        int32_t shift1 = x1_q - y_q;
        int32_t shift2 = x2_q - y_q;

        switch (X1->dtype_) {
            case Int8:
                // Handle PSRAM to shared memory copy for X1
                if (x1_is_psram) {
                    src1 = (int8_t *)Temp->dptr_;
                    memcpy(src1, (void *)X1->dptr_, size * sizeof(int8_t));
                }

                // Scale X1 if needed
                if (x1_q != y_q) {
                    ret = API_LIB(scale_q7_int8)((const q7_t *)src1, 1, (int8_t *)Temp->dptr_, size, shift1);
                    src1 = (int8_t *)Temp->dptr_;
                }

                // Handle PSRAM to shared memory copy for X2
                if (x2_is_psram) {
                    src2 = (int8_t *)Temp->dptr_ + ((x1_is_psram || x1_q != y_q) * size);
                    memcpy(src2, (void *)X2->dptr_, size * sizeof(int8_t));
                }

                // Scale X2 if needed
                if (x2_q != y_q) {
                    ret = API_LIB(scale_q7_int8)((const q7_t *)src2, 1, 
                                                (int8_t *)(Temp->dptr_ + ((x1_is_psram || x1_q != y_q) * size)), 
                                                size, shift2);
                    src2 = (int8_t *)Temp->dptr_ + ((x1_is_psram || x1_q != y_q) * size);
                }

                // Set destination pointer
                if (y_is_psram) {
                    dst = (int8_t *)Temp->dptr_;
                }

                // Perform subtraction
                ret = API_LIB(sub_q7_int8)((const q7_t *)dst, (q7_t *)src2, (int8_t *)dst, size, 0);

                // Copy result to output if needed
                if (y_is_psram) {
                    memcpy((void *)Y->dptr_, dst, size * sizeof(int8_t));
                }
                break;
            default:
                ret = T_ERR_INVALID_DATATYPE;
                break;
        }
    }

    return ret;
}

#endif