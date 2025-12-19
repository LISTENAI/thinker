#ifndef _SUB_LUNA_H_
#define _SUB_LUNA_H_

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "thinker_status.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief Integer Quantized Subtraction operation
 * @param X1 First input tensor
 * @param X2 Second input tensor
 * @param Temp Temporary workspace tensor
 * @param Y Output tensor
 * @return int32_t Operation status
 */
int32_t iqsub_luna(tTensor *X1, tTensor *X2, tTensor *Temp, tTensor *Y) 
{
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    int32_t x1_q = (int32_t)X1->scale_;
    int32_t x2_q = (int32_t)X2->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    void *src1 = (void *)X1->dptr_;
    void *src2 = (void *)X2->dptr_;
    void *dst = (void *)Y->dptr_;
    size_t size = getTensorSize(X1);

    if (x1_q < y_q || x2_q < y_q) {
        return ret;
    }

    bool x1_psram = (X1->mem_.type_ == 1 || X1->mem_.type_ == 3) && Temp;
    bool x2_psram = (X2->mem_.type_ == 1 || X2->mem_.type_ == 3) && Temp;
    bool y_psram = (Y->mem_.type_ == 1 || Y->mem_.type_ == 3) && Temp;

    if (equalShape(&X1->shape_, &X2->shape_) && X1->dtype_ == X2->dtype_) {
        int32_t shift1 = x1_q - y_q;
        int32_t shift2 = x2_q - y_q;

        switch (X1->dtype_) {
            case Int8: {
                if (x1_psram) {
                    src1 = (int8_t *)Temp->dptr_;
                    ret = API_LIB(memcpy_i8o8)(src1, (void *)X1->dptr_, size * sizeof(int8_t));
                }
                if (x1_q != y_q) {
                    ret = API_LIB(scale_i8i8o8)((const int8_t *)src1, 1, (int8_t *)Temp->dptr_, size, shift1);
                    src1 = (int8_t *)Temp->dptr_;
                }

                if (x2_psram) {
                    src2 = (int8_t *)Temp->dptr_ + (x1_psram || (x1_q != y_q)) * size;
                    ret = API_LIB(memcpy_i8o8)(src2, (void *)X2->dptr_, size * sizeof(int8_t));
                }
                if (x2_q != y_q) {
                    ret = API_LIB(scale_i8i8o8)((const int8_t *)src2, 1, 
                                               (int8_t *)(Temp->dptr_ + (x1_psram || (x1_q != y_q)) * size), 
                                               size, shift2);
                    src2 = (int8_t *)Temp->dptr_ + (x1_psram || (x1_q != y_q)) * size;
                }

                if (y_psram) {
                    dst = (int8_t *)Temp->dptr_;
                }

                ret = API_LIB(sub_i8i8o8)((const int8_t *)dst, (int8_t *)src2, (int8_t *)dst, size, 0);

                if (y_psram) {
                    opi_psram_cpy_out((void *)Y->dptr_, dst, size * sizeof(int8_t));
                }
            } break;
            default:
                break;
        }
    }

    return ret;
}

#endif