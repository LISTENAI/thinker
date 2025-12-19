#ifndef _ADD_LUNA_H_
#define _ADD_LUNA_H_

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
 * @brief Integer quantized addition operation
 * @param X1 Input tensor 1
 * @param X2 Input tensor 2
 * @param Temp Temporary tensor (if needed)
 * @param Y Output tensor
 * @return int32_t Operation status
 */
int32_t iqadd_luna(tTensor *X1, tTensor *X2, tTensor *Temp, tTensor *Y) {
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
        return T_ERR_INVALID_DATA;
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
                // Handle PSRAM to shared memory copy
                if (x1_is_psram) {
                    src1 = y_is_psram ? (int8_t *)Temp->dptr_ : (int8_t *)dst;
                    memcpy(src1, (void *)X1->dptr_, size * sizeof(int8_t));
                }

                // Scale input if needed
                if (x1_q != y_q) {
                    int8_t *out_temp = y_is_psram ? (int8_t *)Temp->dptr_ : (int8_t *)dst;
                    ret = API_LIB(scale_q7_int8)((const q7_t *)src1, 1, out_temp, size, shift1);
                    src1 = (int8_t *)out_temp;
                }

                // Handle PSRAM to shared memory copy for X2
                if (x2_is_psram) {
                    src2 = y_is_psram ? ((int8_t *)Temp->dptr_ + ((x1_is_psram || x1_q != y_q) * size)) : 
                                       ((x1_is_psram || x1_q != y_q) ? (int8_t *)Temp->dptr_ : dst);
                    memcpy(src2, (void *)X2->dptr_, size * sizeof(int8_t));
                }

                // Scale X2 if needed
                if (x2_q != y_q) {
                    int8_t *out_temp = y_is_psram ? ((int8_t *)Temp->dptr_ + ((x1_is_psram || x1_q != y_q) * size)) : 
                                                   ((x1_is_psram || x1_q != y_q) ? (int8_t *)Temp->dptr_ : dst);
                    ret = API_LIB(scale_q7_int8)((const q7_t *)src2, 1, out_temp, size, shift2);
                    src2 = (int8_t *)out_temp;
                }

                // Perform actual addition
                dst = y_is_psram ? (int8_t *)Temp->dptr_ : dst;
                ret = API_LIB(add_q7_int8)((const q7_t *)src1, (q7_t *)src2, (int8_t *)dst, size, 0);

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
    // Handle scalar float addition case
    else if ((X2->dtype_ == Float32) && (X2->shape_.ndim_ == 0)) {
        int32_t shift = x1_q - y_q;
        int8_t src2_tmp = ((float *)src2)[0] * (1 << x1_q);
        ret = API_LIB(offset_q7_int8)((const q7_t *)src1, src2_tmp, dst, size, shift);
    }
    else {
        ret = T_ERR_INVALID_DATATYPE;
    }

    return ret;
}

#endif