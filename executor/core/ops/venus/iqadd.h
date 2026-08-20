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
    #if THINKER_PARAM_CHECK
    if (X1 == NULL || X2 == NULL || Y == NULL ||
                        X1->dptr_ == 0 || X2->dptr_ == 0 || Y->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
    #endif
    int32_t x1_q = (int32_t)X1->scale_;
    int32_t x2_q = (int32_t)X2->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    void *src1 = (void *)X1->dptr_;
    void *src2 = (void *)X2->dptr_;
    void *dst = (void *)Y->dptr_;
    size_t size = getTensorSize(X1);
    #if THINKER_PARAM_CHECK
    if (!equalShape(&X1->shape_, &Y->shape_) ||
                        X1->dtype_ != Int8 || Y->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }
    #endif
    if (X2->dtype_ == Float32 && X2->shape_.ndim_ == 0) {
        #if THINKER_PARAM_CHECK
        if (X1->mem_.type_ != 2 || Y->mem_.type_ != 2) {
            return (T_ERR_NO_SUPPORT_OP);
        }
        #endif
    }
    #if THINKER_PARAM_CHECK
    if ((X1->mem_.type_ != 1 && X1->mem_.type_ != 2) ||
                        (X2->mem_.type_ != 1 && X2->mem_.type_ != 2) ||
                        (Y->mem_.type_ != 1 && Y->mem_.type_ != 2)) {
        return (T_ERR_NO_SUPPORT_OP);
    }
    #endif

    // Check if tensors are in PSRAM
    int32_t x1_in_psram = (2 != X1->mem_.type_);
    int32_t x2_in_psram = (2 != X2->mem_.type_);
    int32_t y_in_psram = (2 != Y->mem_.type_);
    if (x1_in_psram || x2_in_psram || y_in_psram) {
        #if THINKER_RUNTIME_CHECK
        if (Temp == NULL || Temp->dptr_ == 0 ||
                              Temp->mem_.type_ != 2 || Temp->dtype_ != Int8 ||
                              Temp->byte_ != 1 ||
                              getTensorDataSize(Temp) <
                              (size_t)size *
                              ((x1_in_psram || x1_q != y_q) &&
                               (x2_in_psram || x2_q != y_q) && y_in_psram ? 2 : 1)) {
            return (T_ERR_NO_WORKSPACE);
        }
        #endif
    }

    // Handle Int8 data type
    if (equalShape(&X1->shape_, &X2->shape_) && (X1->dtype_ == X2->dtype_)) {
        int32_t shift1 = x1_q - y_q;
        int32_t shift2 = x2_q - y_q;
        #if THINKER_PARAM_CHECK
        if (shift1 < 0 || shift1 > 63 ||
                            shift2 < 0 || shift2 > 63) {
            return (T_ERR_INVALID_PARA);
        }
        #endif

        switch (X1->dtype_) {
            case Int8:
                // Handle PSRAM to shared memory copy
                if (x1_in_psram) {
                    src1 = y_in_psram ? (int8_t *)Temp->dptr_ : (int8_t *)dst;
                    memcpy(src1, (void *)X1->dptr_, size * sizeof(int8_t));
                }

                // Scale input if needed
                if (x1_q != y_q) {
                    int8_t *out_temp = y_in_psram ? (int8_t *)Temp->dptr_ : (int8_t *)dst;
                    THINKER_RET_CHECK(API_LIB(scale_q7_int8)((const q7_t *)src1, 1, out_temp, size, shift1), "luna_scale_q7_int8");
                    src1 = (int8_t *)out_temp;
                }

                // Handle PSRAM to shared memory copy for X2
                if (x2_in_psram) {
                    src2 = y_in_psram ? ((int8_t *)Temp->dptr_ + ((x1_in_psram || x1_q != y_q) * size)) : 
                                       ((x1_in_psram || x1_q != y_q) ? (int8_t *)Temp->dptr_ : dst);
                    memcpy(src2, (void *)X2->dptr_, size * sizeof(int8_t));
                }

                // Scale X2 if needed
                if (x2_q != y_q) {
                    int8_t *out_temp = y_in_psram ? ((int8_t *)Temp->dptr_ + ((x1_in_psram || x1_q != y_q) * size)) : 
                                                   ((x1_in_psram || x1_q != y_q) ? (int8_t *)Temp->dptr_ : dst);
                    THINKER_RET_CHECK(API_LIB(scale_q7_int8)((const q7_t *)src2, 1, out_temp, size, shift2), "luna_scale_q7_int8");
                    src2 = (int8_t *)out_temp;
                }

                // Perform actual addition
                dst = y_in_psram ? (int8_t *)Temp->dptr_ : dst;
                THINKER_RET_CHECK(API_LIB(add_q7_int8)((const q7_t *)src1, (q7_t *)src2, (int8_t *)dst, size, 0), "luna_add_q7_int8");

                // Copy result to output if needed
                if (y_in_psram) {
                    memcpy((void *)Y->dptr_, dst, size * sizeof(int8_t));
                }
                break;
            default:
                return T_ERR_INVALID_DATATYPE;
        }
    }
    // Handle scalar float addition case
    else if ((X2->dtype_ == Float32) && (X2->shape_.ndim_ == 0)) {
        int32_t shift = x1_q - y_q;
        #if THINKER_PARAM_CHECK
        if (shift < 0 || shift > 63 || x1_q < 0 || x1_q > 30) {
            return (T_ERR_INVALID_PARA);
        }
        #endif
        int8_t src2_tmp = floor(ldexpf(((float *)src2)[0], x1_q) + 0.5f);
        THINKER_RET_CHECK(API_LIB(offset_q7_int8)((const q7_t *)src1, src2_tmp, dst, size, shift), "luna_offset_q7_int8");
    }
    else {
        #if THINKER_PARAM_CHECK
        if (1) {
            return (T_ERR_INVALID_DATATYPE);
        }
        #endif
    }

    return T_SUCCESS;
}

#endif
