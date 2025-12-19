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
 * @brief Perform element-wise addition on quantized integer tensors
 * @param X1 First input tensor
 * @param X2 Second input tensor
 * @param Temp Temporary workspace buffer
 * @param Y Output tensor
 * @return Operation status
 */
int32_t iqadd_luna(tTensor *X1, tTensor *X2, tTensor *Temp, tTensor *Y) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    int32_t x1_q = (int32_t)X1->scale_;
    int32_t x2_q = (int32_t)X2->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    void *src1 = (void *)X1->dptr_;
    void *src2 = (void *)X2->dptr_;
    void *dst = (void *)Y->dptr_;
    size_t total_size = getTensorSize(X1);

    int32_t x1_is_psram = (2 != X1->mem_.type_) ? 1 : 0;
    int32_t x2_is_psram = (2 != X2->mem_.type_) ? 1 : 0;
    int32_t y_is_psram = (2 != Y->mem_.type_) ? 1 : 0;

    if (equalShape(&X1->shape_, &X2->shape_) && (X1->dtype_ == X2->dtype_)) {
        int32_t shift1 = x1_q - y_q;
        int32_t shift2 = x2_q - y_q;

        switch (X1->dtype_) {
            case Int8: {
                int32_t past_size = 0;
                int32_t workspace_size = Temp ? Temp->shape_.dims_[0] : 0;
                while (past_size < total_size) {
                    int32_t remain_size = total_size - past_size;
                    int32_t cur_size = workspace_size < remain_size ? workspace_size : remain_size;
                    if (!Temp) {
                        cur_size = total_size;
                    }

                    src1 = (int8_t *)X1->dptr_ + past_size;
                    src2 = (int8_t *)X2->dptr_ + past_size;

                    if (x1_q != y_q || x1_is_psram) {
                        int8_t *out_temp = y_is_psram ? (int8_t *)Temp->dptr_ : (int8_t *)dst;
                        int8_t *src1_ptr = x1_is_psram ? (int8_t *)Temp->dptr_ : src1;

                        if (x1_is_psram) {
                            ret = API_LIB(memcpy_i8o8)(src1_ptr, src1, cur_size);
                        }

                        int32_t scale1 = (x1_q > y_q) ? 1 : 1 << (y_q - x1_q);
                        int32_t scale2 = (x1_q > y_q) ? shift1 : 0;

                        ret = API_LIB(scale_i8i8o8)((const int8_t *)src1_ptr, scale1, out_temp, cur_size, scale2);
                        src1 = (int8_t *)out_temp;
                    }

                    if (x2_q != y_q || x2_is_psram) {
                        int8_t *out_temp = y_is_psram ? ((int8_t *)Temp->dptr_ + cur_size) : (int8_t *)dst;
                        int8_t *src2_ptr = x2_is_psram ? ((int8_t *)Temp->dptr_ + cur_size) : src2;

                        if (x2_is_psram) {
                            ret = API_LIB(memcpy_i8o8)(src2_ptr, src2, cur_size);
                        }

                        int32_t scale1 = (x2_q > y_q) ? 1 : 1 << (y_q - x2_q);
                        int32_t scale2 = (x2_q > y_q) ? shift2 : 0;

                        ret = API_LIB(scale_i8i8o8)((const int8_t *)src2_ptr, scale1, out_temp, cur_size, scale2);
                        src2 = (int8_t *)out_temp;
                    }

                    if (Y->dtype_ == Int8) {
                        dst = y_is_psram ? (int8_t *)Temp->dptr_ : dst + past_size;
                        ret = API_LIB(add_i8i8o8)((const int8_t *)src1, (int8_t *)src2, (int8_t *)dst, cur_size, 0);
                        if (y_is_psram) {
                            opi_psram_cpy_out((void *)Y->dptr_ + past_size, dst, cur_size * sizeof(int8_t));
                        }
                    } else if (Y->dtype_ == Int32) {
                        dst = y_is_psram ? (int32_t *)Temp->dptr_ : dst + past_size;
                        ret = API_LIB(add_i8i8o32)((const int8_t *)src1, (int8_t *)src2, (int32_t *)dst, cur_size, 0);
                        if (y_is_psram) {
                            opi_psram_cpy_out((int32_t *)Y->dptr_ + past_size, dst, cur_size * sizeof(int32_t));
                        }
                    } else {
                        return T_ERR_INVALID_DATATYPE;
                    }

                    past_size += cur_size;
                }
            }
            break;

            case Int32: {
                int32_t past_size = 0;
                int32_t workspace_size = Temp ? (Temp->shape_.dims_[0] >> 4) : 0;
                while (past_size < total_size) {
                    int32_t remain_size = total_size - past_size;
                    int32_t cur_size = workspace_size < remain_size ? workspace_size : remain_size;

                    src1 = (int32_t *)X1->dptr_ + past_size;
                    src2 = (int32_t *)X2->dptr_ + past_size;

                    if (x1_q != y_q || x1_is_psram) {
                        int32_t *out_temp = y_is_psram ? (int32_t *)Temp->dptr_ : (int32_t *)dst;
                        int32_t *src1_ptr = x1_is_psram ? (int32_t *)Temp->dptr_ : src1;

                        if (x1_is_psram) {
                            ret = API_LIB(memcpy_i8o8)((int8_t *)src1_ptr, (int8_t *)src1, cur_size * sizeof(int32_t));
                        }

                        int32_t scale1 = (x1_q > y_q) ? 1 : 1 << (y_q - x1_q);
                        int32_t scale2 = (x1_q > y_q) ? shift1 : 0;

                        ret = API_LIB(scale_i32i32o32)(src1_ptr, scale1, out_temp, cur_size, scale2);
                        src1 = out_temp;
                    }

                    if (x2_q != y_q || x2_is_psram) {
                        int32_t *out_temp = y_is_psram ? ((int32_t *)Temp->dptr_ + cur_size) : (int32_t *)dst;
                        int32_t *src2_ptr = x2_is_psram ? ((int32_t *)Temp->dptr_ + cur_size) : src2;

                        if (x2_is_psram) {
                            ret = API_LIB(memcpy_i8o8)((int8_t *)src2_ptr, (int8_t *)src2, cur_size * sizeof(int32_t));
                        }

                        int32_t scale1 = (x2_q > y_q) ? 1 : 1 << (y_q - x2_q);
                        int32_t scale2 = (x2_q > y_q) ? shift2 : 0;
                        ret = API_LIB(scale_i32i32o32)(src2_ptr, scale1, out_temp, cur_size, scale2);
                        src2 = out_temp;
                    }

                    if (Y->dtype_ == Int8) {
                        dst = y_is_psram ? (int8_t *)Temp->dptr_ : dst + past_size;
                        ret = API_LIB(add_i32i32o8)((const int32_t *)src1, (int32_t *)src2, (int8_t *)dst, cur_size, 0);
                        if (y_is_psram) {
                            opi_psram_cpy_out((void *)Y->dptr_ + past_size, dst, cur_size * sizeof(int8_t));
                        }
                    } 
                    else if (Y->dtype_ == Int32) {
                        dst = y_is_psram ? (int32_t *)Temp->dptr_ : dst + past_size;
                        ret = API_LIB(add_i32i32o32)((const int32_t *)src1, (int32_t *)src2, (int32_t *)dst, cur_size, 0);
                        if (y_is_psram) {
                            opi_psram_cpy_out((int32_t *)Y->dptr_ + past_size, dst, cur_size * sizeof(int32_t));
                        }
                    } else {
                        return T_ERR_INVALID_DATATYPE;
                    }

                    past_size += cur_size;
                }
            }
            break;

            default:
                return T_ERR_INVALID_DATATYPE;
        }
    } else {
        return T_ERR_INVALID_DATATYPE;
    }

    return ret;
}

#endif  //_ADD_LUNA_H_