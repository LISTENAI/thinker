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

    void *src1 = (void *)X1->dptr_;
    void *src2 = (void *)X2->dptr_;
    void *dst = (void *)Y->dptr_;
    size_t total_size = getTensorSize(X1);

    bool x1_in_psram = (X1->mem_.type_ != 2);
    bool x2_in_psram = (X2->mem_.type_ != 2);
    bool y_in_psram = (Y->mem_.type_ != 2);

    int32_t shift1 = (int32_t)X1->scale_ - (int32_t)Y->scale_;
    int32_t shift2 = (int32_t)X2->scale_ - (int32_t)Y->scale_;

    int32_t past_size = 0;
    switch (X1->dtype_) {
        case Int8: 
        {
            int32_t workspace_size = Temp ? Temp->shape_.dims_[0] : 0;
            int8_t *workspace = Temp ? (int8_t *)Temp->dptr_ : NULL;
            int8_t *dst_temp = y_in_psram ? workspace : (int8_t *)dst;
            if ((x1_in_psram == x2_in_psram) && (shift1 == 0) && (shift2 == 0))
            {
                ret = API_LIB(add_i8i8o8)((const int8_t *)src1, (int8_t *)src2, (int8_t *)dst, total_size, 0);
            }
            else if ((shift1 != 0) && (shift2 == 0) && (!x2_in_psram))
            {
                if (!y_in_psram) {
                    int8_t *src1_temp = dst_temp;
                    ret = API_LIB(scale_i8i8o8)((int8_t *)src1, 1, (int8_t *)src1_temp, total_size, shift1);
                    ret = API_LIB(add_i8i8o8)((const int8_t *)src1_temp, (int8_t *)src2, (int8_t *)dst, total_size, 0);
                }
                else {
                    while (past_size < total_size)
                    {
                        int32_t remain_size = total_size - past_size;
                        int32_t cur_size = workspace_size < remain_size ? workspace_size : remain_size;

                        int8_t *src1_temp = dst_temp;
                        ret = API_LIB(scale_i8i8o8)((int8_t *)src1 + past_size, 1, (int8_t *)src1_temp, cur_size, shift1);

                        int8_t *src2_temp = (int8_t *)src2 + past_size;
                        ret |= API_LIB(add_i8i8o8)((int8_t *)src1_temp, (int8_t *)src2_temp, (int8_t *)dst_temp, cur_size, 0);
                        opi_psram_cpy_out((void *)dst + past_size, dst_temp, cur_size * sizeof(int8_t));
                        past_size += cur_size;
                    }
                }
            }
            else if ((shift1 == 0) && (shift2 != 0) && (!x1_in_psram))
            {
                if (!y_in_psram) {
                    int8_t *src2_temp = dst_temp;
                    ret = API_LIB(scale_i8i8o8)((int8_t *)src2, 1, (int8_t *)src2_temp, total_size, shift2);
                    ret = API_LIB(add_i8i8o8)((const int8_t *)src1, (int8_t *)src2_temp, (int8_t *)dst, total_size, 0);
                }
                else {
                    while (past_size < total_size)
                    {
                        int32_t remain_size = total_size - past_size;
                        int32_t cur_size = workspace_size < remain_size ? workspace_size : remain_size;

                        int8_t *src1_temp = (int8_t *)src1 + past_size;
                        int8_t *src2_temp = dst_temp;
                        ret = API_LIB(scale_i8i8o8)((int8_t *)src2 + past_size, 1, (int8_t *)src2_temp, cur_size, shift1);
                        ret |= API_LIB(add_i8i8o8)((int8_t *)src1_temp, (int8_t *)src2_temp, (int8_t *)dst_temp, cur_size, 0);
                        opi_psram_cpy_out((void *)dst + past_size, dst_temp, cur_size * sizeof(int8_t));
                        past_size += cur_size;
                    }
                }
            }
            else
            {
                int32_t factor = y_in_psram ? 1 : 0;
                while (past_size < total_size)
                {
                    int32_t remain_size = total_size - past_size;
                    int32_t cur_size = (workspace_size >> factor) < remain_size ? (workspace_size >> factor) : remain_size;

                    int8_t *src1_temp = dst_temp;
                    ret = API_LIB(scale_i8i8o8)((int8_t *)src1 + past_size, 1, (int8_t *)src1_temp, cur_size, shift1);
                    int8_t *src2_temp = y_in_psram ? (workspace + cur_size) : workspace;
                    ret |= API_LIB(scale_i8i8o8)((int8_t *)src2 + past_size, 1, (int8_t *)src2_temp, cur_size, shift2);
                    ret |= API_LIB(add_i8i8o8)((int8_t *)src1_temp, (int8_t *)src2_temp, (int8_t *)dst_temp, cur_size, 0);
                    if (y_in_psram)
                        opi_psram_cpy_out((void *)dst + past_size, dst_temp, cur_size * sizeof(int8_t));
                    past_size += cur_size;
                }
            }
            break;
        }

        case Int32: {
            int32_t workspace_size = Temp ? Temp->shape_.dims_[0] >> 2 : 0;
            int32_t *workspace = Temp ? (int32_t *)Temp->dptr_ : NULL;
            int32_t *dst_temp = y_in_psram ? workspace : (int32_t *)dst;
            if ((x1_in_psram == x2_in_psram) && (shift1 == 0) && (shift2 == 0))
            {
                ret = API_LIB(add_i32i32o32)((const int32_t *)src1, (int32_t *)src2, (int32_t *)dst, total_size, 0);
            }
            else if ((shift1 != 0) && (shift2 == 0) && (!x2_in_psram))
            {
                if (!y_in_psram) {
                    int32_t *src1_temp = dst_temp;
                    ret = API_LIB(scale_i32i32o32)((int32_t *)src1, 1, (int32_t *)src1_temp, total_size, shift1);
                    ret = API_LIB(add_i32i32o32)((const int32_t *)src1_temp, (int32_t *)src2, (int32_t *)dst, total_size, 0);
                }
                else {
                    while (past_size < total_size)
                    {
                        int32_t remain_size = total_size - past_size;
                        int32_t cur_size = workspace_size < remain_size ? workspace_size : remain_size;

                        int32_t *src1_temp = dst_temp;
                        ret = API_LIB(scale_i32i32o32)((int32_t *)src1 + past_size, 1, (int32_t *)src1_temp, cur_size, shift1);

                        int32_t *src2_temp = (int32_t *)src2 + past_size;
                        ret |= API_LIB(add_i32i32o32)((int32_t *)src1_temp, (int32_t *)src2_temp, (int32_t *)dst_temp, cur_size, 0);
                        opi_psram_cpy_out((void *)dst + past_size, dst_temp, cur_size * sizeof(int32_t));
                        past_size += cur_size;
                    }
                }
            }
            else if ((shift1 == 0) && (shift2 != 0) && (!x1_in_psram))
            {
                if (!y_in_psram) {
                    int32_t *src2_temp = dst_temp;
                    ret = API_LIB(scale_i32i32o32)((int32_t *)src2, 1, (int32_t *)src2_temp, total_size, shift1);
                    ret = API_LIB(add_i32i32o32)((const int32_t *)src1, (int32_t *)src2_temp, (int32_t *)dst, total_size, 0);
                }
                else {
                    while (past_size < total_size)
                    {
                        int32_t remain_size = total_size - past_size;
                        int32_t cur_size = workspace_size < remain_size ? workspace_size : remain_size;

                        int32_t *src1_temp = (int32_t *)src1 + past_size;
                        int32_t *src2_temp = dst_temp;
                        ret = API_LIB(scale_i32i32o32)((int32_t *)src2 + past_size, 1, (int32_t *)src2_temp, cur_size, shift1);
                        ret |= API_LIB(add_i32i32o32)((int32_t *)src1_temp, (int32_t *)src2_temp, (int32_t *)dst_temp, cur_size, 0);
                        opi_psram_cpy_out((void *)dst + past_size, dst_temp, cur_size * sizeof(int32_t));
                        past_size += cur_size;
                    }
                }
            }
            else
            {
                int32_t factor = y_in_psram ? 1 : 0;
                while (past_size < total_size)
                {
                    int32_t remain_size = total_size - past_size;
                    int32_t cur_size = (workspace_size >> factor) < remain_size ? (workspace_size >> factor) : remain_size;

                    int32_t *src1_temp = dst_temp;
                    ret = API_LIB(scale_i32i32o32)((int32_t *)src1 + past_size, 1, (int32_t *)src1_temp, cur_size, shift1);
                    int32_t *src2_temp = y_in_psram ? (workspace + cur_size) : workspace;
                    ret |= API_LIB(scale_i32i32o32)((int32_t *)src2 + past_size, 1, (int32_t *)src2_temp, cur_size, shift2);
                    ret |= API_LIB(add_i32i32o32)((int32_t *)src1_temp, (int32_t *)src2_temp, (int32_t *)dst_temp, cur_size, 0);
                    if (y_in_psram)
                        opi_psram_cpy_out((void *)dst + past_size, dst_temp, cur_size * sizeof(int32_t));
                    past_size += cur_size;
                }
            }
            break;
        }
        break;

        default:
            return T_ERR_INVALID_DATATYPE;
    }

    return ret;
}

#endif  //_ADD_LUNA_H_