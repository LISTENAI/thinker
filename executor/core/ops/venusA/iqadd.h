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
 * @brief Quantized tensor addition operation
 * @param X1 Input tensor 1
 * @param X2 Input tensor 2
 * @param Temp Workspace tensor
 * @param Y Output tensor
 * @return int32_t Operation status
 */
int32_t iqadd_luna(tTensor *X1, tTensor *X2, tTensor *Temp, tTensor *Y) {
    // Check if tensors have the same shape and data type
    if (!equalShape(&X1->shape_, &X2->shape_) || 
        X1->dtype_ != X2->dtype_ || 
        X1->dtype_ != Y->dtype_)
    {
        return T_ERR_INVALID_DATATYPE;
    }

    // Get tensor size and data pointers
    void *src1 = (void *)X1->dptr_;
    void *src2 = (void *)X2->dptr_;
    void *dst = (void *)Y->dptr_;
    size_t total_size = getTensorSize(X1);

    // Determine memory types
    bool x1_in_psram = (X1->mem_.type_ != 2);
    bool x2_in_psram = (X2->mem_.type_ != 2);
    bool y_in_psram = (Y->mem_.type_ != 2);

    // Calculate shifts
    int32_t shift1 = (int32_t)X1->scale_ - (int32_t)Y->scale_;
    int32_t shift2 = (int32_t)X2->scale_ - (int32_t)Y->scale_;

    int32_t past_size = 0;
    switch (X1->dtype_)
    {
        case Int8:
        {
            int32_t workspace_size = Temp ? Temp->shape_.dims_[0] : 0;
            int8_t *workspace = Temp ? (int8_t *)Temp->dptr_ : NULL;
            int8_t *dst_temp = y_in_psram ? workspace : (int8_t *)dst;
            if ((x1_in_psram == x2_in_psram) && (shift1 == 0) && (shift2 == 0))
            {   
                if (!y_in_psram) {
                    THINKER_RET_CHECK(API_LIB(add_i8i8o8)((const int8_t *)src1, (int8_t *)src2, (int8_t *)dst_temp, total_size, 0), "luna_add_i8i8o8");
                }
                else {
                    while (past_size < total_size) {
                        int32_t remain_size = total_size - past_size;
                        int32_t cur_size = workspace_size < remain_size ? workspace_size : remain_size;
                        THINKER_RET_CHECK(API_LIB(add_i8i8o8)((const int8_t *)src1 + past_size, (int8_t *)src2 + past_size, (int8_t *)dst_temp, cur_size, 0), "luna_add_i8i8o8");
                        opi_psram_cpy_out((void *)dst + past_size, dst_temp, cur_size * sizeof(int8_t));
                        past_size += cur_size;
                    }
                }
            }
            else if ((shift2 == 0) && (!x2_in_psram))
            {
                uint32_t shift1_0 = shift1 < 0 ? 1UL << -shift1 : 1;
                uint32_t shift1_1 = shift1 < 0 ? 0 : shift1;
                if (!y_in_psram) {
                    int8_t *src1_temp = dst_temp;
                    THINKER_RET_CHECK(API_LIB(scale_i8i8o8)((int8_t *)src1, shift1_0, (int8_t *)src1_temp, total_size, shift1_1), "luna_scale_i8i8o8");
                    THINKER_RET_CHECK(API_LIB(add_i8i8o8)((const int8_t *)src1_temp, (int8_t *)src2, (int8_t *)dst, total_size, 0), "luna_add_i8i8o8");
                }
                else {
                    while (past_size < total_size) {
                        int32_t remain_size = total_size - past_size;
                        int32_t cur_size = workspace_size < remain_size ? workspace_size : remain_size;

                        int8_t *src1_temp = dst_temp;
                        THINKER_RET_CHECK(API_LIB(scale_i8i8o8)((int8_t *)src1 + past_size, shift1_0, (int8_t *)src1_temp, cur_size, shift1_1), "luna_scale_i8i8o8");

                        int8_t *src2_temp = (int8_t *)src2 + past_size;
                        THINKER_RET_CHECK(API_LIB(add_i8i8o8)((int8_t *)src1_temp, (int8_t *)src2_temp, (int8_t *)dst_temp, cur_size, 0), "luna_add_i8i8o8");
                        opi_psram_cpy_out((void *)dst + past_size, dst_temp, cur_size * sizeof(int8_t));
                        past_size += cur_size;
                    }
                }
            }
            else if ((shift1 == 0) && (!x1_in_psram))
            {
                uint32_t shift2_0 = shift2 < 0 ? 1UL << -shift2 : 1;
                uint32_t shift2_1 = shift2 < 0 ? 0 : shift2;
                if (!y_in_psram) {
                    int8_t *src2_temp = dst_temp;
                    THINKER_RET_CHECK(API_LIB(scale_i8i8o8)((int8_t *)src2, shift2_0, (int8_t *)src2_temp, total_size, shift2_1), "luna_scale_i8i8o8");
                    THINKER_RET_CHECK(API_LIB(add_i8i8o8)((const int8_t *)src1, (int8_t *)src2_temp, (int8_t *)dst, total_size, 0), "luna_scale_add_i8i8o8");
                }
                else {
                    while (past_size < total_size)
                    {
                        int32_t remain_size = total_size - past_size;
                        int32_t cur_size = workspace_size < remain_size ? workspace_size : remain_size;

                        int8_t *src1_temp = (int8_t *)src1 + past_size;
                        int8_t *src2_temp = dst_temp;
                        uint32_t shift1_0 = shift1 < 0 ? 1UL << -shift1 : 1;
                        uint32_t shift1_1 = shift1 < 0 ? 0 : shift1;
                        THINKER_RET_CHECK(API_LIB(scale_i8i8o8)((int8_t *)src2 + past_size, shift2_0, (int8_t *)src2_temp, cur_size, shift2_1), "luna_scale_i8i8o8");
                        THINKER_RET_CHECK(API_LIB(add_i8i8o8)((int8_t *)src1_temp, (int8_t *)src2_temp, (int8_t *)dst_temp, cur_size, 0), "luna_add_i8i8o8");
                        opi_psram_cpy_out((void *)dst + past_size, dst_temp, cur_size * sizeof(int8_t));
                        past_size += cur_size;
                    }
                }
            }
            else
            {
                uint32_t shift1_0 = shift1 < 0 ? 1UL << -shift1 : 1;
                uint32_t shift1_1 = shift1 < 0 ? 0 : shift1;
                uint32_t shift2_0 = shift2 < 0 ? 1UL << -shift2 : 1;
                uint32_t shift2_1 = shift2 < 0 ? 0 : shift2;
                int32_t factor = y_in_psram ? 1 : 0;
                while (past_size < total_size)
                {
                    int32_t remain_size = total_size - past_size;
                    int32_t cur_size = (workspace_size >> factor) < remain_size ? (workspace_size >> factor) : remain_size;
                    dst_temp = y_in_psram ? workspace : (int8_t *)dst + past_size;
                    int8_t *src1_temp = y_in_psram ? workspace : (int8_t *)dst + past_size;
                    THINKER_RET_CHECK(API_LIB(scale_i8i8o8)((int8_t *)src1 + past_size, shift1_0, (int8_t *)src1_temp, cur_size, shift1_1), "luna_scale_i8i8o8");
                    int8_t *src2_temp = y_in_psram ? (workspace + cur_size) : workspace;
                    THINKER_RET_CHECK(API_LIB(scale_i8i8o8)((int8_t *)src2 + past_size, shift2_0, (int8_t *)src2_temp, cur_size, shift2_1), "luna_scale_i8i8o8");
                    THINKER_RET_CHECK(API_LIB(add_i8i8o8)((int8_t *)src1_temp, (int8_t *)src2_temp, (int8_t *)dst_temp, cur_size, 0), "luna_add_i8i8o8");
                    if (y_in_psram)
                        opi_psram_cpy_out((void *)dst + past_size, dst_temp, cur_size * sizeof(int8_t));
                    past_size += cur_size;
                }
            }
            break;
        }

        case Int16:
        {
            int32_t workspace_size = Temp ? Temp->shape_.dims_[0] >> 1 : 0;
            int16_t *workspace = Temp ? (int16_t *)Temp->dptr_ : NULL;
            int16_t *dst_temp = y_in_psram ? workspace : (int16_t *)dst;
            if ((x1_in_psram == x2_in_psram) && (shift1 == 0) && (shift2 == 0))
            {
                THINKER_RET_CHECK(API_LIB(add_i16i16o16)((const int16_t *)src1, (int16_t *)src2, (int16_t *)dst, total_size, 0), "luna_add_i16i16o16");
            }
            else if ((shift1 != 0) && (shift2 == 0) && (!x2_in_psram))
            {
                uint32_t shift1_0 = shift1 < 0 ? 1UL << -shift1 : 1;
                uint32_t shift1_1 = shift1 < 0 ? 0 : shift1;
                if (!y_in_psram) {
                    int16_t *src1_temp = dst_temp;
                    THINKER_RET_CHECK(API_LIB(scale_i16i16o16)((int16_t *)src1, shift1_0, (int16_t *)src1_temp, total_size, shift1_1), "luna_scale_i16i16o16");
                    THINKER_RET_CHECK(API_LIB(add_i16i16o16)((const int16_t *)src1_temp, (int16_t *)src2, (int16_t *)dst, total_size, 0), "luna_add_i16i16o16");
                }
                else {
                    while (past_size < total_size)
                    {
                        int32_t remain_size = total_size - past_size;
                        int32_t cur_size = workspace_size < remain_size ? workspace_size : remain_size;

                        int16_t *src1_temp = dst_temp;
                        THINKER_RET_CHECK(API_LIB(scale_i16i16o16)((int16_t *)src1 + past_size, shift1_0, (int16_t *)src1_temp, cur_size, shift1_1), "luna_scale_i16i16o16");

                        int16_t *src2_temp = (int16_t *)src2 + past_size;
                        THINKER_RET_CHECK(API_LIB(add_i16i16o16)((int16_t *)src1_temp, (int16_t *)src2_temp, (int16_t *)dst_temp, cur_size, 0), "luna_add_i16i16o16");
                        opi_psram_cpy_out((void *)dst + past_size, dst_temp, cur_size * sizeof(int16_t));
                        past_size += cur_size;
                    }
                }
            }
            else if ((shift1 == 0) && (shift2 != 0) && (!x1_in_psram))
            {
                uint32_t shift2_0 = shift2 < 0 ? 1UL << -shift2 : 1;
                uint32_t shift2_1 = shift2 < 0 ? 0 : shift2;
                if (!y_in_psram) {
                    int16_t *src2_temp = dst_temp;
                    THINKER_RET_CHECK(API_LIB(scale_i16i16o16)((int16_t *)src2, shift2_0, (int16_t *)src2_temp, total_size, shift2_1), "luna_scale_i16i16o16");
                    THINKER_RET_CHECK(API_LIB(add_i16i16o16)((const int16_t *)src1, (int16_t *)src2_temp, (int16_t *)dst, total_size, 0), "luna_add_i16i16o16");
                }
                else {
                    while (past_size < total_size)
                    {
                        int32_t remain_size = total_size - past_size;
                        int32_t cur_size = workspace_size < remain_size ? workspace_size : remain_size;

                        int16_t *src1_temp = (int16_t *)src1 + past_size;
                        int16_t *src2_temp = dst_temp;
                        THINKER_RET_CHECK(API_LIB(scale_i16i16o16)((int16_t *)src2 + past_size, shift2_0, (int16_t *)src2_temp, cur_size, shift2_1), "luna_scale_i16i16o16");
                        THINKER_RET_CHECK(API_LIB(add_i16i16o16)((int16_t *)src1_temp, (int16_t *)src2_temp, (int16_t *)dst_temp, cur_size, 0), "luna_add_i16i16o16");
                        opi_psram_cpy_out((void *)dst + past_size, dst_temp, cur_size * sizeof(int16_t));
                        past_size += cur_size;
                    }
                }
            }
            else
            {
                uint32_t shift1_0 = shift1 < 0 ? 1UL << -shift1 : 1;
                uint32_t shift1_1 = shift1 < 0 ? 0 : shift1;
                uint32_t shift2_0 = shift2 < 0 ? 1UL << -shift2 : 1;
                uint32_t shift2_1 = shift2 < 0 ? 0 : shift2;
                int32_t factor = y_in_psram ? 1 : 0;
                while (past_size < total_size)
                {
                    int32_t remain_size = total_size - past_size;
                    int32_t cur_size = (workspace_size >> factor) < remain_size ? (workspace_size >> factor) : remain_size;

                    int16_t *src1_temp = dst_temp;
                    THINKER_RET_CHECK(API_LIB(scale_i16i16o16)((int16_t *)src1 + past_size, shift1_0, (int16_t *)src1_temp, cur_size, shift1_1), "luna_scale_i16i16o16");
                    int16_t *src2_temp = y_in_psram ? (workspace + cur_size) : workspace;
                    THINKER_RET_CHECK(API_LIB(scale_i16i16o16)((int16_t *)src2 + past_size, shift2_0, (int16_t *)src2_temp, cur_size, shift2_1), "luna_scale_i16i16o16");
                    THINKER_RET_CHECK(API_LIB(add_i16i16o16)((int16_t *)src1_temp, (int16_t *)src2_temp, (int16_t *)dst_temp, cur_size, 0), "luna_add_i16i16o16");
                    if (y_in_psram)
                        opi_psram_cpy_out((void *)dst + past_size, dst_temp, cur_size * sizeof(int16_t));
                    past_size += cur_size;
                }
            }
            break;
        }

        case Int32:
        {
            int32_t workspace_size = Temp ? Temp->shape_.dims_[0] >> 2 : 0;
            int32_t *workspace = Temp ? (int32_t *)Temp->dptr_ : NULL;
            int32_t *dst_temp = y_in_psram ? workspace : (int32_t *)dst;
            if ((x1_in_psram == x2_in_psram) && (shift1 == 0) && (shift2 == 0))
            {
                THINKER_RET_CHECK(API_LIB(add_i32i32o32)((const int32_t *)src1, (int32_t *)src2, (int32_t *)dst, total_size, 0), "luna_add_i32i32o32");
            }
            else if ((shift1 != 0) && (shift2 == 0) && (!x2_in_psram))
            {
                uint32_t shift1_0 = shift1 < 0 ? 1UL << -shift1 : 1;
                uint32_t shift1_1 = shift1 < 0 ? 0 : shift1;
                if (!y_in_psram) {
                    int32_t *src1_temp = dst_temp;
                    THINKER_RET_CHECK(API_LIB(scale_i32i32o32)((int32_t *)src1, shift1_0, (int32_t *)src1_temp, total_size, shift1_1), "luna_scale_i32i32o32");
                    THINKER_RET_CHECK(API_LIB(add_i32i32o32)((const int32_t *)src1_temp, (int32_t *)src2, (int32_t *)dst, total_size, 0), "luna_add_i32i32o32");
                }
                else {
                    while (past_size < total_size)
                    {
                        int32_t remain_size = total_size - past_size;
                        int32_t cur_size = workspace_size < remain_size ? workspace_size : remain_size;

                        int32_t *src1_temp = dst_temp;
                        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)((int32_t *)src1 + past_size, shift1_0, (int32_t *)src1_temp, cur_size, shift1_1), "luna_scale_i32i32o32");

                        int32_t *src2_temp = (int32_t *)src2 + past_size;
                        THINKER_RET_CHECK(API_LIB(add_i32i32o32)((int32_t *)src1_temp, (int32_t *)src2_temp, (int32_t *)dst_temp, cur_size, 0), "luna_add_i32i32o32");
                        opi_psram_cpy_out((void *)dst + past_size, dst_temp, cur_size * sizeof(int32_t));
                        past_size += cur_size;
                    }
                }
            }
            else if ((shift1 == 0) && (shift2 != 0) && (!x1_in_psram))
            {
                uint32_t shift2_0 = shift2 < 0 ? 1UL << -shift2 : 1;
                uint32_t shift2_1 = shift2 < 0 ? 0 : shift2;
                if (!y_in_psram) {
                    int32_t *src2_temp = dst_temp;
                    THINKER_RET_CHECK(API_LIB(scale_i32i32o32)((int32_t *)src2, shift2_0, (int32_t *)src2_temp, total_size, shift2_1), "luna_scale_i32i32o32");
                    THINKER_RET_CHECK(API_LIB(add_i32i32o32)((const int32_t *)src1, (int32_t *)src2_temp, (int32_t *)dst, total_size, 0), "luna_add_i32i32o32");
                }
                else {
                    while (past_size < total_size)
                    {
                        int32_t remain_size = total_size - past_size;
                        int32_t cur_size = workspace_size < remain_size ? workspace_size : remain_size;

                        int32_t *src1_temp = (int32_t *)src1 + past_size;
                        int32_t *src2_temp = dst_temp;
                        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)((int32_t *)src2 + past_size, shift2_0, (int32_t *)src2_temp, cur_size, shift2_1), "luna_scale_i32i32o32");
                        THINKER_RET_CHECK(API_LIB(add_i32i32o32)((int32_t *)src1_temp, (int32_t *)src2_temp, (int32_t *)dst_temp, cur_size, 0), "luna_add_i32i32o32");
                        opi_psram_cpy_out((void *)dst + past_size, dst_temp, cur_size * sizeof(int32_t));
                        past_size += cur_size;
                    }
                }
            }
            else
            {
                uint32_t shift1_0 = shift1 < 0 ? 1UL << -shift1 : 1;
                uint32_t shift1_1 = shift1 < 0 ? 0 : shift1;
                uint32_t shift2_0 = shift2 < 0 ? 1UL << -shift2 : 1;
                uint32_t shift2_1 = shift2 < 0 ? 0 : shift2;
                int32_t factor = y_in_psram ? 1 : 0;
                while (past_size < total_size)
                {
                    int32_t remain_size = total_size - past_size;
                    int32_t cur_size = (workspace_size >> factor) < remain_size ? (workspace_size >> factor) : remain_size;

                    int32_t *src1_temp = dst_temp;
                    THINKER_RET_CHECK(API_LIB(scale_i32i32o32)((int32_t *)src1 + past_size, shift1_0, (int32_t *)src1_temp, cur_size, shift1_1), "luna_scale_i32i32o32");
                    int32_t *src2_temp = y_in_psram ? (workspace + cur_size) : workspace;
                    THINKER_RET_CHECK(API_LIB(scale_i32i32o32)((int32_t *)src2 + past_size, shift2_0, (int32_t *)src2_temp, cur_size, shift2_1), "luna_scale_i32i32o32");
                    THINKER_RET_CHECK(API_LIB(add_i32i32o32)((int32_t *)src1_temp, (int32_t *)src2_temp, (int32_t *)dst_temp, cur_size, 0), "luna_add_i32i32o32");
                    if (y_in_psram)
                        opi_psram_cpy_out((void *)dst + past_size, dst_temp, cur_size * sizeof(int32_t));
                    past_size += cur_size;
                }
            }
            break;
        }

        default:
            return T_ERR_INVALID_DATATYPE;
    }

    return T_SUCCESS;
}

#endif