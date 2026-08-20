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

static int32_t iqadd_check_scale_shift(int32_t shift) {
    return (shift < -6 || shift > 63) ? T_ERR_INVALID_PARA : T_SUCCESS;
}

static uint32_t iqadd_scale_multiplier(int32_t shift) {
    return shift < 0 ? (1U << (uint32_t)(-shift)) : 1U;
}

/**
 * @brief Quantized tensor addition operation
 * @param X1 Input tensor 1
 * @param X2 Input tensor 2
 * @param Temp Workspace tensor
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
    if (!equalShape(&X1->shape_, &X2->shape_) || !equalShape(&X1->shape_, &Y->shape_) ||
        X1->dtype_ != X2->dtype_ || X1->dtype_ != Y->dtype_ || X1->dtype_ != Int8) {
        return T_ERR_INVALID_DATATYPE;
    }
    #if THINKER_PARAM_CHECK
    if (!equalShape(&X1->shape_, &X2->shape_) ||
        X1->dtype_ != X2->dtype_ ||
        X1->dtype_ != Y->dtype_ || X1->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }
#endif

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
    int32_t check_ret = T_SUCCESS;
    if ((check_ret = iqadd_check_scale_shift(shift1)) != T_SUCCESS ||
        (check_ret = iqadd_check_scale_shift(shift2)) != T_SUCCESS) {
        return check_ret;
    }
    #if THINKER_PARAM_CHECK
    if ((check_ret = iqadd_check_scale_shift(shift1)) != T_SUCCESS) {
        return (check_ret);
    }

    if ((check_ret = iqadd_check_scale_shift(shift2)) != T_SUCCESS) {
        return (check_ret);
    }
#endif

    int32_t past_size = 0;
    #if THINKER_PARAM_CHECK
    if ((X1->mem_.type_ != 1 && X1->mem_.type_ != 2) ||
                        (X2->mem_.type_ != 1 && X2->mem_.type_ != 2) ||
                        (Y->mem_.type_ != 1 && Y->mem_.type_ != 2)) {
        return (T_ERR_NO_SUPPORT_OP);
    }
#endif
    int32_t workspace_size = Temp ? (int32_t)getTensorDataSize(Temp) : 0;
    int8_t *workspace = Temp ? (int8_t *)Temp->dptr_ : NULL;
    #if THINKER_RUNTIME_CHECK
    if (Temp != NULL &&
                          (Temp->dptr_ == 0 || Temp->mem_.type_ != 2 ||
                           Temp->dtype_ != Int8 || Temp->byte_ != 1 ||
                           ((uintptr_t)Temp->dptr_ & 3U) != 0)) {
        return (T_ERR_NO_WORKSPACE);
    }

    if (workspace_size > 0 && workspace_size < 2) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif

    // Branch 1: Simple case - no scale conversion needed
    if ((!x1_in_psram) && (!x2_in_psram) && (shift1 == 0) && (shift2 == 0))
    {
        if (!y_in_psram) {
            THINKER_RET_CHECK(API_LIB(add_i8i8o8)((const int8_t *)src1, (int8_t *)src2,
                                                   (int8_t *)dst, total_size, 0), "luna_add_i8i8o8");
        } else {
            if (workspace == NULL || workspace_size <= 0) return T_ERR_NO_WORKSPACE;
            while (past_size < total_size) {
                int32_t remain_size = total_size - past_size;
                int32_t cur_size = workspace_size < remain_size ? workspace_size : remain_size;
                THINKER_RET_CHECK(API_LIB(add_i8i8o8)((int8_t *)src1 + past_size,
                                                       (int8_t *)src2 + past_size, workspace,
                                                       cur_size, 0), "luna_add_i8i8o8");
                opi_psram_cpy_out((int8_t *)dst + past_size, workspace, cur_size);
                past_size += cur_size;
            }
        }
    }
    else if (((x1_in_psram) || (shift1 != 0)) && (!x2_in_psram) && (shift2 == 0))
    {
        if (y_in_psram) {
            if (workspace_size <= 0)
                return T_ERR_NO_WORKSPACE;
            while (past_size < total_size) {
                int32_t remain_size = total_size - past_size;
                int32_t cur_size = workspace_size < remain_size ? workspace_size : remain_size;

                int8_t *dst_temp = workspace;
                int8_t *src1_temp = (int8_t *)src1 + past_size;
                int8_t *src2_temp = (int8_t *)src2 + past_size;
                if (x1_in_psram) {
                    src1_temp = dst_temp;
                    THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)src1_temp, (int8_t *)src1 + past_size, cur_size), "luna_memcpy_i8o8");
                }

                if (shift1 != 0) {
                    uint32_t shift1_0 = iqadd_scale_multiplier(shift1);
                    uint32_t shift1_1 = shift1 < 0 ? 0 : shift1;
                    THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(src1_temp, shift1_0, (int8_t *)dst_temp, cur_size, shift1_1), "luna_scale_i8i8o8");
                    src1_temp = dst_temp;
                }

                THINKER_RET_CHECK(API_LIB(add_i8i8o8)((int8_t *)src1_temp, (int8_t *)src2_temp, (int8_t *)dst_temp, cur_size, 0), "luna_add_i8i8o8");
                opi_psram_cpy_out((void *)dst + past_size, dst_temp, cur_size * sizeof(int8_t));
                past_size += cur_size;
            }
        }
        else {
            int8_t *src1_temp = (int8_t *)src1;
            int8_t *src2_temp = (int8_t *)src2;
            if (x1_in_psram) {
                src1_temp = (int8_t *)dst;
                THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)src1_temp, (int8_t *)src1, total_size), "luna_memcpy_i8o8");
            }

            if (shift1 != 0) {
                uint32_t shift1_0 = iqadd_scale_multiplier(shift1);
                uint32_t shift1_1 = shift1 < 0 ? 0 : shift1;
                THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(src1_temp, shift1_0, (int8_t *)dst, total_size, shift1_1), "luna_scale_i8i8o8");
                src1_temp = (int8_t *)dst;
            }

            THINKER_RET_CHECK(API_LIB(add_i8i8o8)((int8_t *)src1_temp, (int8_t *)src2_temp, (int8_t *)dst, total_size, 0), "luna_add_i8i8o8");
        }
    }
    // Branch 2: Need to scale X1, X2 is ready (in share-memory, no scale)
    else if ((!x1_in_psram) && (shift1 == 0) && ((x2_in_psram) || (shift2 != 0)))
    {
        if (y_in_psram) {
            if (workspace_size <= 0)
                return T_ERR_NO_WORKSPACE;
            while (past_size < total_size) {
                int32_t remain_size = total_size - past_size;
                int32_t cur_size = workspace_size < remain_size ? workspace_size : remain_size;

                int8_t *dst_temp = workspace;
                int8_t *src1_temp = (int8_t *)src1 + past_size;
                int8_t *src2_temp = (int8_t *)src2 + past_size;
                if (x2_in_psram) {
                    src2_temp = dst_temp;
                    THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)src2_temp, (int8_t *)src2 + past_size, cur_size), "luna_memcpy_i8o8");
                }

                if (shift2 != 0) {
                    uint32_t shift2_0 = iqadd_scale_multiplier(shift2);
                    uint32_t shift2_1 = shift2 < 0 ? 0 : shift2;
                    THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(src2_temp, shift2_0, (int8_t *)dst_temp, cur_size, shift2_1), "luna_scale_i8i8o8");
                    src2_temp = dst_temp;
                }

                THINKER_RET_CHECK(API_LIB(add_i8i8o8)((int8_t *)src1_temp, (int8_t *)src2_temp, (int8_t *)dst_temp, cur_size, 0), "luna_add_i8i8o8");
                opi_psram_cpy_out((void *)dst + past_size, dst_temp, cur_size * sizeof(int8_t));
                past_size += cur_size;
            }
        }
        else {
            int8_t *src1_temp = (int8_t *)src1;
            int8_t *src2_temp = (int8_t *)src2;
            if (x2_in_psram) {
                THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)dst, (int8_t *)src2, total_size), "luna_memcpy_i8o8");
                src2_temp = (int8_t *)dst;
            }

            if (shift2 != 0) {
                uint32_t shift2_0 = iqadd_scale_multiplier(shift2);
                uint32_t shift2_1 = shift2 < 0 ? 0 : shift2;
                THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(src2_temp, shift2_0, (int8_t *)dst, total_size, shift2_1), "luna_scale_i8i8o8");
                src2_temp = (int8_t *)dst;
            }

            THINKER_RET_CHECK(API_LIB(add_i8i8o8)((int8_t *)src1_temp, (int8_t *)src2_temp, (int8_t *)dst, total_size, 0), "luna_add_i8i8o8");
        }
    }
    else {
        if (workspace_size <= 0)
            return T_ERR_NO_WORKSPACE;
        if (y_in_psram) {
            while (past_size < total_size) {
                int32_t remain_size = total_size - past_size;
                int32_t cur_size = (workspace_size >> 1) < remain_size ? (workspace_size >> 1) : remain_size;

                int8_t *dst_temp = workspace;
                int8_t *src1_temp = (int8_t *)src1 + past_size;
                int8_t *src2_temp = (int8_t *)src2 + past_size;
                if (x1_in_psram) {
                    src1_temp = (int8_t *)dst_temp;
                    THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)src1_temp, (int8_t *)src1 + past_size, cur_size), "luna_memcpy_i8o8");
                }

                if (shift1 != 0) {
                    uint32_t shift1_0 = iqadd_scale_multiplier(shift1);
                    uint32_t shift1_1 = shift1 < 0 ? 0 : shift1;
                    THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(src1_temp, shift1_0, (int8_t *)dst_temp, cur_size, shift1_1), "luna_scale_i8i8o8");
                    src1_temp = dst_temp;
                }

                if (x2_in_psram) {
                    src2_temp = (int8_t *)dst_temp + cur_size;
                    THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)src2_temp, (int8_t *)src2 + past_size, cur_size), "luna_memcpy_i8o8");
                }

                if (shift2 != 0) {
                    uint32_t shift2_0 = iqadd_scale_multiplier(shift2);
                    uint32_t shift2_1 = shift2 < 0 ? 0 : shift2;
                    THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(src2_temp, shift2_0, (int8_t *)dst_temp + cur_size, cur_size, shift2_1), "luna_scale_i8i8o8");
                    src2_temp = (int8_t *)dst_temp + cur_size;
                }

                THINKER_RET_CHECK(API_LIB(add_i8i8o8)((int8_t *)src1_temp, (int8_t *)src2_temp, (int8_t *)dst_temp, cur_size, 0), "luna_add_i8i8o8");
                opi_psram_cpy_out((void *)dst + past_size, dst_temp, cur_size * sizeof(int8_t));
                past_size += cur_size;
            }
        }
        else {
            while (past_size < total_size) {
                int32_t remain_size = total_size - past_size;
                int32_t cur_size = (workspace_size >> 1) < remain_size ? (workspace_size >> 1) : remain_size;

                int8_t *dst_temp = (int8_t *)dst + past_size;
                int8_t *src1_temp = (int8_t *)src1 + past_size;
                int8_t *src2_temp = (int8_t *)src2 + past_size;
                if (x1_in_psram) {
                    src1_temp = (int8_t *)dst_temp;
                    THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)src1_temp, (int8_t *)src1 + past_size, cur_size), "luna_memcpy_i8o8");
                }

                if (shift1 != 0) {
                    uint32_t shift1_0 = iqadd_scale_multiplier(shift1);
                    uint32_t shift1_1 = shift1 < 0 ? 0 : shift1;
                    THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(src1_temp, shift1_0, (int8_t *)dst_temp, cur_size, shift1_1), "luna_scale_i8i8o8");
                    src1_temp = dst_temp;
                }

                if (x2_in_psram) {
                    src2_temp = (int8_t *)workspace;
                    THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)src2_temp, (int8_t *)src2 + past_size, cur_size), "luna_memcpy_i8o8");
                }

                if (shift2 != 0) {
                    uint32_t shift2_0 = iqadd_scale_multiplier(shift2);
                    uint32_t shift2_1 = shift2 < 0 ? 0 : shift2;
                    THINKER_RET_CHECK(API_LIB(scale_i8i8o8)(src2_temp, shift2_0, (int8_t *)workspace, cur_size, shift2_1), "luna_scale_i8i8o8");
                    src2_temp = (int8_t *)workspace;
                }

                THINKER_RET_CHECK(API_LIB(add_i8i8o8)((int8_t *)src1_temp, (int8_t *)src2_temp, (int8_t *)dst_temp, cur_size, 0), "luna_add_i8i8o8");
                past_size += cur_size;
            }
        }
    }
#if !(defined(WIN32) || defined(linux))
    if (y_in_psram)
        HAL_FlushInvalidateDCache_by_Addr((uint32_t *)dst, total_size);
#endif
    return T_SUCCESS;
}

#endif
