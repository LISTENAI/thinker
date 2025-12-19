#ifndef _SOFTMAXINT_LUNA_H_
#define _SOFTMAXINT_LUNA_H_

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "thinker_status.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief Softmax operation for integer tensors
 * @param data Input tensor
 * @param out Output tensor
 * @param Workspace Workspace buffer
 * @param attrs Softmax attributes
 * @return Operation result status
 */
int32_t softmaxint_luna(tTensor *data, tTensor *out, tTensor *Workspace, SoftmaxIntAttrs *attrs)
{
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    const int32_t SOFTMAX_Q_IN = 25;
    const int32_t SOFTMAX_Q_OUT = 15;
    int32_t leading = 1, stride = 1;
    int32_t i = 0;
    int32_t axis = 1;
    
    // Handle negative axis values
    if (attrs->axis < 0)
    {
        axis = data->shape_.ndim_ + attrs->axis;
    }
    
    // Calculate leading dimensions
    for (; i < axis; ++i)
    {
        leading *= data->shape_.dims_[i];
    }
    
    // Calculate stride (dimension along which softmax is applied)
    for (; i < data->shape_.ndim_; ++i)
    {
        stride *= data->shape_.dims_[i];
    }
    
    int32_t data_size = leading * stride;

    // Only support fast memory (type 2) for both input and output
    if ((2 != data->mem_.type_) || (2 != out->mem_.type_))
        return T_ERR_INVALID_PLATFROM;

    // Process only Int8 data type
    if (Int8 == data->dtype_)
    {
        int32_t *data_temp = (int32_t *)(Workspace->dptr_);
        int32_t workspace_size = Workspace ? Workspace->shape_.dims_[0] : 0;

        int32_t x_scale = (int32_t)data->scale_;
        int32_t y_scale = (int32_t)out->scale_;

        // Convert input to appropriate fixed-point format
        if (Int8 == data->dtype_) {
            ret = API_LIB(scale_i8i8o32)((int8_t *)data->dptr_, 1, (int32_t *)data_temp, data_size, 0);  // Q4->Q25
            ret = API_LIB(scale_i32i32o32)((int32_t *)data_temp, (1 << (SOFTMAX_Q_IN - x_scale)), (int32_t *)data_temp, data_size, 0);  // Q4->Q25
        }
        else if (Int32 == data->dtype_)
            ret = API_LIB(scale_i32i32o32)((int32_t *)data->dptr_, (1 << (SOFTMAX_Q_IN - x_scale)), (int32_t *)data_temp, data_size, 0);  // Q4->Q25;
        else
            return T_ERR_INVALID_DATATYPE;

        // Handle different output data types
        if (Int8 == out->dtype_) {
            int32_t *dst_tmp = (int32_t *)Workspace->dptr_ + (Int8 == data->dtype_) * data_size;
            for (int32_t l = 0; l < leading; ++l)
            {
                int32_t offset = l * stride;
                ret = API_LIB(softmax_i32o32)((int32_t *)data_temp + offset, (int32_t *)dst_tmp, stride);  // Q25->Q15        
                ret |= API_LIB(scale_i32i32o8)((int32_t *)dst_tmp, 1, (int8_t *)out->dptr_ + offset, stride, (SOFTMAX_Q_OUT - y_scale));
            }
        }
        else if (Int32 == out->dtype_) {
            for (int32_t l = 0; l < leading; ++l)
            {
                int32_t offset = l * stride;
                ret = API_LIB(softmax_i32o32)((int32_t *)data_temp + offset, (int32_t *)out->dptr_ + offset, stride);  // Q25->Q15        
            }
        }
    }

    return ret;
}

#endif
