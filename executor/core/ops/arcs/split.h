#ifndef _SPLIT_LUNA_H_
#define _SPLIT_LUNA_H_

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/type_switch.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "thinker_status.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_basic_math.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

#if !(defined(WIN32) || defined(linux))
#pragma clang optimize off
#endif

/**
 * @brief Split tensor along specified axis into multiple tensors
 * @param X Input tensor to split
 * @param tensors Array of output tensors
 * @param attrs Split attributes including axis and split sizes
 * @return Operation status
 */
int32_t split_venus(tTensor *X, tTensor **tensors, SliceAttrs *attrs) 
{
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
    // Handle negative axis values
    if (attrs->axis < 0) {
        attrs->axis += X->shape_.ndim_;
    }
    if (attrs->axis >= X->shape_.ndim_) {
        return T_ERR_INVALID_PARA;
    }

    // Calculate dimensions for memory operations
    int32_t leading = 1, middle = 1, stride = 1;
    int32_t index = 0;
    
    // Compute leading dimensions
    for (; index < attrs->axis; ++index) {
        leading *= X->shape_.dims_[index];
    }
    
    // Get middle dimension
    middle = X->shape_.dims_[index++];
    
    // Compute stride dimensions
    for (; index < X->shape_.ndim_; ++index) {
        stride *= X->shape_.dims_[index];
    }

    // Split data into output tensors
    int32_t offset = 0;
    for (int32_t n = 0; n < attrs->dims; ++n) {
        const tTensor *out = tensors[n + 1];
        
        // Validate data types match
        if (X->dtype_ != out->dtype_)
            return T_ERR_INVALID_DATATYPE;

        // Copy data based on memory type
        if(2 == out->mem_.type_) {
            for (int32_t i = 0; i < leading; ++i) {
                int8_t *idst = (int8_t *)(out->dptr_) + i * attrs->split[n] * stride * out->byte_;
                int8_t *isrc = (int8_t *)(X->dptr_) + i * middle * stride + offset * stride * X->byte_;
                ret = API_LIB(memcpy_i8o8)(idst, isrc, sizeof(int8_t) * attrs->split[n] * stride * out->byte_);
            }
        } else {
            for (int32_t i = 0; i < leading; ++i) {
                int8_t *idst = (int8_t *)(out->dptr_) + i * attrs->split[n] * stride * out->byte_;
                int8_t *isrc = (int8_t *)(X->dptr_) + i * middle * stride * X->byte_ + offset * stride * X->byte_;
                opi_psram_cpy_out(idst, isrc, sizeof(int8_t) * attrs->split[n] * stride * out->byte_);
            }
            ret = T_SUCCESS;
        }
        
        ret = T_SUCCESS;
        offset += attrs->split[n];
    }

    return ret;
}

#if !(defined(WIN32) || defined(linux))
#pragma clang optimize on
#endif
#endif