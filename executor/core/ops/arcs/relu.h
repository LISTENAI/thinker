#ifndef __RELU_H__
#define __RELU_H__

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
 * @brief ReLU activation function implementation for various data types
 * @param X Input tensor
 * @param Y Output tensor
 * @param Workspace Workspace buffer
 * @return Operation result status
 */
tStatus relu_luna(tTensor *X, tTensor *Y, tTensor *Workspace) 
{
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    uint32_t size = getShapeSize(&(X->shape_));
    uint32_t workspace_size = Workspace ? getShapeSize(&(Workspace->shape_)) : 0;
    
    int32_t shift = Y->scale_ - X->scale_;
    
    // Handle Int8 -> Int8 case
    if (Int8 == X->dtype_ && Int8 == Y->dtype_) {
        int8_t *src = (int8_t *)X->dptr_;
        int8_t *dst = (int8_t *)Y->dptr_;

        // Handle PSRAM input
        if (2 != X->mem_.type_) { // X is psram
            ret = API_LIB(memcpy_i8o8)((int8_t *)Workspace->dptr_, src, size);
            if (size > workspace_size)
                return T_ERR_NO_WORKSPACE;
            src = (int8_t *)Workspace->dptr_;
        }
        
        // Handle PSRAM output
        if (2 != Y->mem_.type_) {
            if (size > workspace_size)
                return T_ERR_NO_WORKSPACE;
            dst = (int8_t *)Workspace->dptr_;
        }

        ret = API_LIB(relu_i8o8)(src, dst, size, shift);
        if (2 != Y->mem_.type_) {
            opi_psram_cpy_out((int8_t *)Y->dptr_, dst, size);
        }
    }
    // Handle Int8 -> Int32 case
    else if (Int8 == X->dtype_ && Int32 == Y->dtype_) {
        int8_t *src = (int8_t *)X->dptr_;
        int32_t *dst = (int32_t *)Y->dptr_;

        if (2 != X->mem_.type_) { // X is psram
            if (size > workspace_size)
                return T_ERR_NO_WORKSPACE;
            ret = API_LIB(memcpy_i8o8)((int8_t *)Workspace->dptr_, src, size);
            src = (int8_t *)Workspace->dptr_;
        }
        
        if (2 != Y->mem_.type_) {
            if (size > workspace_size)
                return T_ERR_NO_WORKSPACE;
            dst = (int32_t *)Workspace->dptr_;
        }
        
        ret = API_LIB(relu_i8o32)(src, dst, size, shift);

        if (2 != Y->mem_.type_) {
            opi_psram_cpy_out((int8_t *)Y->dptr_, dst, size * 4);
        }
    }
    // Handle Int32 -> Int8 case
    else if (Int32 == X->dtype_ && Int8 == Y->dtype_) {
        int32_t *src = (int32_t *)X->dptr_;
        int8_t *dst = (int8_t *)Y->dptr_;

        if (2 != X->mem_.type_) { // X is psram
            ret = API_LIB(memcpy_i8o8)((int8_t *)Workspace->dptr_, (int8_t *)src, size * 4);
            src = (int32_t *)Workspace->dptr_;
        }
        
        if (2 != Y->mem_.type_)
            dst = (int8_t *)Workspace->dptr_;

        ret = API_LIB(relu_i32o8)(src, dst, size, shift);

        if (2 != Y->mem_.type_) {
            opi_psram_cpy_out((int8_t *)Y->dptr_, dst, size);
        }
    }
    // Handle Int32 -> Int32 case
    else if (Int32 == X->dtype_ && Int32 == Y->dtype_) {
        int32_t *src = (int32_t *)X->dptr_;
        int32_t *dst = (int32_t *)Y->dptr_;

        if (2 != X->mem_.type_) { // X is psram
            ret = API_LIB(memcpy_i8o8)((int8_t *)Workspace->dptr_, (int8_t *)src, size * 4);
            src = (int32_t *)Workspace->dptr_;
        }
        
        if (2 != Y->mem_.type_)
            dst = (int32_t *)Workspace->dptr_;

        ret = API_LIB(relu_i32o32)(src, dst, size, shift);

        if (2 != Y->mem_.type_) {
            opi_psram_cpy_out((int8_t *)Y->dptr_, dst, size * 4);
        }
    }
    else {
        return T_ERR_INVALID_DATATYPE;
    }

    return ret;
}

#endif