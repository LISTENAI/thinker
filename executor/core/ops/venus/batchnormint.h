#ifndef _BATCHNORMINT_VENUS_H_
#define _BATCHNORMINT_VENUS_H_

#include <math.h>
#include "c_api/thinker_define.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

#include "thinker_status.h"

/**
 * @brief Perform batch normalization on quantized integer tensors
 * @param X Input tensor
 * @param W Scale tensor
 * @param Bias Offset tensor
 * @param Y Output tensor
 * @param workspace Workspace buffer
 * @return Operation status
 */
int32_t batchnormint_luna(const tTensor *X, const tTensor *W, const tTensor *Bias, tTensor *Y, tTensor *workspace) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
    int32_t N = X->shape_.dims_[0];  // Batch size
    int32_t F = X->shape_.dims_[2] * X->shape_.dims_[3];  // Channels per feature map
    int32_t C = X->shape_.dims_[1];  // Number of channels
    int32_t one_batch_size = F * C;  // Size of one batch
    
    int8_t *p_src = (int8_t *)X->dptr_;  // Pointer to input data
    int8_t *p_dst = (int8_t *)Y->dptr_;  // Pointer to output data
    int8_t *p_weight = (int8_t *)W->dptr_;  // Pointer to scale weights
    int32_t *p_bias = (int32_t *)Bias->dptr_;  // Pointer to bias
    int32_t *p_tmp = (int32_t *)workspace->dptr_;  // Pointer to temporary workspace
    
    int32_t q_x = (int32_t)X->scale_;  // Input scale
    int32_t q_w = (int32_t)W->scale_;  // Weight scale
    int32_t q_o = (int32_t)Y->scale_;  // Output scale
    int32_t shift = q_x + q_w - q_o;  // Scale shift factor
    
    for (int32_t i = 0; i < N; i++) {  // Iterate over batches
        for (int32_t j = 0; j < C; j++) {  // Iterate over channels
            int8_t w_val = *(p_weight + j);  // Current channel's scale
            int32_t b_val = *(p_bias + j);  // Current channel's bias
            
            int8_t *p_in = p_src + i * one_batch_size + j * F;  // Input pointer for current channel
            int8_t *p_ou = p_dst + i * one_batch_size + j * F;  // Output pointer for current channel
            
            // Apply scale and bias
            ret = API_LIB(scale_q7_int32)(p_in, w_val, p_tmp, F, 0);  // Scale input
            ret = API_LIB(offset_q31_int8)(p_tmp, b_val, p_ou, F, shift);  // Apply bias and shift
        }
    }
    
    return ret;
}

#endif  //_BATCHNORMINT_VENUS_H_