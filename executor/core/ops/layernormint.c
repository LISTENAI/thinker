#undef __OP__
#define __OP__ LayerNormInt
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_VENUS
#include "./venus/layernormint.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/layernormint.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/layernormint.h"
#endif

/**
 * Forward pass implementation for Integer Quantized Layer Normalization operator
 * Applies layer normalization to input tensor
 * @param op: Operator structure containing layer normalization attributes
 * @param tensors: Array of input/output tensors (input, weight, output, optional bias, optional workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list for weight data handling
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Validate tensor count
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
    
    // Get layer normalization attributes
    LayerNormIntAttrs *attrs = (LayerNormIntAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
    // Get input, weight, and output tensors
    tTensor *X = ((tTensor **)tensors)[0];
    tTensor *W = ((tTensor **)tensors)[1];
    tTensor *Y = ((tTensor **)tensors)[op->num_input_];
    
    tTensor weight_tmp = W[0];
    tTensor *bias = NULL;
    tTensor *workspace = NULL;
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
    if (list->total_ != 0)
        getWeightData(list, 0);
    
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif
    
    if (list->total_ > 0) {
        tTensor *dma_buffer = NULL;
        if (num_tensor == op->num_input_ + op->num_output_ + 1) {
            dma_buffer = ((tTensor **)tensors)[op->num_input_ + op->num_output_];
            weight_tmp.dptr_ = (addr_type)dma_buffer->dptr_;
        } else if (num_tensor == op->num_input_ + op->num_output_ + 2) {
            workspace = ((tTensor **)tensors)[op->num_input_ + op->num_output_];
            dma_buffer = ((tTensor **)tensors)[op->num_input_ + op->num_output_ + 1];
            weight_tmp.dptr_ = (addr_type)dma_buffer->dptr_;
        }
        
        if (3 == op->num_input_) {
            bias = ((tTensor **)tensors)[op->num_input_ - 1];
            tTensor Bias_temp = bias[0];
            Bias_temp.scale_ = X->scale_ + W->scale_;
            int32_t size = getShapeSize(&(W->shape_)) * W->byte_;
            Bias_temp.dptr_ = (addr_type)((int8_t *)weight_tmp.dptr_ + ALIGN16(size));
        }
        
        ret = layernormalint_venus(X, &weight_tmp, bias, Y, workspace, attrs);
    } else {
        if (3 == op->num_input_) {
            bias = ((tTensor **)tensors)[op->num_input_ - 1];
            bias->scale_ = X->scale_ + W->scale_;
        }
        if (num_tensor == op->num_input_ + op->num_output_ + 1) {
            workspace = ((tTensor **)tensors)[op->num_input_ + op->num_output_];
        }
        ret = layernormalint_venus(X, W, bias, Y, workspace, attrs);
    }
    
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","LayernormInt", total_t);  
#endif
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__