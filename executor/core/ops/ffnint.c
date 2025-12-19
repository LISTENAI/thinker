#undef __OP__
#define __OP__ FFNInt
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_ARCS
#include "./arcs/ffnint.h"
#endif

/**
 * Forward pass implementation for Feed Forward Network Integer operator
 * Performs two-layer linear transformation with activation
 * @param op: Operator structure containing FFN attributes
 * @param tensors: Array of input/output tensors (input, weight1, bias1, weight2, bias2, output, optional workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Validate tensor count
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
    
    // Get FFN attributes
    FFNIntAttrs *attrs = (FFNIntAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
    // Get all tensor pointers
    tTensor *input    = tensors[0];
    tTensor *weight1  = tensors[1];
    tTensor *bias1    = tensors[2];
    tTensor *weight2  = tensors[3];
    tTensor *bias2    = tensors[4];
    tTensor *output   = tensors[op->num_input_];
    tTensor *workspace = NULL;
    
#ifdef THINKER_USE_ARCS
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif
    
    // Get workspace tensor if present
    if (num_tensor == op->num_input_ + op->num_output_ + 1) {
        workspace = ((tTensor **)tensors)[op->num_input_ + op->num_output_];
    }
    
    // Call hardware-specific FFN implementation
    ret = ffnint_luna(input, weight1, bias1, weight2, bias2, workspace, output, attrs);
    
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","FFNInt", total_t);  
#endif
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__