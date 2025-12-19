#undef __OP__
#define __OP__ iqMul
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/iqmul.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/iqmul.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/iqmul.h"
#endif

/**
 * Forward pass implementation for Integer Quantized Multiplication operator
 * Performs element-wise multiplication of two quantized tensors
 * @param op: Operator structure containing binary operation attributes
 * @param tensors: Array of input/output tensors (tensor1, tensor2, output, workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Validate tensor count
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
    
    // Get binary operation attributes
    iqBinaryAttrs *attrs = (iqBinaryAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
    // Get workspace tensor
    tTensor *workspace = ((tTensor **)tensors)[num_tensor - 1];
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    // Call hardware-specific multiplication implementation
    ret = iqmul_luna(tensors[0], tensors[1], tensors[op->num_input_], workspace, attrs);

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","iqMul", total_t);  
#endif
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__