#undef __OP__
#define __OP__ iqVar
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_VENUS
#include "./venus/iqvar.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/iqvar.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/iqvar.h"
#endif

/**
 * Forward pass implementation for Integer Quantized Variance operator
 * Computes the variance of elements in the input tensor
 * @param op: Operator structure containing variance attributes
 * @param tensors: Array of input/output tensors (input, output, workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Validate tensor count
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
    
    // Get variance attributes
    iqvarAttrs *attrs = (iqvarAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
    // Get input, output, and workspace tensors
    tTensor *X = ((tTensor **)tensors)[0];
    tTensor *workspace = ((tTensor **)tensors)[num_tensor - 1];
    tTensor *Y = ((tTensor **)tensors)[op->num_input_];
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    // Call hardware-specific variance implementation
    ret = iqvar_luna(X, Y, workspace, attrs);

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","iqVar", total_t);  
#endif
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__