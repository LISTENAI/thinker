#undef __OP__
#define __OP__ iqTanh
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/iqtanh.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/iqtanh.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/iqtanh.h"
#endif

/**
 * Forward pass implementation for Integer Quantized Hyperbolic Tangent operator
 * Applies tanh activation to input tensor
 * @param op: Operator structure
 * @param tensors: Array of input/output tensors (input, output)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Validate tensor count
    CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));
    
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    // Call hardware-specific tanh implementation
    ret = iqtanh(tensors[0], tensors[op->num_input_]);

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","iqTanh", total_t);  
#endif
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__