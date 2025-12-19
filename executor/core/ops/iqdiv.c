#undef __OP__
#define __OP__ iqDiv
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/iqdiv.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/iqdiv.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/iqdiv.h"
#endif

/**
 * Forward pass implementation for Integer Quantized Division operator
 * Performs element-wise division of two quantized tensors
 * @param op: Operator structure containing binary operation attributes
 * @param tensors: Array of input/output tensors (tensor1, tensor2, output)
 * @param num_tensor: Total number of tensors (must equal input + output count)
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Validate tensor count
    CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));
    
    // Get binary operation attributes
    iqBinaryAttrs *attrs = (iqBinaryAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    // Call hardware-specific division implementation
    ret = iqdiv_luna(tensors[0], tensors[1], tensors[op->num_input_]);

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","iqDiv", total_t);  
#endif
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__