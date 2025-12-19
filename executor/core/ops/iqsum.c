#undef __OP__
#define __OP__ iqSum
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/iqsum.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/iqsum.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/iqsum.h"
#endif

/**
 * Forward pass implementation for Integer Quantized Sum operator
 * Computes the sum of elements in the input tensor
 * @param op: Operator structure containing sum attributes
 * @param tensors: Array of input/output tensors (input, output, workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Validate tensor count
    CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_ + 1));
    
    // Get sum attributes
    iqSumAttrs *attrs = (iqSumAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
    // Get temporary workspace tensor
    tTensor *Temp = tensors[op->num_input_ + op->num_output_];
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    // Call hardware-specific sum implementation
    ret = iqsum_luna(tensors[0], Temp, tensors[op->num_input_], attrs);

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","iqSum", total_t);  
#endif
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__