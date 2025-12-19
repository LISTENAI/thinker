#undef __OP__
#define __OP__ Gather
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/gather.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/gather.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/gather.h"
#endif

/**
 * Forward pass implementation for Gather operator
 * Gathers slices from input tensor along a given axis using indices
 * @param op: Operator structure containing gather attributes
 * @param tensors: Array of input tensors (input, indices, output)
 * @param num_tensor: Total number of tensors (must be 3)
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Validate tensor count
    CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));
    
    // Get gather attributes
    GatherAttrs *attr = (GatherAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
    // Validate exact number of tensors
    if (num_tensor != 3) return T_ERR_INVALID_PARA;
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif
    
    // Call hardware-specific gather implementation
    ret = gather_luna(tensors[0], tensors[1], tensors[op->num_input_], attr);
    
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","Gather", total_t);  
#endif
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__