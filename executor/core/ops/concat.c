#undef __OP__
#define __OP__ iqCat
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/concat.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/concat.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/concat.h"
#endif

/**
 * Forward pass implementation for Concat operator
 * @param op: Operator structure containing concat attributes
 * @param tensors: Array of input/output tensors
 * @param num_tensor: Total number of tensors
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Validate tensor count
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
    
    // Get concat attributes
    iqCatAttrs *attr = (iqCatAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t axis = attr->axis;
    
    // Handle negative axis indexing
    if (axis < 0) {
        axis += tensors[0]->shape_.ndim_;
    }
    
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
    // Check for workspace tensor
    tTensor *workspace = NULL;
    if (num_tensor == op->num_input_ + op->num_output_ + 1){
        workspace = ((tTensor**)tensors)[op->num_input_ + op->num_output_];
    }

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif
    // Call hardware-specific concat implementation
    ret = concat_luna(tensors, axis, op->num_input_, workspace, tensors[op->num_input_]);
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","Concat", total_t);  
#endif
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__