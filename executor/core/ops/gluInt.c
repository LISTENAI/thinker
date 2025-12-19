#undef __OP__
#define __OP__ GluInt
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_ARCS
#include "./arcs/gluint.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/gluint.h"
#endif // THINKER_USE_VENUSA

/**
 * Forward pass implementation for GLU (Gated Linear Unit) Integer operator
 * Applies gated linear unit activation to input tensor
 * @param op: Operator structure containing GLU attributes
 * @param tensors: Array of input/output tensors (input, output, workspace)
 * @param num_tensor: Total number of tensors (must equal input + output + workspace count)
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Validate tensor count
    CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_ + 1));
    
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    GluIntAttrs *attrs = (GluIntAttrs *)((int8_t *)op + op->attr_offset_);
    
#if THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif
    
    // Call hardware-specific GLU implementation
    ret = gluint_luna(tensors[0], tensors[op->num_input_], tensors[op->num_input_ + 1], attrs);
    
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","GluInt", total_t);
#endif

#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__