#undef __OP__
#define __OP__ Flatten
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/flatten.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/flatten.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/flatten.h"
#endif

/**
 * Forward pass implementation for Flatten operator
 * Reshapes input tensor to 2D format (batch_size, total_elements)
 * @param op: Operator structure containing flatten attributes
 * @param tensors: Array of input/output tensors (input, output)
 * @param num_tensor: Total number of tensors (must equal input + output count)
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Validate tensor count
    CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));
    
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    FlattenAttrs *attr = (FlattenAttrs *)((int8_t *)op + op->attr_offset_);
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
    // Call hardware-specific flatten implementation
    ret = flatten_luna(tensors[0], tensors[op->num_input_], attr);
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__