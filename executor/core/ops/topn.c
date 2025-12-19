// topN operator implementation

#undef __OP__
#define __OP__ topN
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/topn.h"  // Venus backend implementation
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/topn.h"   // Arcs backend implementation
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/topn.h" // VenusA backend implementation
#endif

/**
 * @brief Execute the topN operation
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));  // Validate tensor count

    topNAttrs *attrs = (topNAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t ret = T_ERR_NO_IMPLEMENTED;

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
    ret = topn_luna(tensors[0], tensors[1], tensors[op->num_input_], tensors[num_tensor - 1], attrs);
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__