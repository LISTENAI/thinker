// Split operator implementation

#undef __OP__
#define __OP__ Split
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/split.h"  // Venus backend implementation
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/split.h"   // Arcs backend implementation
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/split.h" // VenusA backend implementation
#endif

/**
 * @brief Execute the Split operation
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));  // Validate tensor count

    SliceAttrs *attr = (SliceAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t ret = T_ERR_NO_IMPLEMENTED;

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
    ret = split_venus(tensors[0], tensors, attr);  // Execute split operation
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__