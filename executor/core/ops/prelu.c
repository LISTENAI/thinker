// PRelu operator implementation

#undef __OP__
#define __OP__ PRelu
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/prelu.h"  // Venus backend implementation
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/prelu.h"   // Arcs backend implementation
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/prelu.h" // VenusA backend implementation
#endif

/**
 * @brief Execute the PRelu operation
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));  // Validate tensor count
    
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    PreluAttrs *attrs = (PreluAttrs *)((int8_t *)op + op->attr_offset_);

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();  // Start profiling
#endif
    ret = prelu_luna(tensors[0], tensors[op->num_input_], attrs);  // Execute PRelu operation
    
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","prelu", total_t);  // Print profiling results
#endif
#endif

    return 0;
}

#include "core/operator_template.h"
#undef __OP__