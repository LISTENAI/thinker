// Slice operator implementation

#undef __OP__
#define __OP__ Slice
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/slice.h"  // Venus backend implementation
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/slice.h"   // Arcs backend implementation
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/slice.h" // VenusA backend implementation
#endif

/**
 * @brief Execute the Slice operation
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));  // Validate tensor count

    // Get slice parameters
    int32_t start = (int32_t)(((int64_t *)(tensors[1]->dptr_))[0]);
    int32_t end = (int32_t)(((int64_t *)(tensors[2]->dptr_))[0]);
    int32_t axis = (int32_t)(((int64_t *)(tensors[3]->dptr_))[0]);
    int32_t step = 1;
    if (5 == op->num_input_) {
        step = (int32_t)(((int64_t *)(tensors[4]->dptr_))[0]);
    }

    int32_t ret = T_ERR_NO_IMPLEMENTED;

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();  // Start profiling
#endif
    ret = slice_luna(tensors[0], start, end, axis, step, tensors[op->num_input_]);  // Execute slice operation
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","Slice", total_t);  // Print profiling results
#endif
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__