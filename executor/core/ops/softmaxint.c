// SoftmaxInt operator implementation

#undef __OP__
#define __OP__ SoftmaxInt
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/softmaxint.h"  // Venus backend implementation
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/softmaxint.h"   // Arcs backend implementation
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/softmaxint.h" // VenusA backend implementation
#endif

/**
 * @brief Execute the SoftmaxInt operation
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));  // Validate tensor count

    SoftmaxIntAttrs *attr = (SoftmaxIntAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    tTensor *workspace = NULL;
    if (num_tensor > op->num_input_ + op->num_output_) {
        workspace = tensors[num_tensor - 1];
    }

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();  // Start profiling
#endif
    ret = softmaxint_luna(tensors[0], tensors[op->num_input_], workspace, attr);
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","SoftmaxInt", total_t);  // Print profiling results
#endif
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__