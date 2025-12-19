// SparifyFFNInt operator implementation

#undef __OP__
#define __OP__ SparifyFFNInt
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_ARCS
#include "./arcs/sparifyffnint.h"  // Arcs backend implementation
#endif

/**
 * @brief Execute the SparifyFFNInt operation
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));  // Validate tensor count

    SparifyFFNIntAttrs *attrs = (SparifyFFNIntAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    // Input and output tensors
    tTensor *input = tensors[0];
    tTensor *weight1 = tensors[1];
    tTensor *weight2 = tensors[2];
    tTensor *weight3 = tensors[3];
    tTensor *bias1 = tensors[4];
    tTensor *bias2 = tensors[5];
    tTensor *bias3 = tensors[6];
    tTensor *output = tensors[op->num_input_];
    tTensor *workspace = NULL;

#if THINKER_USE_ARCS
#if THINKER_PROFILE
    uint64_t start_t = tick_count();  // Start profiling
#endif
    if (num_tensor == op->num_input_ + op->num_output_ + 1) {
        workspace = tensors[op->num_input_ + op->num_output_];
    }
    ret = sparifyffnint_luna(input, weight1, bias1, weight2, bias2, weight3, bias3, workspace, output, attrs);
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","SparifyFFNInt", total_t);  // Print profiling results
#endif
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__