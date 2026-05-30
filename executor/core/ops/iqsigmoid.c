#undef __OP__
#define __OP__ iqSigmoid
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/iqsigmoid.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/iqsigmoid.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/iqsigmoid.h"
#endif

/**
 * Forward pass implementation for Integer Quantized Sigmoid operator
 * Applies sigmoid activation to input tensor
 * @param op: Operator structure
 * @param tensors: Array of input/output tensors (input, output, workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Validate tensor count
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));

    // Get input, output, and workspace tensors
    tTensor *X = tensors[0];
    tTensor *Y = tensors[1];
    tTensor *workspace = (num_tensor > (op->num_input_ + op->num_output_)) ? tensors[op->num_input_ + op->num_output_] : NULL;

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    // Call hardware-specific sigmoid implementation
    THINKER_RET_CHECK(iqsigmoid(X, Y, workspace), "iqsigmoid");

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","iqSigmoid", total_t);  
#endif
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__