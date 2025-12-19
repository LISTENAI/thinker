// ReLU operator implementation

#undef __OP__
#define __OP__ Relu
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/relu.h"  // Venus backend implementation
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/relu.h"   // Arcs backend implementation
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/relu.h" // VenusA backend implementation
#endif

/**
 * @brief Execute the ReLU operation
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));  // Validate tensor count
    
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    tTensor *Workspace = NULL;

    // Get workspace tensor if available
    if (num_tensor > op->num_input_ + op->num_output_) {
        Workspace = tensors[op->num_input_ + op->num_output_];
    }

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
    ret = relu_luna(tensors[0], tensors[op->num_input_], Workspace);  // Execute ReLU operation
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__