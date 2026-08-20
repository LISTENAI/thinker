// ReLU operator implementation

#undef __OP__
#define __OP__ Relu
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
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
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || op->num_input_ != 1 ||
                        op->num_output_ != 1 || num_tensor < 2 || num_tensor > 3) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    tTensor *X = tensors[0];
    tTensor *Y = tensors[1];
#if THINKER_PARAM_CHECK
    if (X == NULL || Y == NULL || X->dptr_ == 0 || Y->dptr_ == 0 ||
                        !equalShape(&X->shape_, &Y->shape_)) {
        return (T_ERR_INVALID_DATA);
    }
#endif

    tTensor *Workspace = NULL;

    // Get workspace tensor if available
    if (num_tensor > op->num_input_ + op->num_output_) {
        Workspace = tensors[op->num_input_ + op->num_output_];
#if THINKER_RUNTIME_CHECK
        if (Workspace == NULL || Workspace->dptr_ == 0 ||
                              Workspace->mem_.type_ != 2 || Workspace->dtype_ != Int8) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
    }

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
    THINKER_RET_CHECK(relu_luna(X, Y, Workspace), "relu_luna");
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
