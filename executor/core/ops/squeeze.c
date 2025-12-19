// Squeeze operator implementation

#undef __OP__
#define __OP__ Squeeze
#include "core/operator_register.h"
#include "thinker_status.h"

/**
 * @brief Execute the Squeeze operation
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));  // Validate tensor count

    tTensor *X = tensors[0];  // Input tensor
    tTensor *Y = tensors[op->num_input_];  // Output tensor

    if (num_tensor != 2) {
        return T_ERR_INVALID_PARA;  // Invalid number of tensors
    }

    if (X->dptr_ != Y->dptr_) {
        return T_ERR_INVALID_DATA;  // Input and output data pointers must be the same
    }

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__