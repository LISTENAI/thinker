// Shape operator implementation

#undef __OP__
#define __OP__ Shape
#include "core/operator_register.h"
#include "thinker_status.h"

/**
 * @brief Get the shape of a tensor
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));  // Validate tensor count
    if (num_tensor != 2) return T_ERR_INVALID_PARA;             // Ensure valid number of tensors

    tTensor *X = tensors[0];  // Input tensor
    tTensor *Y = tensors[op->num_input_];  // Output tensor

    // Check if the number of dimensions matches
    if (X->shape_.ndim_ != Y->shape_.dims_[0]) {
        return T_ERR_INVALID_DATA;
    }

    int64_t *shape = (int64_t *)Y->dptr_;  // Pointer to output shape
    for (int32_t i = 0; i < X->shape_.ndim_; i++) {
        shape[i] = X->shape_.dims_[i];  // Copy dimensions
    }

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__