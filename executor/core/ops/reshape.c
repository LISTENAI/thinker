// Reshape operator implementation

#undef __OP__
#define __OP__ Reshape
#include "core/operator_register.h"
#include "core/comm/utils.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/reshape.h"  // Venus backend implementation
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/reshape.h"   // Arcs backend implementation
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/reshape.h" // VenusA backend implementation
#endif

/**
 * @brief Execute the Reshape operation
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    (void)list;
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || op->num_input_ != 2 ||
                        op->num_output_ != 1 || num_tensor != 3) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    tTensor *input = tensors[0];
    tTensor *shape = tensors[1];
    tTensor *output = tensors[2];
#if THINKER_PARAM_CHECK
    if (input == NULL || shape == NULL || output == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if (input->shape_.ndim_ > 7) {
        return (T_ERR_INVALID_PARA);
    }

    if ((shape->dtype_ != Int32 && shape->dtype_ != Int64) ||
                        shape->shape_.ndim_ != 1 || shape->shape_.dims_[0] > 7 ||
                        output->shape_.ndim_ != shape->shape_.dims_[0] ||
                        (shape->shape_.dims_[0] > 0 && shape->dptr_ == 0)) {
        return (T_ERR_INVALID_PARA);
    }

    if (input->dtype_ != output->dtype_ ||
                        input->byte_ != output->byte_) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (input->layout_ != output->layout_ ||
                        getTensorDataSize(input) != getTensorDataSize(output)) {
        return (T_ERR_INVALID_DATA);
    }

    if (getTensorDataSize(input) > 0 &&
                        (input->dptr_ == 0 || output->dptr_ == 0)) {
        return (T_ERR_INVALID_PARA);
    }
#endif

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();  // Start profiling
#endif
    THINKER_RET_CHECK(reshape_luna(input, output), "reshape_luna");  // Execute Reshape operation
    
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (", "reshape", total_t);  // Print profiling results
#endif
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
