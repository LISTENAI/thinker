// GELU operator implementation

#undef __OP__
#define __OP__ QGelu
#include "core/operator_register.h"
#include "thinker_status.h"
#include "core/comm/utils.h"

#ifdef THINKER_USE_VENUSA
#include "./venusA/gelu.h"  // VenusA backend implementation
#endif

/**
 * @brief Execute the GELU operation
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));

    tTensor *workspace = NULL;
    if (num_tensor > op->num_input_ + op->num_output_) {
        workspace = tensors[op->num_input_ + op->num_output_];
    }

    // Debug info
    tTensor *X = tensors[0];
    tTensor *Y = tensors[op->num_input_];
    
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

#ifdef THINKER_USE_VENUSA
    THINKER_RET_CHECK(gelu_luna(tensors[0], tensors[op->num_input_], workspace), "gelu_luna");
#endif
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","gelu", total_t);  
#endif
    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__