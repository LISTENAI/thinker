// PRelu operator implementation

#undef __OP__
#define __OP__ PRelu
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/prelu.h"  // Venus backend implementation
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/prelu.h"   // Arcs backend implementation
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/prelu.h" // VenusA backend implementation
#endif

/**
 * @brief Execute the PRelu operation
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || op->num_input_ != 1 ||
                        op->num_output_ != 1 || num_tensor != 2) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    PreluAttrs *attrs = (PreluAttrs *)((int8_t *)op + op->attr_offset_);
    tTensor *X = tensors[0];
    tTensor *Y = tensors[1];
#if THINKER_PARAM_CHECK
    if (X == NULL || Y == NULL || X->dptr_ == 0 || Y->dptr_ == 0 ||
                        !equalShape(&X->shape_, &Y->shape_) || X->dtype_ != Y->dtype_) {
        return (T_ERR_INVALID_DATA);
    }
#endif

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();  // Start profiling
#endif
    THINKER_RET_CHECK(prelu_luna(X, Y, attrs), "prelu_luna");
    
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","prelu", total_t);  // Print profiling results
#endif
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
