#undef __OP__
#define __OP__ MaxPool
#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/maxpool.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/maxpool.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/maxpool.h"
#endif

/**
 * Forward pass implementation for Max Pooling operator
 * Applies max pooling operation to input tensor
 * @param op: Operator structure containing pooling attributes
 * @param tensors: Array of input/output tensors (input, output, temporary workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Validate tensor count
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
    
    // Get pooling attributes
    PoolAttrs *attrs = (PoolAttrs *)((int8_t *)op + op->attr_offset_);
    
    // Get input, output, and temporary workspace tensors
    tTensor *X = ((tTensor **)tensors)[0];
    tTensor *Y = ((tTensor **)tensors)[op->num_input_];
    tTensor *Temp = ((tTensor **)tensors)[op->num_input_ + 1];
    int32_t ret = T_SUCCESS;
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    // Call hardware-specific max pooling implementation
    ret = maxpool_luna(X, Y, Temp, attrs);

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","MaxPool", total_t);
#endif
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__