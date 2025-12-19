#undef __OP__
#define __OP__ LogSoftmax
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/logsoftmax.h"
#endif

/**
 * Forward pass implementation for LogSoftmax operator
 * Applies log-softmax activation to input tensor
 * @param op: Operator structure containing log-softmax attributes
 * @param tensors: Array of input/output tensors (input, output)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Validate tensor count
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
    
    // Get log-softmax attributes
    LogSoftmaxAttrs *attrs = (LogSoftmaxAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
#if THINKER_USE_VENUS
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    // Call hardware-specific log-softmax implementation
    ret = logsoftmax_luna(tensors[0], tensors[op->num_input_], attrs);

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","LogSoftmax", total_t);  
#endif
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__