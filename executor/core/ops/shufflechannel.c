// ShuffleChannel operator implementation

#undef __OP__
#define __OP__ ShuffleChannel
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/shufflechannel.h"  // Venus backend implementation
#endif


/**
 * @brief Execute the ShuffleChannel operation
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));  // Validate tensor count

    ShuffleChannelAttrs *attrs = (ShuffleChannelAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t ret = T_ERR_NO_IMPLEMENTED;

#if THINKER_USE_VENUS
    ret = shufflechannel_venus(tensors[0], tensors[op->num_input_], attrs);  // Execute shufflechannel operation
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__