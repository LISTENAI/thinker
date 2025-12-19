#undef __OP__
#define __OP__ iqPad
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/iqpad.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/iqpad.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/iqpad.h"
#endif

/**
 * Forward pass implementation for Integer Quantized Padding operator
 * Performs padding on input tensor with specified constants
 * @param op: Operator structure containing padding attributes
 * @param tensors: Array of input/output tensors (input, pads, constants, workspace, output)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Validate tensor count
    CHECK_GT(num_tensor, (op->num_input_ + op->num_output_));
    
    // Get padding attributes
    iqPadAttrs *attr = (iqPadAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    // Get input, pads, constants, and output tensors
    tTensor *X = tensors[0];
    tTensor *pads = tensors[1];
    tTensor *constants = tensors[2];
    tTensor *workspace = ((tTensor**)tensors)[op->num_input_ + op->num_output_];
    tTensor *Y = tensors[op->num_input_];
    
    // Call hardware-specific padding implementation
    ret = iqpad_luna(X, pads, constants, workspace, Y, attr);

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","iqPad", total_t);  
#endif
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__