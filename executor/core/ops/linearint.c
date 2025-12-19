#undef __OP__
#define __OP__ LinearInt
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_VENUS
#include "./venus/linearint.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/linearint.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/linearint.h"
#endif

/**
 * Forward pass implementation for Integer Quantized Linear operator
 * Performs linear transformation (matrix multiplication) on input tensor
 * @param op: Operator structure containing linear transformation attributes
 * @param tensors: Array of input/output tensors (input, weight, optional bias, output, optional workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list for weight data handling
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Validate tensor count
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
    
    // Get linear transformation attributes
    LinearIntAttrs *attrs = (LinearIntAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
    // Get input, weight, output tensors
    tTensor *input = tensors[0];
    tTensor weight_tmp;
    memcpy(&weight_tmp, tensors[1], sizeof(tTensor));
    tTensor *weight = &weight_tmp;
    tTensor *bias = NULL;
    tTensor *output = tensors[op->num_input_];
    tTensor *workspace = NULL;
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
    if (list->total_ != 0)
        getWeightData(list, 0);
#endif

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    if (list->total_ > 0) {
        tTensor *dma_buffer = NULL;
        if (num_tensor == op->num_input_ + op->num_output_ + 1) {
            dma_buffer = ((tTensor **)tensors)[op->num_input_ + op->num_output_];
            weight->dptr_ = (addr_type)dma_buffer->dptr_;
        } else if (num_tensor == op->num_input_ + op->num_output_ + 2) {
            workspace = ((tTensor **)tensors)[op->num_input_ + op->num_output_];
            dma_buffer = ((tTensor **)tensors)[op->num_input_ + op->num_output_ + 1];
            weight->dptr_ = (addr_type)dma_buffer->dptr_;
        }
        
        if (3 == op->num_input_) {
            bias = ((tTensor **)tensors)[op->num_input_ - 1];
            bias->scale_ = input->scale_ + weight->scale_;
            int32_t size = getShapeSize(&(weight->shape_)) * weight->byte_;
            bias->dptr_ = (addr_type)((int8_t *)dma_buffer->dptr_ + ALIGN16(size));
        }
        
        ret = linearint_luna(input, weight, bias, attrs, workspace, output);
    } else {
        if (3 == op->num_input_) {
            bias = ((tTensor **)tensors)[op->num_input_ - 1];
            bias->scale_ = input->scale_ + weight->scale_;
        }
        if (num_tensor == op->num_input_ + op->num_output_ + 1) {
            workspace = ((tTensor **)tensors)[op->num_input_ + op->num_output_];
        }
        ret = linearint_luna(input, weight, bias, attrs, workspace, output);
    }

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","LinearInt", total_t);  
#endif
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__