#undef __OP__
#define __OP__ LogSoftmaxInt
#include <math.h>
#include <stdint.h>
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/logsoftmaxint.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/logsoftmaxint.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/logsoftmaxint.h"
#endif

/**
 * Forward pass implementation for Integer Quantized LogSoftmax operator
 * Applies log-softmax activation to quantized input tensor
 * @param op: Operator structure containing log-softmax attributes
 * @param tensors: Array of input/output tensors (input, output, optional workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    (void)list;
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || op->num_input_ != 1 ||
                        op->num_output_ != 1 || num_tensor != 3) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    LogSoftmaxIntAttrs *attrs = (LogSoftmaxIntAttrs *)((int8_t *)op + op->attr_offset_);
    tTensor *input = tensors[0];
    tTensor *output = tensors[1];
    tTensor *workspace = tensors[2];
#if THINKER_PARAM_CHECK
    if (input == NULL || output == NULL || workspace == NULL ||
                        input->dptr_ == 0 || output->dptr_ == 0 ||
                        workspace->dptr_ == 0 || input->shape_.ndim_ == 0 ||
                        !equalShape(&input->shape_, &output->shape_)) {
        return (T_ERR_INVALID_PARA);
    }
#endif
#if THINKER_USE_VENUSA
    #if THINKER_PARAM_CHECK
    if ((input->dtype_ != Int8 && input->dtype_ != Int16 &&
                         input->dtype_ != Int32) ||
                        (output->dtype_ != Int8 && output->dtype_ != Int16 &&
                         output->dtype_ != Int32)) {
        return (T_ERR_INVALID_DATATYPE);
    }
#endif
#else
    #if THINKER_PARAM_CHECK
    if (input->dtype_ != Int8 || output->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }
#endif
#endif
    int32_t axis = attrs->axis < 0 ? attrs->axis + input->shape_.ndim_ : attrs->axis;
    #if THINKER_PARAM_CHECK
    if (axis != (int32_t)input->shape_.ndim_ - 1 ||
                        input->shape_.dims_[axis] == 0 ||
                        input->shape_.dims_[axis] > 2048 || input->zero_ != 0 ||
                        output->zero_ != 0 || !isfinite(input->scale_) ||
                        !isfinite(output->scale_) || input->scale_ != floorf(input->scale_) ||
                        output->scale_ != floorf(output->scale_)) {
        return (T_ERR_INVALID_DATA);
    }
#endif
    size_t elements = 1;
    for (int32_t i = 0; i < input->shape_.ndim_; ++i) {
        #if THINKER_PARAM_CHECK
        if (input->shape_.dims_[i] == 0 ||
                            elements > INT32_MAX / input->shape_.dims_[i]) {
            return (T_ERR_INVALID_DATA);
        }
#endif
        elements *= input->shape_.dims_[i];
    }
    #if THINKER_RUNTIME_CHECK
    if (workspace->dtype_ != Int8 || workspace->byte_ != 1 ||
                          workspace->shape_.ndim_ != 1 || workspace->mem_.type_ != 2 ||
                          ((uintptr_t)workspace->dptr_ & 3U) != 0 ||
                          workspace->dptr_ == input->dptr_ ||
                          workspace->dptr_ == output->dptr_) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif
#if THINKER_USE_VENUS || THINKER_USE_VENUSA
    #if THINKER_PARAM_CHECK
    if (input->mem_.type_ != 2 || output->mem_.type_ != 2) {
        return (T_ERR_INVALID_PLATFROM);
    }
#endif
#endif
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    // Call hardware-specific log-softmax implementation
    THINKER_RET_CHECK(logsoftmaxint_luna(input, output, workspace, attrs), "logsoftmaxint_luna");

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","LogSoftmaxInt", total_t);  
#endif
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
