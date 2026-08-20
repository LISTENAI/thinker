#undef __OP__
#define __OP__ GluInt
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "core/comm/utils.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/gluint.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/gluint.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/gluint.h"
#endif // THINKER_USE_VENUSA

/**
 * Forward pass implementation for GLU (Gated Linear Unit) Integer operator
 * Applies gated linear unit activation to input tensor
 * @param op: Operator structure containing GLU attributes
 * @param tensors: Array of input/output tensors (input, output, workspace)
 * @param num_tensor: Total number of tensors (must equal input + output + workspace count)
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
    
    GluIntAttrs *attrs = (GluIntAttrs *)((int8_t *)op + op->attr_offset_);
    tTensor *input = tensors[0];
    tTensor *output = tensors[1];
    tTensor *workspace = tensors[2];
#if THINKER_PARAM_CHECK
    if (input == NULL || output == NULL || workspace == NULL ||
                        input->shape_.ndim_ == 0 || input->shape_.ndim_ > 7 ||
                        output->shape_.ndim_ != input->shape_.ndim_) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    int32_t axis = attrs->axis;
    if (axis < 0) axis += input->shape_.ndim_;
#if THINKER_PARAM_CHECK
    if (axis < 0 || axis >= input->shape_.ndim_ ||
                        input->shape_.dims_[axis] == 0 ||
                        (input->shape_.dims_[axis] & 1U)) {
        return (T_ERR_INVALID_PARA);
    }

    for (int32_t i = 0; i < input->shape_.ndim_; ++i) {
        uint32_t expected = i == axis ? input->shape_.dims_[i] / 2
                                      : input->shape_.dims_[i];
        if (output->shape_.dims_[i] != expected) {
            return (T_ERR_INVALID_DATA);
        }

    }

    if (input->dtype_ != Int8 ||
                        (output->dtype_ != Int8 && output->dtype_ != Int16 &&
                         output->dtype_ != Int32)) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (input->layout_ != output->layout_ ||
                        input->zero_ != 0 || output->zero_ != 0) {
        return (T_ERR_INVALID_DATA);
    }

    if (getTensorSize(output) == 0 ||
                        getTensorSize(output) > INT32_MAX) {
        return (T_ERR_INVALID_DATA);
    }

    if (input->dptr_ == 0 || output->dptr_ == 0 ||
                        workspace->dptr_ == 0 || input->dptr_ == output->dptr_ ||
                        workspace->dptr_ == input->dptr_ ||
                        workspace->dptr_ == output->dptr_) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif
    
    // Call hardware-specific GLU implementation
    THINKER_RET_CHECK(gluint_luna(input, output, workspace, attrs), "gluint_luna");
    
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","GluInt", total_t);
#endif

#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
