#undef __OP__
#define __OP__ Gather
#include "core/operator_attrs.h"
#include "core/comm/utils.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/gather.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/gather.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/gather.h"
#endif

/**
 * Forward pass implementation for Gather operator
 * Gathers slices from input tensor along a given axis using indices
 * @param op: Operator structure containing gather attributes
 * @param tensors: Array of input tensors (input, indices, output)
 * @param num_tensor: Total number of tensors (must be 3)
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    (void)list;
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || op->num_input_ != 2 ||
                        op->num_output_ != 1 || num_tensor != 3) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
    // Get gather attributes
    GatherAttrs *attr = (GatherAttrs *)((int8_t *)op + op->attr_offset_);
    
    tTensor *input = tensors[0];
    tTensor *indices = tensors[1];
    tTensor *output = tensors[2];
#if THINKER_PARAM_CHECK
    if (input == NULL || indices == NULL || output == NULL ||
                        input->shape_.ndim_ == 0 || input->shape_.ndim_ > 7 ||
                        indices->shape_.ndim_ > 7) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    int32_t axis = attr->axis;
    if (axis < 0) axis += input->shape_.ndim_;
    #if THINKER_PARAM_CHECK
    if (axis < 0 || axis >= input->shape_.ndim_ ||
                        input->shape_.ndim_ - 1 + indices->shape_.ndim_ > 7) {
        return (T_ERR_INVALID_PARA);
    }

    if (indices->dtype_ != Int32 && indices->dtype_ != Int64) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (input->layout_ != output->layout_ ||
                        output->scale_ != input->scale_ ||
                        output->shape_.ndim_ !=
                            input->shape_.ndim_ - 1 + indices->shape_.ndim_) {
        return (T_ERR_INVALID_DATA);
    }
#endif
#if THINKER_USE_ARCS
    #if THINKER_PARAM_CHECK
    if ((input->dtype_ == Int4 && output->dtype_ != Int8) ||
                        (input->dtype_ != Int4 &&
                         (output->dtype_ != input->dtype_ ||
                          output->byte_ != input->byte_))) {
        return (T_ERR_INVALID_DATATYPE);
    }
#endif
#else
    #if THINKER_PARAM_CHECK
    if (input->dtype_ == Int4 || output->dtype_ != input->dtype_ ||
                        output->byte_ != input->byte_) {
        return (T_ERR_INVALID_DATATYPE);
    }
#endif
#endif
    for (int32_t i = 0; i < axis; ++i) {
#if THINKER_PARAM_CHECK
        if (output->shape_.dims_[i] != input->shape_.dims_[i]) {
            return (T_ERR_INVALID_DATA);
        }
#endif
    }
    for (int32_t i = 0; i < indices->shape_.ndim_; ++i) {
#if THINKER_PARAM_CHECK
        if (output->shape_.dims_[axis + i] !=
                                indices->shape_.dims_[i]) {
            return (T_ERR_INVALID_DATA);
        }
#endif
    }
    for (int32_t i = axis + 1; i < input->shape_.ndim_; ++i) {
#if THINKER_PARAM_CHECK
        if (output->shape_.dims_[i - 1 + indices->shape_.ndim_] !=
                                input->shape_.dims_[i]) {
            return (T_ERR_INVALID_DATA);
        }
#endif
    }
    size_t indices_count = getTensorSize(indices);
    #if THINKER_PARAM_CHECK
    if (indices_count > INT32_MAX ||
                        getTensorSize(input) > INT32_MAX ||
                        getTensorSize(output) > INT32_MAX) {
        return (T_ERR_INVALID_DATA);
    }
#endif
    #if THINKER_PARAM_CHECK
    if (indices_count > 0 && indices->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    #if THINKER_PARAM_CHECK
    if (getTensorDataSize(output) > 0 &&
                        (input->dptr_ == 0 || output->dptr_ == 0)) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif
    
    // Call hardware-specific gather implementation
    THINKER_RET_CHECK(gather_luna(input, indices, output, attr), "gather_luna");
    
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","Gather", total_t);  
#endif
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
