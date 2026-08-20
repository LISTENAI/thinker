#undef __OP__
#define __OP__ iqCat
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/concat.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/concat.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/concat.h"
#endif

/**
 * Forward pass implementation for Concat operator
 * @param op: Operator structure containing concat attributes
 * @param tensors: Array of input/output tensors
 * @param num_tensor: Total number of tensors
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if (op->num_input_ < 2 || op->num_output_ != 1 ||
                        num_tensor < op->num_input_ + op->num_output_) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    tTensor *output = tensors[op->num_input_];
#if THINKER_PARAM_CHECK
    if (tensors[0] == NULL || output == NULL ||
                        tensors[0]->shape_.ndim_ == 0 ||
                        output->shape_.ndim_ != tensors[0]->shape_.ndim_ ||
                        (output->dtype_ != Int8 && output->dtype_ != Int16 &&
                         output->dtype_ != Int32)) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
    // Get concat attributes
    iqCatAttrs *attr = (iqCatAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t axis = attr->axis;
    
    // Handle negative axis indexing
    if (axis < 0) {
        axis += tensors[0]->shape_.ndim_;
    }
#if THINKER_PARAM_CHECK
    if (axis < 0 || axis >= tensors[0]->shape_.ndim_) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    int64_t output_axis = 0;
    for (int32_t i = 0; i < op->num_input_; ++i) {
        tTensor *input = tensors[i];
#if THINKER_PARAM_CHECK
        if (input == NULL ||
                            input->shape_.ndim_ != output->shape_.ndim_ ||
                            input->dtype_ != output->dtype_) {
            return (T_ERR_INVALID_DATATYPE);
        }
#endif
#if THINKER_RUNTIME_CHECK
        if (input->dptr_ == 0 || input->dptr_ == output->dptr_) {
            return (T_ERR_INVALID_PARA);
        }
#endif
        #if THINKER_RUNTIME_CHECK
        if (getTensorSize(input) > INT32_MAX) {
            return (T_ERR_INVALID_PARA);
        }
#endif
        for (int32_t dim = 0; dim < output->shape_.ndim_; ++dim) {
#if THINKER_RUNTIME_CHECK
            if (input->shape_.dims_[dim] < 0 ||
                                  (dim != axis &&
                                   input->shape_.dims_[dim] !=
                                       output->shape_.dims_[dim])) {
                return (T_ERR_INVALID_PARA);
            }
#endif
        }
        output_axis += input->shape_.dims_[axis];
    }
#if THINKER_RUNTIME_CHECK
    if (output->dptr_ == 0 ||
                          output_axis > INT32_MAX ||
                          output->shape_.dims_[axis] != output_axis ||
                          getTensorSize(output) > INT32_MAX) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
    // Check for workspace tensor
    tTensor *workspace = NULL;
    if (num_tensor > op->num_input_ + op->num_output_){
        workspace = ((tTensor**)tensors)[op->num_input_ + op->num_output_];
#if THINKER_RUNTIME_CHECK
        if (workspace == NULL || workspace->dptr_ == 0 ||
                              workspace->mem_.type_ != 2 ||
                              workspace->shape_.ndim_ != 1) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
    }

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif
    // Call hardware-specific concat implementation
    THINKER_RET_CHECK(concat_luna(tensors, axis, op->num_input_, workspace, output), "concat_luna");
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","Concat", total_t);  
#endif
#else
    return T_ERR_NO_SUPPORT_OP;
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
