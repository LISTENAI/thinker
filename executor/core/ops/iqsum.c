#undef __OP__
#define __OP__ iqSum
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/iqsum.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/iqsum.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/iqsum.h"
#endif

/**
 * Forward pass implementation for Integer Quantized Sum operator
 * Computes the sum of elements in the input tensor
 * @param op: Operator structure containing sum attributes
 * @param tensors: Array of input/output tensors (input, output, workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    (void)list;
#ifdef THINKER_USE_VENUSA
    const int32_t expected_tensors = 2;
#else
    const int32_t expected_tensors = 3;
#endif
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || op->num_input_ != 1 ||
                        op->num_output_ != 1 || num_tensor != expected_tensors) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    iqSumAttrs *attrs = (iqSumAttrs *)((int8_t *)op + op->attr_offset_);
    tTensor *input = tensors[0];
    tTensor *output = tensors[1];
    tTensor *Temp = expected_tensors == 3 ? tensors[2] : NULL;
#if THINKER_PARAM_CHECK
    if (input == NULL || output == NULL || input->dptr_ == 0 ||
                        output->dptr_ == 0 || input->shape_.ndim_ == 0 ||
                        input->shape_.ndim_ != output->shape_.ndim_) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    int32_t axis = attrs->axis < 0 ? attrs->axis + input->shape_.ndim_ : attrs->axis;
#if THINKER_PARAM_CHECK
    if (axis != input->shape_.ndim_ - 1 ||
                        input->shape_.dims_[axis] <= 0) {
        return (T_ERR_INVALID_PARA);
    }

    for (int32_t i = 0; i < input->shape_.ndim_; ++i) {
        if (output->shape_.dims_[i] !=
                            (i == axis ? 1 : input->shape_.dims_[i])) {
            return (T_ERR_INVALID_DATA);
        }
    }

    if (input->zero_ != 0 || output->zero_ != 0 ||
                        !isfinite(input->scale_) || !isfinite(output->scale_) ||
                        input->scale_ != floorf(input->scale_) ||
                        output->scale_ != floorf(output->scale_) ||
                        input->scale_ - output->scale_ < 0 ||
                        input->scale_ - output->scale_ > 63) {
        return (T_ERR_INVALID_DATA);
    }

    if (input->mem_.type_ != 2 || output->mem_.type_ != 2) {
        return (T_ERR_NO_SUPPORT_OP);
    }
#endif
#ifndef THINKER_USE_VENUSA
#if THINKER_RUNTIME_CHECK
    if (Temp == NULL || Temp->dptr_ == 0 ||
                          Temp->mem_.type_ != 2 || Temp->dtype_ != Int8 ||
                          Temp->byte_ != 1 || Temp->shape_.ndim_ != 1 ||
                          ((uintptr_t)Temp->dptr_ & 3U) != 0 ||
                          getTensorDataSize(Temp) < getTensorSize(output) * sizeof(int32_t)) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif
#endif
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    // Call hardware-specific sum implementation
    THINKER_RET_CHECK(iqsum_luna(input, Temp, output, attrs), "iqsum_luna");

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","iqSum", total_t);  
#endif
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
