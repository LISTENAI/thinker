#undef __OP__
#define __OP__ iqMul
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "core/comm/utils.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/iqmul.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/iqmul.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/iqmul.h"
#endif

/**
 * Forward pass implementation for Integer Quantized Multiplication operator
 * Performs element-wise multiplication of two quantized tensors
 * @param op: Operator structure containing binary operation attributes
 * @param tensors: Array of input/output tensors (tensor1, tensor2, output, workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    (void)list;
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || op->num_input_ != 2 ||
                        op->num_output_ != 1 || (num_tensor != 3 && num_tensor != 4)) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    iqBinaryAttrs *attrs = (iqBinaryAttrs *)((int8_t *)op + op->attr_offset_);
    tTensor *lhs = tensors[0];
    tTensor *rhs = tensors[1];
    tTensor *output = tensors[2];
    tTensor *workspace = num_tensor == 4 ? tensors[3] : NULL;
#if THINKER_PARAM_CHECK
    if (lhs == NULL || rhs == NULL || output == NULL ||
                        lhs->dptr_ == 0 || rhs->dptr_ == 0 || output->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }

    if (!equalShape(&lhs->shape_, &output->shape_)) {
        return (T_ERR_INVALID_DATA);
    }
#endif
    int32_t scalar = rhs->shape_.ndim_ == 0;
    int32_t broadcast = lhs->shape_.ndim_ == 4 && rhs->shape_.ndim_ == 4 &&
                        lhs->shape_.dims_[1] == rhs->shape_.dims_[1] &&
                        rhs->shape_.dims_[2] == 1 && rhs->shape_.dims_[3] == 1 &&
                        (rhs->shape_.dims_[0] == 1 ||
                         rhs->shape_.dims_[0] == lhs->shape_.dims_[0]);
#if THINKER_PARAM_CHECK
    if (!scalar && !broadcast &&
                        !equalShape(&lhs->shape_, &rhs->shape_)) {
        return (T_ERR_INVALID_DATA);
    }

    if (lhs->zero_ != 0 || rhs->zero_ != 0 || output->zero_ != 0 ||
                        !isfinite(lhs->scale_) || !isfinite(rhs->scale_) ||
                        !isfinite(output->scale_) || floorf(lhs->scale_) != lhs->scale_ ||
                        floorf(rhs->scale_) != rhs->scale_ ||
                        floorf(output->scale_) != output->scale_) {
        return (T_ERR_INVALID_PARA);
    }

    if ((lhs->mem_.type_ != 1 && lhs->mem_.type_ != 2) ||
                        (!scalar && rhs->mem_.type_ != 1 && rhs->mem_.type_ != 2) ||
                        (output->mem_.type_ != 1 && output->mem_.type_ != 2)) {
        return (T_ERR_NO_SUPPORT_OP);
    }
#endif
#if THINKER_RUNTIME_CHECK
    if (workspace != NULL &&
                          (workspace->dptr_ == 0 || workspace->mem_.type_ != 2 ||
                           workspace->dtype_ != Int8 || workspace->byte_ != 1 ||
                           ((uintptr_t)workspace->dptr_ & 3U) != 0)) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    // Call hardware-specific multiplication implementation
    THINKER_RET_CHECK(iqmul_luna(lhs, rhs, output, workspace, attrs), "iqmul_luna");

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","iqMul", total_t);  
#endif
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
