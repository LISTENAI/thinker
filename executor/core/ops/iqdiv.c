#undef __OP__
#define __OP__ iqDiv
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "core/comm/utils.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/iqdiv.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/iqdiv.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/iqdiv.h"
#endif

/**
 * Forward pass implementation for Integer Quantized Division operator
 * Performs element-wise division of two quantized tensors
 * @param op: Operator structure containing binary operation attributes
 * @param tensors: Array of input/output tensors (tensor1, tensor2, output)
 * @param num_tensor: Total number of tensors (must equal input + output count)
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
    iqBinaryAttrs *attrs = (iqBinaryAttrs *)((int8_t *)op + op->attr_offset_);
    (void)attrs;
    tTensor *lhs = tensors[0];
    tTensor *rhs = tensors[1];
    tTensor *output = tensors[2];
#if THINKER_PARAM_CHECK
    if (lhs == NULL || rhs == NULL || output == NULL ||
                        lhs->dptr_ == 0 || rhs->dptr_ == 0 || output->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }

    if (!equalShape(&lhs->shape_, &output->shape_) ||
                        (rhs->shape_.ndim_ != 0 &&
                         !equalShape(&lhs->shape_, &rhs->shape_))) {
        return (T_ERR_INVALID_DATA);
    }

    if (lhs->zero_ != 0 || rhs->zero_ != 0 || output->zero_ != 0 ||
                        !isfinite(lhs->scale_) || !isfinite(rhs->scale_) ||
                        !isfinite(output->scale_) || floorf(lhs->scale_) != lhs->scale_ ||
                        floorf(rhs->scale_) != rhs->scale_ ||
                        floorf(output->scale_) != output->scale_) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    // Call hardware-specific division implementation
    THINKER_RET_CHECK(iqdiv_luna(lhs, rhs, output), "iqdiv_luna");

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","iqDiv", total_t);  
#endif
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
