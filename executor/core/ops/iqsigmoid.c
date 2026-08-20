#undef __OP__
#define __OP__ iqSigmoid
#include "core/comm/utils.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/iqsigmoid.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/iqsigmoid.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/iqsigmoid.h"
#endif

/**
 * Forward pass implementation for Integer Quantized Sigmoid operator
 * Applies sigmoid activation to input tensor
 * @param op: Operator structure
 * @param tensors: Array of input/output tensors (input, output, workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    (void)list;
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || op->num_input_ != 1 ||
                        op->num_output_ != 1 || (num_tensor != 2 && num_tensor != 3)) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    tTensor *X = tensors[0];
    tTensor *Y = tensors[1];
    tTensor *workspace = num_tensor == 3 ? tensors[2] : NULL;
#if THINKER_PARAM_CHECK
    if (X == NULL || Y == NULL || X->dptr_ == 0 || Y->dptr_ == 0 ||
                        X->shape_.ndim_ == 0 || !equalShape(&X->shape_, &Y->shape_)) {
        return (T_ERR_INVALID_PARA);
    }

    if (Y->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (X->zero_ != 0 || Y->zero_ != 0 ||
                        !isfinite(X->scale_) || !isfinite(Y->scale_) ||
                        X->scale_ != floorf(X->scale_) || Y->scale_ != 7) {
        return (T_ERR_INVALID_DATA);
    }

    if ((X->mem_.type_ != 1 && X->mem_.type_ != 2) ||
                        (Y->mem_.type_ != 1 && Y->mem_.type_ != 2)) {
        return (T_ERR_NO_SUPPORT_OP);
    }
#endif
#if THINKER_RUNTIME_CHECK
    if (workspace != NULL &&
                          (workspace->dptr_ == 0 || workspace->mem_.type_ != 2 ||
                           workspace->dtype_ != Int8 || workspace->byte_ != 1 ||
                           workspace->shape_.ndim_ != 1 ||
                           ((uintptr_t)workspace->dptr_ & 3U) != 0)) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    // Call hardware-specific sigmoid implementation
    THINKER_RET_CHECK(iqsigmoid(X, Y, workspace), "iqsigmoid");

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","iqSigmoid", total_t);  
#endif
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
