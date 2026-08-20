#undef __OP__
#define __OP__ iqVar
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_VENUS
#include "./venus/iqvar.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/iqvar.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/iqvar.h"
#endif

/**
 * Forward pass implementation for Integer Quantized Variance operator
 * Computes the variance of elements in the input tensor
 * @param op: Operator structure containing variance attributes
 * @param tensors: Array of input/output tensors (input, output, workspace)
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
    iqvarAttrs *attrs = (iqvarAttrs *)((int8_t *)op + op->attr_offset_);
    tTensor *X = tensors[0];
    tTensor *Y = tensors[1];
    tTensor *workspace = tensors[2];
#if THINKER_PARAM_CHECK
    if (X == NULL || Y == NULL || workspace == NULL ||
                        X->dptr_ == 0 || Y->dptr_ == 0 || workspace->dptr_ == 0 ||
                        X->shape_.ndim_ < 3 || Y->shape_.ndim_ != X->shape_.ndim_) {
        return (T_ERR_INVALID_PARA);
    }

    if (X->dtype_ != Int8 || Y->dtype_ != Int8 ||
                        workspace->dtype_ != Int8 || workspace->byte_ != 1) {
        return (T_ERR_INVALID_DATATYPE);
    }
#endif
    int32_t axis = attrs->dims < 0 ? attrs->dims + X->shape_.ndim_ : attrs->dims;
#if THINKER_PARAM_CHECK
    if ((axis != X->shape_.ndim_ - 1 && axis != X->shape_.ndim_ - 2) ||
                        X->shape_.dims_[axis] <= 0) {
        return (T_ERR_INVALID_PARA);
    }

    for (int32_t i = 0; i < X->shape_.ndim_; ++i) {
        if (Y->shape_.dims_[i] !=
                            (i == axis ? 1 : X->shape_.dims_[i])) {
            return (T_ERR_INVALID_DATA);
        }
    }

    if (X->zero_ != 0 || Y->zero_ != 0 ||
                        !isfinite(X->scale_) || !isfinite(Y->scale_) ||
                        X->scale_ != floorf(X->scale_) || Y->scale_ != floorf(Y->scale_)) {
        return (T_ERR_INVALID_DATA);
    }

    if (X->mem_.type_ != 2 || Y->mem_.type_ != 2 ||
                        workspace->mem_.type_ != 2 || workspace->shape_.ndim_ != 1 ||
                        ((uintptr_t)workspace->dptr_ & 3U) != 0) {
        return (T_ERR_NO_SUPPORT_OP);
    }
#endif
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    // Call hardware-specific variance implementation
    THINKER_RET_CHECK(iqvar_luna(X, Y, workspace, attrs), "iqvar_luna");

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","iqVar", total_t);  
#endif
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
