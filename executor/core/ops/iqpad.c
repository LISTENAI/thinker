#undef __OP__
#define __OP__ iqPad
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "core/comm/utils.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/iqpad.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/iqpad.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/iqpad.h"
#endif

/**
 * Forward pass implementation for Integer Quantized Padding operator
 * Performs padding on input tensor with specified constants
 * @param op: Operator structure containing padding attributes
 * @param tensors: Array of input/output tensors (input, pads, constants, workspace, output)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    (void)list;
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || op->num_input_ != 3 ||
                        op->num_output_ != 1 || (num_tensor != 4 && num_tensor != 5)) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    iqPadAttrs *attr = (iqPadAttrs *)((int8_t *)op + op->attr_offset_);
    tTensor *X = tensors[0];
    tTensor *pads = tensors[1];
    tTensor *constants = tensors[2];
    tTensor *Y = tensors[3];
    tTensor *workspace = num_tensor == 5 ? tensors[4] : NULL;
#if THINKER_PARAM_CHECK
    if (X == NULL || pads == NULL || constants == NULL || Y == NULL ||
                        X->dptr_ == 0 || pads->dptr_ == 0 || constants->dptr_ == 0 ||
                        Y->dptr_ == 0 || attr->mode < 0 || attr->mode > 2) {
        return (T_ERR_INVALID_PARA);
    }

    if (X->dtype_ != Int8 || Y->dtype_ != Int8 ||
                        pads->dtype_ != Int64 || pads->shape_.ndim_ != 1 ||
                        constants->dtype_ != Int8 || getTensorSize(constants) != 1) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (X->zero_ != 0 || Y->zero_ != 0 ||
                        !isfinite(X->scale_) || !isfinite(Y->scale_) ||
                        X->scale_ != Y->scale_) {
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
                           ((uintptr_t)workspace->dptr_ & 3U) != 0)) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    // Call hardware-specific padding implementation
    THINKER_RET_CHECK(iqpad_luna(X, pads, constants, workspace, Y, attr), "iqpad_luna");

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","iqPad", total_t);  
#endif
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
