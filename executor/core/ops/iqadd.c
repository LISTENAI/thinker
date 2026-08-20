#undef __OP__
#define __OP__ iqAdd
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "core/comm/utils.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/iqadd.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/iqadd.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/iqadd.h"
#endif

/**
 * Forward pass implementation for Integer Quantized Addition operator
 * Performs element-wise addition of two quantized tensors
 * @param op: Operator structure containing binary operation attributes
 * @param tensors: Array of input/output tensors (tensor1, tensor2, output, optional workspace)
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
    tTensor *x1 = tensors[0];
    tTensor *x2 = tensors[1];
    tTensor *y = tensors[2];
    tTensor *workspace = num_tensor == 4 ? tensors[3] : NULL;
#if THINKER_PARAM_CHECK
    if (x1 == NULL || x2 == NULL || y == NULL ||
                        x1->dptr_ == 0 || x2->dptr_ == 0 || y->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    int32_t scalar_rhs = x2->dtype_ == Float32 && x2->shape_.ndim_ == 0;
#if THINKER_PARAM_CHECK
    if (!scalar_rhs &&
                        (!equalShape(&x1->shape_, &x2->shape_) ||
                         !equalShape(&x1->shape_, &y->shape_))) {
        return (T_ERR_INVALID_DATA);
    }

    if (scalar_rhs ? y->dtype_ != Int8 :
                        (x1->dtype_ != x2->dtype_ || x1->dtype_ != y->dtype_)) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (x1->zero_ != 0 || x2->zero_ != 0 || y->zero_ != 0 ||
                        !isfinite(x1->scale_) || !isfinite(x2->scale_) ||
                        !isfinite(y->scale_) || floorf(x1->scale_) != x1->scale_ ||
                        floorf(y->scale_) != y->scale_ ||
                        (!scalar_rhs && floorf(x2->scale_) != x2->scale_)) {
        return (T_ERR_INVALID_PARA);
    }
#endif
#if THINKER_RUNTIME_CHECK
    if (workspace != NULL &&
                          (workspace->dptr_ == 0 || workspace->mem_.type_ != 2 ||
                           workspace->dtype_ != Int8 || workspace->byte_ != 1)) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    // Get workspace tensor if present
    // Call hardware-specific addition implementation
    THINKER_RET_CHECK(iqadd_luna(x1, x2, workspace, y), "iqadd_luna");

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","iqAdd", total_t);  
#endif

#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
