#undef __OP__
#define __OP__ MaxPool
#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/maxpool.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/maxpool.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/maxpool.h"
#endif

/**
 * Forward pass implementation for Max Pooling operator
 * Applies max pooling operation to input tensor
 * @param op: Operator structure containing pooling attributes
 * @param tensors: Array of input/output tensors (input, output, temporary workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || op->num_input_ != 1 ||
                        op->num_output_ != 1 || num_tensor < 2 || num_tensor > 3) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    PoolAttrs *attrs = (PoolAttrs *)((int8_t *)op + op->attr_offset_);
    tTensor *X = tensors[0];
    tTensor *Y = tensors[op->num_input_];
    tTensor *Temp = NULL;
    if (num_tensor > op->num_input_ + op->num_output_) {
        Temp = tensors[op->num_input_ + op->num_output_];
    }

#if THINKER_PARAM_CHECK
    if (X == NULL || Y == NULL || attrs->ceil != 0 || attrs->layout != 0 ||
                        X->dptr_ == 0 || Y->dptr_ == 0 || X->shape_.ndim_ != 4 ||
                        Y->shape_.ndim_ != 4 || X->dtype_ != Int8 || Y->dtype_ != Int8 ||
                        X->byte_ != 1 || Y->byte_ != 1) {
        return (T_ERR_INVALID_DATA);
    }

    if (attrs->kernel[0] == 0 || attrs->kernel[1] == 0 ||
                        (attrs->stride[0] != 1 && attrs->stride[0] != 2 && attrs->stride[0] != 4) ||
                        (attrs->stride[1] != 1 && attrs->stride[1] != 2 && attrs->stride[1] != 4) ||
                        attrs->kernel[0] < attrs->stride[0] || attrs->kernel[1] < attrs->stride[1] ||
                        attrs->pad[0] >= attrs->kernel[0] || attrs->pad[2] >= attrs->kernel[0] ||
                        attrs->pad[1] >= attrs->kernel[1] || attrs->pad[3] >= attrs->kernel[1]) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    int32_t expected_h = (X->shape_.dims_[2] + attrs->pad[0] + attrs->pad[2] -
                          attrs->kernel[0]) / attrs->stride[0] + 1;
    int32_t expected_w = (X->shape_.dims_[3] + attrs->pad[1] + attrs->pad[3] -
                          attrs->kernel[1]) / attrs->stride[1] + 1;
#if THINKER_PARAM_CHECK
    if (expected_h <= 0 || expected_w <= 0 ||
                        Y->shape_.dims_[0] != X->shape_.dims_[0] ||
                        Y->shape_.dims_[1] != X->shape_.dims_[1] ||
                        Y->shape_.dims_[2] != (uint32_t)expected_h ||
                        Y->shape_.dims_[3] != (uint32_t)expected_w) {
        return (T_ERR_INVALID_DATA);
    }
#endif
#if THINKER_RUNTIME_CHECK
    if (Temp != NULL && (Temp->dptr_ == 0 || Temp->mem_.type_ != 2)) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    // Call hardware-specific max pooling implementation
    THINKER_RET_CHECK(maxpool_luna(X, Y, Temp, attrs), "maxpool_luna");

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","MaxPool", total_t);
#endif
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
