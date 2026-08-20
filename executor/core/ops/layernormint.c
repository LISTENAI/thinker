#undef __OP__
#define __OP__ LayerNormInt
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_VENUS
#include "./venus/layernormint.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/layernormint.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/layernormint.h"
#endif

/**
 * Forward pass implementation for Integer Quantized Layer Normalization operator
 * Applies layer normalization to input tensor
 * @param op: Operator structure containing layer normalization attributes
 * @param tensors: Array of input/output tensors (input, weight, output, optional bias, optional workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list for weight data handling
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || list == NULL ||
                        op->num_input_ != 3 || op->num_output_ != 1) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    int32_t expected_tensors = op->num_input_ + op->num_output_ + 1;
    if (list->total_ > 0) expected_tensors++;
#if THINKER_PARAM_CHECK
    if (num_tensor != expected_tensors) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    LayerNormIntAttrs *attrs = (LayerNormIntAttrs *)((int8_t *)op + op->attr_offset_);
    tTensor *X = tensors[0];
    tTensor *W = tensors[1];
    tTensor *bias = tensors[2];
    tTensor *Y = tensors[3];
    tTensor *workspace = tensors[4];
#if THINKER_PARAM_CHECK
    if (attrs == NULL || X == NULL || W == NULL || bias == NULL ||
                        Y == NULL || workspace == NULL || X->dptr_ == 0 ||
                        W->dptr_ == 0 || bias->dptr_ == 0 || Y->dptr_ == 0 ||
                        workspace->dptr_ == 0 || X->shape_.ndim_ < 2 ||
                        !equalShape(&X->shape_, &Y->shape_)) {
        return (T_ERR_INVALID_PARA);
    }

    if (X->dtype_ != Int8 || bias->dtype_ != Int32 ||
                        (Y->dtype_ != Int8 && Y->dtype_ != Int16 &&
                         Y->dtype_ != Int32) || workspace->dtype_ != Int8 ||
                        workspace->byte_ != 1) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (getTensorSize(W) == 0 ||
                        getTensorSize(bias) != getTensorSize(W) ||
                        X->zero_ != 0 || W->zero_ != 0 || bias->zero_ != 0 ||
                        Y->zero_ != 0 || !isfinite(X->scale_) ||
                        !isfinite(W->scale_) || !isfinite(Y->scale_) ||
                        X->scale_ != floorf(X->scale_) ||
                        W->scale_ != floorf(W->scale_) ||
                        Y->scale_ != floorf(Y->scale_)) {
        return (T_ERR_INVALID_DATA);
    }
#endif
#if THINKER_RUNTIME_CHECK
    if (workspace->mem_.type_ != 2 ||
                          workspace->shape_.ndim_ != 1 ||
                          ((uintptr_t)workspace->dptr_ & 3U) != 0) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif

    tTensor weight_tmp = W[0];
    tTensor bias_tmp;
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
    if (list->total_ != 0)
        getWeightData(list, 0);
    
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif
    
    if (list->total_ > 0) {
        tTensor *dma_buffer = tensors[5];
#if THINKER_RUNTIME_CHECK
        if (dma_buffer == NULL || dma_buffer->dptr_ == 0) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
        weight_tmp.dptr_ = (addr_type)dma_buffer->dptr_;
        weight_tmp.mem_.type_ = 2;
        bias_tmp = bias[0];
        bias_tmp.scale_ = X->scale_ + W->scale_;
        size_t size = getTensorDataSize(&weight_tmp);
        bias_tmp.dptr_ = (addr_type)((int8_t *)weight_tmp.dptr_ + ALIGN16(size));
        bias_tmp.mem_.type_ = 2;
        bias = &bias_tmp;
        
        THINKER_RET_CHECK(layernormalint_venus(X, &weight_tmp, bias, Y, workspace, attrs), "layernromalint_venus");
    } else {
        bias->scale_ = X->scale_ + W->scale_;
        THINKER_RET_CHECK(layernormalint_venus(X, W, bias, Y, workspace, attrs), "layernromalint_venus");
    }
    
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","LayernormInt", total_t);  
#endif
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
