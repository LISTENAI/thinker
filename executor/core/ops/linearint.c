#undef __OP__
#define __OP__ LinearInt
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_VENUS
#include "./venus/linearint.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/linearint.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/linearint.h"
#endif

/**
 * Forward pass implementation for Integer Quantized Linear operator
 * Performs linear transformation (matrix multiplication) on input tensor
 * @param op: Operator structure containing linear transformation attributes
 * @param tensors: Array of input/output tensors (input, weight, optional bias, output, optional workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list for weight data handling
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || list == NULL ||
                        (op->num_input_ != 2 && op->num_input_ != 3) ||
                        op->num_output_ != 1) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    LinearIntAttrs *attrs = (LinearIntAttrs *)((int8_t *)op + op->attr_offset_);
    tTensor *input = tensors[0];
    tTensor *weight_src = tensors[1];
    tTensor *bias = op->num_input_ == 3 ? tensors[2] : NULL;
    tTensor *output = tensors[op->num_input_];
    int32_t base_tensors = op->num_input_ + op->num_output_;
    int32_t extra_tensors = num_tensor - base_tensors;
    int32_t has_dma = list->total_ > 0;
#if THINKER_PARAM_CHECK
    if (extra_tensors < has_dma || extra_tensors > has_dma + 1) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    tTensor *workspace = extra_tensors > has_dma ? tensors[base_tensors] : NULL;
    tTensor *dma_buffer = has_dma ? tensors[num_tensor - 1] : NULL;
#if THINKER_PARAM_CHECK
    if (attrs == NULL || input == NULL || weight_src == NULL ||
                        output == NULL || input->dptr_ == 0 || weight_src->dptr_ == 0 ||
                        output->dptr_ == 0 || input->shape_.ndim_ < 1 ||
                        weight_src->shape_.ndim_ != 2 ||
                        (bias != NULL && bias->dptr_ == 0)) {
        return (T_ERR_INVALID_PARA);
    }

    if (input->zero_ != 0 || weight_src->zero_ != 0 ||
                        output->zero_ != 0 || (bias != NULL && bias->zero_ != 0) ||
                        !isfinite(input->scale_) || !isfinite(weight_src->scale_) ||
                        !isfinite(output->scale_) || input->scale_ != floorf(input->scale_) ||
                        weight_src->scale_ != floorf(weight_src->scale_) ||
                        output->scale_ != floorf(output->scale_)) {
        return (T_ERR_INVALID_DATA);
    }
#endif
    int32_t N = input->shape_.dims_[input->shape_.ndim_ - 1];
    int32_t L = attrs->transB ? weight_src->shape_.dims_[0] : weight_src->shape_.dims_[1];
#if THINKER_PARAM_CHECK
    if (N <= 0 || L <= 0 ||
                        (attrs->transB ? weight_src->shape_.dims_[1] :
                         weight_src->shape_.dims_[0]) != N ||
                        (bias != NULL && getTensorSize(bias) != (size_t)L) ||
                        output->shape_.ndim_ != input->shape_.ndim_) {
        return (T_ERR_INVALID_DATA);
    }

    for (int32_t i = 0; i < input->shape_.ndim_; ++i) {
        if (output->shape_.dims_[i] !=
                            (i == input->shape_.ndim_ - 1 ? L : input->shape_.dims_[i])) {
            return (T_ERR_INVALID_DATA);
        }
    }
#endif
#if THINKER_RUNTIME_CHECK
    if (workspace != NULL &&
                          (workspace->dptr_ == 0 || workspace->dtype_ != Int8 ||
                           workspace->byte_ != 1 || workspace->mem_.type_ != 2 ||
                           workspace->shape_.ndim_ != 1 ||
                           ((uintptr_t)workspace->dptr_ & 3U) != 0)) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif

    tTensor weight_tmp;
    memcpy(&weight_tmp, weight_src, sizeof(tTensor));
    tTensor *weight = &weight_tmp;
    tTensor bias_tmp;
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
    if (list->total_ != 0)
        getWeightData(list, 0);
#endif

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    if (list->total_ > 0) {
#if THINKER_RUNTIME_CHECK
        if (dma_buffer == NULL || dma_buffer->dptr_ == 0) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
        size_t dma_required = weight_src->mem_.type_ != 2 ?
                              ALIGN16(getTensorDataSize(weight_src)) : 0;
        if (bias != NULL && bias->mem_.type_ != 2)
            dma_required += ALIGN16(getTensorDataSize(bias));
#if THINKER_RUNTIME_CHECK
        if (dma_buffer->mem_.type_ != 2 ||
                              getTensorDataSize(dma_buffer) < dma_required) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
        size_t dma_offset = 0;
        if (weight_src->mem_.type_ != 2) {
            weight->dptr_ = (addr_type)dma_buffer->dptr_;
            weight->mem_.type_ = 2;
            dma_offset = ALIGN16(getTensorDataSize(weight));
        }
        if (bias != NULL) {
            bias_tmp = *bias;
            bias_tmp.scale_ = input->scale_ + weight->scale_;
            if (bias->mem_.type_ != 2) {
                bias_tmp.dptr_ = (addr_type)((int8_t *)dma_buffer->dptr_ + dma_offset);
                bias_tmp.mem_.type_ = 2;
            }
            bias = &bias_tmp;
        }
        
        THINKER_RET_CHECK(linearint_luna(input, weight, bias, attrs, workspace, output), "linearint_luna");
    } else {
        if (bias != NULL) {
            bias_tmp = *bias;
            bias_tmp.scale_ = input->scale_ + weight->scale_;
            bias = &bias_tmp;
        }
        THINKER_RET_CHECK(linearint_luna(input, weight, bias, attrs, workspace, output), "linearint_luna");
    }

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","LinearInt", total_t);  
#endif
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
