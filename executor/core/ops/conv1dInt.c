#undef __OP__
#define __OP__ Conv1dInt
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/conv1dint.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/conv1dint.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/conv1dint.h"
#endif

/**
 * Forward pass implementation for 1D Convolution Integer operator
 * @param op: Operator structure containing convolution attributes
 * @param tensors: Array of input/output tensors (input, weight, optional bias, output, optional temp)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list for weight data handling
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator* op, tTensor** tensors, int32_t num_tensor, tDMA_List* list) {
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || list == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if ((op->num_input_ != 2 && op->num_input_ != 3) ||
                        op->num_output_ != 1 ||
                        num_tensor < op->num_input_ + op->num_output_) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
    // Get convolution attributes
    Conv1dIntAttrs* attrs = (Conv1dIntAttrs*)((int8_t*)op + op->attr_offset_);
    
    // Get input tensor
    tTensor* X = ((tTensor**)tensors)[0];
    
    // Handle weight data from DMA list if present
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
    if (list->total_ != 0)
        getWeightData(list, 0);
#endif

    // Get weight and output tensors
    tTensor* W = ((tTensor**)tensors)[1];
    tTensor* Y = ((tTensor**)tensors)[op->num_input_];

#if THINKER_PARAM_CHECK
    if (X == NULL || W == NULL || Y == NULL ||
                        X->shape_.ndim_ != 3 || Y->shape_.ndim_ != 3 ||
                        X->dtype_ != Int8 ||
                        (W->dtype_ != Int4 && W->dtype_ != Int8) ||
                        (Y->dtype_ != Int8 && Y->dtype_ != Int16 &&
                         Y->dtype_ != Int32) || attrs->group <= 0 ||
                        attrs->kernel == 0 || attrs->stride == 0) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    tTensor *Bias = op->num_input_ == 3 ? tensors[2] : NULL;
#if THINKER_PARAM_CHECK
    if (Bias != NULL && Bias->dtype_ != Int32) {
        return (T_ERR_INVALID_DATATYPE);
    }
#endif

    int32_t expected_w =
        (X->shape_.dims_[2] + attrs->pad[0] + attrs->pad[1] - attrs->kernel) /
            attrs->stride + 1;
#if THINKER_RUNTIME_CHECK
    if (X->shape_.dims_[0] <= 0 || X->shape_.dims_[1] <= 0 ||
                          X->shape_.dims_[2] <= 0 ||
                          X->shape_.dims_[2] + attrs->pad[0] + attrs->pad[1] <
                              attrs->kernel ||
                          X->shape_.dims_[1] % attrs->group != 0 ||
                          Y->shape_.dims_[1] % attrs->group != 0 ||
                          Y->shape_.dims_[0] != X->shape_.dims_[0] ||
                          Y->shape_.dims_[1] <= 0 || expected_w <= 0 ||
                          Y->shape_.dims_[2] != expected_w || X->dptr_ == 0 ||
                          W->dptr_ == 0 || Y->dptr_ == 0 || X->dptr_ == Y->dptr_) {
        return (T_ERR_INVALID_PARA);
    }

    if (Bias != NULL &&
                          (Bias->dptr_ == 0 ||
                           getTensorSize(Bias) != (size_t)Y->shape_.dims_[1])) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
    // Initialize temporary tensors
    tTensor* Temp = NULL;
    tTensor* dma_temp = NULL;
    tTensor Weight_temp = W[0];
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif
    
    // Process based on DMA list presence and tensor count
    if (list->total_ > 0) {
        // Handle DMA-based weight processing
        if (num_tensor == op->num_input_ + op->num_output_ + 1) {
            dma_temp = ((tTensor**)tensors)[op->num_input_ + op->num_output_];
            Weight_temp.dptr_ = (addr_type)dma_temp->dptr_;
            Weight_temp.mem_.type_ = 2;
        }
        else if (num_tensor == op->num_input_ + op->num_output_ + 2) {
            Temp = ((tTensor**)tensors)[op->num_input_ + op->num_output_];
            dma_temp = ((tTensor**)tensors)[op->num_input_ + op->num_output_ + 1];
            Weight_temp.dptr_ = (addr_type)dma_temp->dptr_;
            Weight_temp.mem_.type_ = 2;
        }
#if THINKER_RUNTIME_CHECK
        if (dma_temp == NULL || dma_temp->dptr_ == 0 ||
                              dma_temp->mem_.type_ != 2) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
        
        // Handle bias tensor if present
        if (Bias != NULL) {
            tTensor Bias_temp = Bias[0];
            Bias_temp.scale_ = X->scale_ + W->scale_;
            size_t size = getTensorDataSize(&Weight_temp);
            Bias_temp.dptr_ = (addr_type)((int8_t*)Weight_temp.dptr_ + ALIGN16(size));
            THINKER_RET_CHECK(conv1dint_luna(X, &Weight_temp, &Bias_temp, Y, Temp, attrs), "conv1dint_luna");
        }
        else {
            THINKER_RET_CHECK(conv1dint_luna(X, &Weight_temp, NULL, Y, Temp, attrs), "conv1dint_luna");
        }
    }
    else {
        // Handle non-DMA case
        if (num_tensor == op->num_input_ + op->num_output_ + 1) {
            Temp = ((tTensor**)tensors)[op->num_input_ + op->num_output_];
        }
#if THINKER_RUNTIME_CHECK
        if (Temp != NULL &&
                              (Temp->dptr_ == 0 || Temp->mem_.type_ != 2 ||
                               Temp->shape_.ndim_ != 1)) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
        
        // Handle bias tensor if present
        if (Bias != NULL) {
            tTensor Bias_temp = Bias[0];
            Bias_temp.scale_ = X->scale_ + W->scale_;
            THINKER_RET_CHECK(conv1dint_luna(X, &Weight_temp, &Bias_temp, Y, Temp, attrs), "conv1dint_luna");
        }
        else {
            THINKER_RET_CHECK(conv1dint_luna(X, &Weight_temp, NULL, Y, Temp, attrs), "conv1dint_luna");
        }
    }
    
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (", "Conv1dInt", total_t); 
#endif
#else
    return T_ERR_NO_SUPPORT_OP;
#endif
    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
