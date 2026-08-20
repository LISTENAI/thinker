#undef __OP__
#define __OP__ Conv2dInt
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/conv2dint.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/conv2dint.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/conv2dint.h"
#endif

/**
 * Forward pass implementation for 2D Convolution Integer operator
 * @param op: Operator structure containing convolution attributes
 * @param tensors: Array of input/output tensors (input, weight, optional bias, output, optional temp)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list for weight data handling
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator* op, tTensor** tensors, int32_t num_tensor, tDMA_List* list) {
    // Validate tensor count and input requirements
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || list == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if (op->num_output_ != 1) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
    CHECK_GE(op->num_input_, 2);
    CHECK_LE(op->num_input_, 3);
    
    // Get convolution attributes
    Conv2dIntAttrs* attrs = (Conv2dIntAttrs*)((int8_t*)op + op->attr_offset_);
    
    // Get input, weight, and output tensors
    tTensor* X = ((tTensor**)tensors)[0];
    tTensor* W = ((tTensor**)tensors)[1];
    tTensor* Y = ((tTensor**)tensors)[op->num_input_];
#if THINKER_PARAM_CHECK
    if (X == NULL || W == NULL || Y == NULL ||
                        X->shape_.ndim_ != 4 || Y->shape_.ndim_ != 4 ||
                        X->shape_.dims_[0] != 1 || Y->shape_.dims_[0] != 1) {
        return (T_ERR_INVALID_PARA);
    }

    if (attrs->quant_type > 1) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    if (op->num_input_ == 3) {
        tTensor *Bias = tensors[op->num_input_ - 1];
        #if THINKER_PARAM_CHECK
        if (Bias == NULL || Bias->dtype_ != Int32) {
            return (T_ERR_INVALID_DATATYPE);
        }
#endif
    }
    
    // Handle weight data from DMA list if present
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
    if (list->total_ != 0)
        getWeightData(list, 0);
#endif

    // Initialize temporary tensors
    tTensor* Temp = NULL;
    tTensor* dma_temp = NULL;
    tTensor Weight_temp = W[0];

#if THINKER_PARAM_CHECK
    if (list->total_ > 0 &&
                        num_tensor > op->num_input_ + op->num_output_ + 2) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
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

#if THINKER_PARAM_CHECK
        if (dma_temp == NULL || dma_temp->dptr_ == 0 ||
                            dma_temp->mem_.type_ != 2) {
            return (T_ERR_NO_WORKSPACE);
        }

        if (Temp != NULL) {
            if (Temp->dptr_ == 0 || Temp->mem_.type_ != 2 ||
                                Temp->shape_.ndim_ != 1 || Temp->shape_.dims_[0] <= 0) {
                return (T_ERR_NO_WORKSPACE);
            }
        }
#endif
        
        // Handle bias tensor if present
        if (3 == op->num_input_) {
            tTensor* Bias = ((tTensor**)tensors)[op->num_input_ - 1];
            tTensor Bias_temp = Bias[0];
            Bias_temp.scale_ = X->scale_ + W->scale_;
            size_t size = getTensorDataSize(&Weight_temp);
            Bias_temp.dptr_ = (addr_type)((int8_t*)Weight_temp.dptr_ + ALIGN16(size));
            THINKER_RET_CHECK(conv2dint_luna(X, &Weight_temp, &Bias_temp, Y, Temp, attrs), "conv2dint_luna");
        }
        else {
            THINKER_RET_CHECK(conv2dint_luna(X, &Weight_temp, NULL, Y, Temp, attrs), "conv2dint_luna");
        }
    }
    else {
        // Handle non-DMA case
        if (num_tensor == op->num_input_ + op->num_output_ + 1) {
            Temp = ((tTensor**)tensors)[op->num_input_ + op->num_output_];
        }

#if THINKER_PARAM_CHECK
        if (Temp != NULL) {
            if (Temp->dptr_ == 0 || Temp->mem_.type_ != 2 ||
                                Temp->shape_.ndim_ != 1 || Temp->shape_.dims_[0] <= 0) {
                return (T_ERR_NO_WORKSPACE);
            }
        }
#endif
        
        // Handle bias tensor if present
        if (3 == op->num_input_) {
            tTensor* Bias = ((tTensor**)tensors)[op->num_input_ - 1];
            tTensor Bias_temp = Bias[0];
            Bias_temp.scale_ = X->scale_ + W->scale_;
            THINKER_RET_CHECK(conv2dint_luna(X, &Weight_temp, &Bias_temp, Y, Temp, attrs), "conv2dint_luna");
        }
        else {
            THINKER_RET_CHECK(conv2dint_luna(X, &Weight_temp, NULL, Y, Temp, attrs), "conv2dint_luna");
        }
    }
    
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","Conv2dInt", total_t);  
#endif
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
