#undef __OP__
#define __OP__ ConvTranspose2dInt

#include <math.h>

#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/deconv2dint.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/deconv2dint.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/deconv2dint.h"
#endif

/**
 * Forward pass implementation for 2D Convolution Transpose Integer operator
 * @param op: Operator structure containing transpose convolution attributes
 * @param tensors: Array of input/output tensors (input, weight, optional bias, output, optional temp)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list for weight data handling
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator* op, tTensor** tensors, int32_t num_tensor, tDMA_List* list) {
    // Validate tensor count and input requirements
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
    CHECK_GE(op->num_input_, 2);
    CHECK_LE(op->num_input_, 3);
    
    // Get transpose convolution attributes
    ConvTranspose2dIntAttrs* attrs = (ConvTranspose2dIntAttrs*)((int8_t*)op + op->attr_offset_);
    tTensor* X = ((tTensor**)tensors)[0];
    
    // Handle weight data from DMA list if present
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
    if (list->total_ != 0)
        getWeightData(list, 0);
#endif

    // Get weight and output tensors
    tTensor* W = ((tTensor**)tensors)[1];
    tTensor* Y = ((tTensor**)tensors)[op->num_input_];
    
    // Initialize temporary tensors
    tTensor* Temp = NULL;
    tTensor* dma_temp = NULL;
    tTensor Weight_temp = W[0];
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
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
        } else if (num_tensor == op->num_input_ + op->num_output_ + 2) {
            Temp = ((tTensor**)tensors)[op->num_input_ + op->num_output_];
            dma_temp = ((tTensor**)tensors)[op->num_input_ + op->num_output_ + 1];
            Weight_temp.dptr_ = (addr_type)dma_temp->dptr_;
            Weight_temp.mem_.type_ = 2;
        }
        
        // Handle bias tensor if present
        if (3 == op->num_input_) {
            tTensor* Bias = ((tTensor**)tensors)[op->num_input_ - 1];
            tTensor Bias_temp = Bias[0];
            Bias_temp.scale_ = X->scale_ + W->scale_;
            int32_t size = getShapeSize(&(W->shape_));
            Bias_temp.dptr_ = (addr_type)((int8_t*)Weight_temp.dptr_ + ALIGN16(size));
            ret = deconv2dint_luna(X, &Weight_temp, &Bias_temp, Y, Temp, attrs);
        } 
        else {
            ret = deconv2dint_luna(X, &Weight_temp, NULL, Y, Temp, attrs);
        }
    }
    else {
        // Handle non-DMA case
        if (num_tensor == op->num_input_ + op->num_output_ + 1) {
            Temp = ((tTensor**)tensors)[op->num_input_ + op->num_output_];
        }
        if (3 == op->num_input_) {
            tTensor* Bias = ((tTensor**)tensors)[op->num_input_ - 1];
            tTensor Bias_temp = Bias[0];
            Bias_temp.scale_ = X->scale_ + W->scale_;
            ret = deconv2dint_luna(X, &Weight_temp, &Bias_temp, Y, Temp, attrs);
        } 
        else {
            ret = deconv2dint_luna(X, &Weight_temp, NULL, Y, Temp, attrs);
        }
    }
    
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","ConvTranspose2dInt", total_t);  
#endif
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__