#undef __OP__
#define __OP__ GRUInt
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_VENUS
#include "./venus/gruint.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/gruint.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/gruint.h"
#endif

/**
 * Forward pass implementation for Gated Recurrent Unit Integer operator
 * Performs GRU computation with integer quantization
 * @param op: Operator structure containing GRU attributes
 * @param tensors: Array of input/output tensors (input, i2h_weights, h2h_weights, i2h_bias, h2h_bias, output, hidden_output, optional workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list for weight data handling
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator* op, tTensor** tensors, int32_t num_tensor, tDMA_List* list) {
    // Validate tensor count
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
    
    // Get GRU attributes
    GRUIntAttrs* attr = (GRUIntAttrs*)((int8_t*)op + op->attr_offset_);
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    // Get all tensor pointers
    tTensor* input = tensors[0];
    tTensor* i2h_w = tensors[1];
    tTensor* h2h_w = tensors[2];
    tTensor* i2h_bias = tensors[3];
    tTensor* h2h_bias = tensors[4];

    tTensor* output = tensors[op->num_input_];
    tTensor* hidden_o = tensors[op->num_input_ + 1];

    // Get workspace tensor if present
    tTensor* workspace = NULL;
    if (num_tensor > op->num_input_ + op->num_output_) {
        workspace = tensors[op->num_input_ + op->num_output_];
    }

    // Initialize dummy tensors for hidden state and mask
    tTensor hidden_i_inst;
    hidden_i_inst.shape_.ndim_ = 0;
    
    tTensor mask;
    mask.shape_.ndim_ = 0;

#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

#ifdef THINKER_USE_VENUS
    // Venus hardware implementation
    if (num_tensor > op->num_input_ + op->num_output_) {
        workspace = tensors[op->num_input_ + op->num_output_];
        tTensor *dma_temp   = ((tTensor**)tensors)[op->num_input_ + op->num_output_ + 1];
        tTensor i2h_w_temp  = i2h_w[0];
        i2h_w_temp.dptr_    = dma_temp->dptr_;

        dma_temp  = ((tTensor**)tensors)[op->num_input_ + op->num_output_ + 2];
        tTensor i2h_bias_temp = i2h_bias[0];
        i2h_bias_temp.dptr_ = dma_temp->dptr_;

        dma_temp  = ((tTensor**)tensors)[op->num_input_ + op->num_output_ + 3];
        tTensor h2h_w_temp  = h2h_w[0];
        h2h_w_temp.dptr_    = dma_temp->dptr_;

        dma_temp  = ((tTensor**)tensors)[op->num_input_ + op->num_output_ + 4];
        tTensor h2h_bias_temp     = h2h_bias[0];
        h2h_bias_temp.dptr_          = dma_temp->dptr_;

        ret = gruint_luna(input, &hidden_i_inst, i2h_w, h2h_w, i2h_bias, h2h_bias,
                      &mask, output, hidden_o, attr, workspace);
    }
#elif defined(THINKER_USE_ARCS) || defined(THINKER_USE_VENUSA)
    // ARC/VENUSA hardware implementation
    if(list->total_ > 0) {
        if (num_tensor > op->num_input_ + op->num_output_) {
            workspace = tensors[op->num_input_ + op->num_output_];
            tTensor *dma_temp   = ((tTensor**)tensors)[op->num_input_ + op->num_output_ + 1];
            tTensor i2h_w_temp  = i2h_w[0];
            i2h_w_temp.dptr_    = dma_temp->dptr_;

            dma_temp  = ((tTensor**)tensors)[op->num_input_ + op->num_output_ + 2];
            tTensor i2h_bias_temp = i2h_bias[0];
            i2h_bias_temp.dptr_ = dma_temp->dptr_;

            dma_temp  = ((tTensor**)tensors)[op->num_input_ + op->num_output_ + 3];
            tTensor h2h_w_temp  = h2h_w[0];
            h2h_w_temp.dptr_    = dma_temp->dptr_;

            dma_temp  = ((tTensor**)tensors)[op->num_input_ + op->num_output_ + 4];
            tTensor h2h_bias_temp     = h2h_bias[0];
            h2h_bias_temp.dptr_          = dma_temp->dptr_;

            ret = gruint_luna(input, &hidden_i_inst, i2h_w, h2h_w, i2h_bias, h2h_bias,
                          &mask, output, hidden_o, attr, workspace);
        }
    }
    else {
        if (num_tensor > op->num_input_ + op->num_output_) {
            workspace = tensors[op->num_input_ + op->num_output_];
        }

        ret = gruint_luna(input, &hidden_i_inst, i2h_w, h2h_w, i2h_bias, h2h_bias,
                          &mask, output, hidden_o, attr, workspace);
    }
#endif

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","GruInt", total_t);  
#endif
    
    return ret;
}

#include "core/operator_template.h"
#undef __OP__