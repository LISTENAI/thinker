#undef __OP__
#define __OP__ LSTMInt
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_VENUS
#include "./venus/lstmint.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/lstmint.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/lstmint.h"
#endif

/**
 * Forward pass implementation for Integer Quantized LSTM operator
 * Performs LSTM (Long Short-Term Memory) computation on input tensor
 * @param op: Operator structure containing LSTM attributes
 * @param tensors: Array of input/output tensors (input, weights, biases, optional sequence length, hidden states, output, cell states, workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list for weight data handling
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Validate tensor count
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
    
    // Get LSTM attributes
    LstmIntAttrs *attr = (LstmIntAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
    // Get input, weights, biases, and output tensors
    int32_t w_idx = 1;
    tTensor *t_seq = NULL;
    tTensor *t_hidden_in = NULL;
    tTensor *t_cell_in = NULL;
    
    if (op->num_input_ == 5) {
        w_idx = 1;
    } else if (op->num_input_ == 6) { // include sequence length
        w_idx = 2;
        t_seq = tensors[1];
    } else if (op->num_input_ == 7) {
        w_idx = 3;
        t_hidden_in = tensors[1];
        t_cell_in = tensors[2];
    } else if (op->num_input_ == 8) {
        w_idx = 4;
        t_seq = tensors[1];
        t_hidden_in = tensors[2];
        t_cell_in = tensors[3];
    }
    
    tTensor *input = tensors[0];
    tTensor *i2h_w = tensors[w_idx];
    tTensor *h2h_w = tensors[w_idx + 1];
    tTensor *i2h_bias = tensors[w_idx + 2];
    tTensor *h2h_bias = tensors[w_idx + 3];
    
    tTensor *output = tensors[op->num_input_];
    tTensor *hidden_o = tensors[op->num_input_ + 1];
    tTensor *hidden_c = tensors[op->num_input_ + 2];
    tTensor *workspace = NULL;
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
    if (list->total_ != 0)
        getWeightData(list, 0);
#endif

#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

#if THINKER_USE_VENUS
    if (list->total_ > 0) {
        workspace = tensors[op->num_input_ + op->num_output_];
        tTensor *dma_temp = ((tTensor **)tensors)[op->num_input_ + op->num_output_ + 1];
        tTensor i2h_w_temp = i2h_w[0];
        i2h_w_temp.dptr_ = (int8_t *)dma_temp->dptr_;
        
        tTensor h2h_w_temp = h2h_w[0];
        h2h_w_temp.dptr_ = (int8_t *)i2h_w_temp.dptr_ + getShapeSize(&(i2h_w_temp.shape_)) * i2h_w_temp.byte_;

        tTensor i2h_bias_temp = i2h_bias[0];
        i2h_bias_temp.dptr_ = (int8_t *)h2h_w_temp.dptr_ + getShapeSize(&(h2h_w_temp.shape_)) * h2h_w_temp.byte_;

        tTensor h2h_bias_temp = h2h_bias[0];
        h2h_bias_temp.dptr_ = (int8_t *)i2h_bias_temp.dptr_ + getShapeSize(&(i2h_bias_temp.shape_)) * i2h_bias_temp.byte_;
        
        ret = lstmint_luna(input, t_hidden_in, t_cell_in, &i2h_w_temp, &h2h_w_temp,
                        &i2h_bias_temp, &h2h_bias_temp, t_seq, output, hidden_o, hidden_c, attr, workspace);
    }
    else {
        if (num_tensor > op->num_input_ + op->num_output_) {
            workspace = tensors[op->num_input_ + op->num_output_];
        }

        ret = lstmint_luna(input, t_hidden_in, t_cell_in, i2h_w, h2h_w, i2h_bias, h2h_bias, 
                            t_seq, output, hidden_o, hidden_c, attr, workspace);
    }
#elif THINKER_USE_ARCS
    if (list->total_ > 0) {
        if (num_tensor > op->num_input_ + op->num_output_) {
            workspace = tensors[op->num_input_ + op->num_output_];
            tTensor *dma_temp = ((tTensor **)tensors)[op->num_input_ + op->num_output_ + 1];
            tTensor i2h_w_temp = i2h_w[0];
            i2h_w_temp.dptr_ = dma_temp->dptr_;
            
            dma_temp = ((tTensor **)tensors)[op->num_input_ + op->num_output_ + 2];
            tTensor i2h_bias_temp = i2h_bias[0];
            i2h_bias_temp.dptr_ = dma_temp->dptr_;
            
            dma_temp = ((tTensor **)tensors)[op->num_input_ + op->num_output_ + 3];
            tTensor h2h_w_temp = h2h_w[0];
            h2h_w_temp.dptr_ = dma_temp->dptr_;
            
            dma_temp = ((tTensor **)tensors)[op->num_input_ + op->num_output_ + 4];
            tTensor h2h_bias_temp = h2h_bias[0];
            h2h_bias_temp.dptr_ = dma_temp->dptr_;
            
            ret = lstmint_luna2(input, t_hidden_in, t_cell_in, &i2h_w_temp, &h2h_w_temp, &i2h_bias_temp, &h2h_bias_temp,
                               t_seq, output, hidden_o, hidden_c, attr, workspace, list);
        }
    } else {
        if (num_tensor > op->num_input_ + op->num_output_) {
            workspace = tensors[op->num_input_ + op->num_output_];
        }
        ret = lstmint_luna(input, t_hidden_in, t_cell_in, i2h_w, h2h_w, i2h_bias, h2h_bias,
                          t_seq, output, hidden_o, hidden_c, attr, workspace);
    }
#elif THINKER_USE_VENUSA
    if (list->total_ > 0) {
        if (num_tensor > op->num_input_ + op->num_output_) {
            workspace = tensors[op->num_input_ + op->num_output_];
            tTensor *dma_temp = ((tTensor **)tensors)[op->num_input_ + op->num_output_ + 1];
            tTensor i2h_w_temp = i2h_w[0];
            i2h_w_temp.dptr_ = dma_temp->dptr_;
            
            dma_temp = ((tTensor **)tensors)[op->num_input_ + op->num_output_ + 2];
            tTensor i2h_bias_temp = i2h_bias[0];
            i2h_bias_temp.dptr_ = dma_temp->dptr_;
            
            dma_temp = ((tTensor **)tensors)[op->num_input_ + op->num_output_ + 3];
            tTensor h2h_w_temp = h2h_w[0];
            h2h_w_temp.dptr_ = dma_temp->dptr_;
            
            dma_temp = ((tTensor **)tensors)[op->num_input_ + op->num_output_ + 4];
            tTensor h2h_bias_temp = h2h_bias[0];
            h2h_bias_temp.dptr_ = dma_temp->dptr_;
            
            ret = lstmint_luna(input, t_hidden_in, t_cell_in, &i2h_w_temp, &h2h_w_temp, &i2h_bias_temp, &h2h_bias_temp,
                               t_seq, output, hidden_o, hidden_c, attr, workspace);
        }
    } else {
        if (num_tensor > op->num_input_ + op->num_output_) {
            workspace = tensors[op->num_input_ + op->num_output_];
        }
        ret = lstmint_luna(input, t_hidden_in, t_cell_in, i2h_w, h2h_w, i2h_bias, h2h_bias,
                          t_seq, output, hidden_o, hidden_c, attr, workspace);
    }
#endif

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","LSTMInt", total_t);
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__