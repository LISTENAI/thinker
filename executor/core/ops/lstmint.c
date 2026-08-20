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

#ifdef THINKER_USE_VENUS
static int32_t lstm_aligned_tensor_bytes(tTensor *tensor) {
    return ALIGN16(getShapeSize(&(tensor->shape_)) * tensor->byte_);
}
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
        CHECK_GE(num_tensor, (op->num_input_ + op->num_output_ + 2));
        workspace = tensors[op->num_input_ + op->num_output_];
        tTensor *dma_temp = ((tTensor **)tensors)[op->num_input_ + op->num_output_ + 1];
        int32_t dma_bytes = lstm_aligned_tensor_bytes(i2h_w) + lstm_aligned_tensor_bytes(h2h_w) +
                            lstm_aligned_tensor_bytes(i2h_bias) + lstm_aligned_tensor_bytes(h2h_bias);
#if THINKER_RUNTIME_CHECK
        if (dma_temp->mem_.type_ != 2 ||
                              getTensorSize(dma_temp) * dma_temp->byte_ < dma_bytes) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
        tTensor i2h_w_temp = i2h_w[0];
        i2h_w_temp.dptr_ = (addr_type)((int8_t *)dma_temp->dptr_);
        i2h_w_temp.mem_.type_ = 2;

        tTensor h2h_w_temp = h2h_w[0];
        h2h_w_temp.dptr_ = (addr_type)((int8_t *)i2h_w_temp.dptr_ + lstm_aligned_tensor_bytes(&i2h_w_temp));
        h2h_w_temp.mem_.type_ = 2;

        tTensor i2h_bias_temp = i2h_bias[0];
        i2h_bias_temp.dptr_ = (addr_type)((int8_t *)h2h_w_temp.dptr_ + lstm_aligned_tensor_bytes(&h2h_w_temp));
        i2h_bias_temp.mem_.type_ = 2;

        tTensor h2h_bias_temp = h2h_bias[0];
        h2h_bias_temp.dptr_ = (addr_type)((int8_t *)i2h_bias_temp.dptr_ + lstm_aligned_tensor_bytes(&i2h_bias_temp));
        h2h_bias_temp.mem_.type_ = 2;
        
        THINKER_RET_CHECK(lstmint_luna(input, t_hidden_in, t_cell_in, &i2h_w_temp, &h2h_w_temp,
                        &i2h_bias_temp, &h2h_bias_temp, t_seq, output, hidden_o, hidden_c, attr, workspace), "lstmint_luna");
    }
    else {
        if (num_tensor > op->num_input_ + op->num_output_) {
            workspace = tensors[op->num_input_ + op->num_output_];
        }

        THINKER_RET_CHECK(lstmint_luna(input, t_hidden_in, t_cell_in, i2h_w, h2h_w, i2h_bias, h2h_bias, 
                            t_seq, output, hidden_o, hidden_c, attr, workspace), "lstmint_luna");
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
            
            THINKER_RET_CHECK(lstmint_luna2(input, t_hidden_in, t_cell_in, &i2h_w_temp, &h2h_w_temp, &i2h_bias_temp, &h2h_bias_temp,
                               t_seq, output, hidden_o, hidden_c, attr, workspace, list), "lstmint_luna");
        }
    } else {
    if (num_tensor > op->num_input_ + op->num_output_) {
        workspace = tensors[op->num_input_ + op->num_output_];
    }
    THINKER_RET_CHECK(lstmint_luna(input, t_hidden_in, t_cell_in, i2h_w, h2h_w, i2h_bias, h2h_bias,
                      t_seq, output, hidden_o, hidden_c, attr, workspace), "lstmint_luna");
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
            
            THINKER_RET_CHECK(lstmint_luna(input, t_hidden_in, t_cell_in, &i2h_w_temp, &h2h_w_temp, &i2h_bias_temp, &h2h_bias_temp,
                               t_seq, output, hidden_o, hidden_c, attr, workspace), "lstmint_luna");
        }
    } else {
        if (num_tensor > op->num_input_ + op->num_output_) {
            workspace = tensors[op->num_input_ + op->num_output_];
        }
        THINKER_RET_CHECK(lstmint_luna(input, t_hidden_in, t_cell_in, i2h_w, h2h_w, i2h_bias, h2h_bias,
                          t_seq, output, hidden_o, hidden_c, attr, workspace), "lstmint_luna");
    }
#endif

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","LSTMInt", total_t);
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
