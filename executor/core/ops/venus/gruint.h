#ifndef __GRUINT_H__
#define __GRUINT_H__

#include <assert.h>
#include <math.h>
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif
#include "thinker_status.h"
#define ALIGN(X, T) ((((X) + (T)-1) / (T)) * (T))

/**
 * @brief Calculate the number of splits for matrix multiplication
 * @param row Number of rows in the matrix
 * @param col Number of columns in the matrix
 * @param col2 Number of columns in the second matrix
 * @return int32_t Number of splits
 */
static int32_t calc_mat_mul_split_num(int32_t row, int32_t col, int32_t col2) {
    int32_t split = 0;
    int32_t split_size = 0;
    do {
        split++;
        int32_t split_col = col / split;
        split_size = ALIGN(split_col, 8) * ALIGN(col2, 4);
    } while (col % split != 0 || split_size > 32 * 1024);
    return split;
}

/**
 * @brief GRU operation parameters
 */
typedef struct _bigru_param {
    int32_t go_forward;      // Direction of processing (forward or backward)
    int32_t batch_size;
    int32_t hidden_size;     // Size of hidden state
    int32_t input_size;      // Size of input
    int32_t iw_size;         // Size of input-to-hidden weights
    int32_t hw_size;         // Size of hidden-to-hidden weights
    int32_t ib_size;         // Size of input bias
    int32_t hb_size;         // Size of hidden bias
    int32_t q_i;             // Quantization scale for input
    int32_t q_iw;            // Quantization scale for input weights
    int32_t q_h;             // Quantization scale for hidden state
    int32_t q_hw;            // Quantization scale for hidden weights
    int32_t q_ib;            // Quantization scale for input bias
    int32_t q_hb;            // Quantization scale for hidden bias
    int32_t q_o;             // Quantization scale for output
    void *p_h_in;            // Pointer to previous hidden state
    void *p_iw;              // Pointer to input-to-hidden weights
    void *p_hw;              // Pointer to hidden-to-hidden weights
    void *p_ib;              // Pointer to input bias
    void *p_hb;              // Pointer to hidden bias
} gru_param_t;

/**
 * @brief Perform GRU operation for a single time step
 * @param params Pointer to GRU parameters
 * @param t Current time step
 * @param p_input Pointer to input data
 * @param p_output Pointer to output data
 * @param p_tmp Pointer to temporary workspace
 * @param tmp_size Size of temporary workspace
 * @return int32_t Return status (T_ERR_NO_IMPLEMENTED if not implemented, T_SUCCESS if successful)
 */
static int32_t gru_luna_inner(gru_param_t *params, int32_t t, int8_t *p_input,
                             int8_t *p_output, int8_t *p_tmp, int32_t tmp_size) {
    const int32_t active_q_in = 11;
    const int32_t active_q_out = 15;

    gru_param_t *p_gru_param = params;
    int32_t input_size = p_gru_param->input_size;
    int32_t hidden_size = p_gru_param->hidden_size;

    int8_t *p_in = p_input;
    int8_t *p_out = p_output;
    int8_t *p_h_in = (int8_t *)p_gru_param->p_h_in;
    int8_t *p_iw_weight = (int8_t *)p_gru_param->p_iw;
    int8_t *p_hw_weight = (int8_t *)p_gru_param->p_hw;
    int32_t *p_ib_bias = (int32_t *)p_gru_param->p_ib;
    int32_t *p_hb_bias = (int32_t *)p_gru_param->p_hb;

    int32_t i_q = p_gru_param->q_i;
    int32_t h_q = p_gru_param->q_h;
    int32_t iw_q = p_gru_param->q_iw;
    int32_t hw_q = p_gru_param->q_hw;
    int32_t ib_q = p_gru_param->q_ib;
    int32_t hb_q = p_gru_param->q_hb;
    int32_t o_q = p_gru_param->q_o;

    int32_t *p_out1 = (int32_t *)p_tmp;
    int32_t *p_out2 = p_out1 + hidden_size * 3;
    int32_t *p_h_in_16 = p_out2 + hidden_size * 3;
    // Compute input-to-hidden transformation
    // Debug print p_in and p_iw_weight
    THINKER_RET_CHECK(API_LIB(mat_mul_q7_int32)(p_in, p_iw_weight, p_out1, 1, input_size, hidden_size * 3, 0), "luna_mat_mul_q7_int32");
    THINKER_RET_CHECK(API_LIB(add_q31_int32)(p_out1, p_ib_bias, p_out1, hidden_size * 3, 0), "luna_add_q31_int32");
    THINKER_RET_CHECK(API_LIB(scale_q31_int32)(p_out1, 1, p_out1, hidden_size * 3, (ib_q - active_q_in)), "luna_scale_q31_int32");

    // Compute hidden-to-hidden transformation
    THINKER_RET_CHECK(API_LIB(mat_mul_q7_int32)(p_h_in, p_hw_weight, p_out2, 1, hidden_size, hidden_size * 3, 0), "luna_mat_mul_q7_int32");
    THINKER_RET_CHECK(API_LIB(add_q31_int32)(p_out2, p_hb_bias, p_out2, hidden_size * 3, 0), "luna_add_q31_int32");
    THINKER_RET_CHECK(API_LIB(scale_q31_int32)(p_out2, 1, p_out2, hidden_size * 3, (hb_q - active_q_in)), "luna_scale_q31_int32");

    // Compute gates and activation
    int16_t *reset_gate = (int16_t *)p_out1;
    int16_t *update_gate = (int16_t *)p_out1 + hidden_size;
    THINKER_RET_CHECK(API_LIB(add_q31_int16)((const q31_t *)p_out1, (q31_t *)p_out2, (int16_t *)p_out1, hidden_size * 2, 0), "luna_add_q31_int16");
    THINKER_RET_CHECK(API_LIB(sigmoid)((int16_t *)p_out1, reset_gate, hidden_size), "luna_sigmoid");   //reset_gate, active_q_in => active_q_out
    THINKER_RET_CHECK(API_LIB(sigmoid)((int16_t *)p_out1 + hidden_size, update_gate, hidden_size), "luna_sigmoid");   //update gate, active_q_in => active_q_out

    // calculate ht'
    int16_t *h_n_int = (int16_t *)p_out2;
    int16_t *reset_h_n_int = (int16_t *)p_out2 + hidden_size;
    int16_t *candidate_hidden_state = (int16_t *)p_out1;
    THINKER_RET_CHECK(API_LIB(scale_q31_int16)((int32_t *)p_out2 + 2 * hidden_size, 1, h_n_int, hidden_size, 0), "luna_scale_q31_int16");
    THINKER_RET_CHECK(API_LIB(scale_q31_int16)((int32_t *)p_out1 + 2 * hidden_size, 1, reset_h_n_int, hidden_size, 0), "luna_scale_q31_int16");
    THINKER_RET_CHECK(API_LIB(mul_q15_int16)(reset_gate, (int16_t *)h_n_int, candidate_hidden_state, hidden_size, active_q_out), "luna_mul_q15_int16"); // active_q_out + i_q => active_q_in
    THINKER_RET_CHECK(API_LIB(add_q15_int16)(candidate_hidden_state, reset_h_n_int, candidate_hidden_state, hidden_size, 0), "luna_add_q15_int16"); // active_q_in
    THINKER_RET_CHECK(API_LIB(tanh)(candidate_hidden_state, candidate_hidden_state, hidden_size), "luna_tanh"); // active_q_in => active_q_out
    THINKER_RET_CHECK(API_LIB(scale_q15_int16)(candidate_hidden_state, 1, candidate_hidden_state, hidden_size, active_q_out - h_q), "luna_scale_q31_int16"); // active_q_out => h_q

    // ht = (1-zt)*ht-1 + zt * ht' = ht-1 + zt * (ht' - ht-1)
    int16_t *final_hidden_state = (int16_t *)p_out1;
    int16_t *h_in_int16 = (int16_t *)p_out2 + hidden_size;
    THINKER_RET_CHECK(API_LIB(scale_q7_int16)(p_h_in, 1, h_in_int16, hidden_size, 0), "luna_scale_q7_int16");
    THINKER_RET_CHECK(API_LIB(sub_q15_int16)(candidate_hidden_state, h_in_int16, final_hidden_state, hidden_size, 0), "luna_sub_q15_int16"); // active_q_out
    THINKER_RET_CHECK(API_LIB(mul_q15_int16)(update_gate, final_hidden_state, final_hidden_state, hidden_size, (active_q_out + h_q - o_q)), "luna_mul_q15_int16"); // active_q_out + active_q_out => active_q_out
    THINKER_RET_CHECK(API_LIB(add_q15_int8)(final_hidden_state, h_in_int16, p_out, hidden_size, 0), "luna_add_q15_int8"); // o_q => o_q

    THINKER_RET_CHECK(API_LIB(memcpy)(p_h_in, p_out, hidden_size), "luna_memcpy");

    return T_SUCCESS;
}

/**
 * @brief Perform GRU operation for a sequence
 * @param input Pointer to input tensor
 * @param history_h Pointer to previous hidden state tensor
 * @param i2h_w Pointer to input-to-hidden weights tensor
 * @param h2h_w Pointer to hidden-to-hidden weights tensor
 * @param i2h_bias Pointer to input bias tensor
 * @param h2h_bias Pointer to hidden bias tensor
 * @param output Pointer to output tensor
 * @param hidden_o Pointer to current hidden state tensor
 * @param params Pointer to GRU attributes
 * @param workspace Pointer to workspace tensor
 * @return int32_t Return status (T_ERR_NO_IMPLEMENTED if not implemented, T_ERR_INVALID_DATATYPE for invalid data type, T_SUCCESS if successful)
 */
int32_t gruint_luna(tTensor *input, tTensor *history_h, tTensor *i2h_w,
                   tTensor *h2h_w, tTensor *i2h_bias, tTensor *h2h_bias,
                   tTensor *output, tTensor *hidden_o,
                   GRUIntAttrs *params, tTensor *workspace) {
     #if THINKER_PARAM_CHECK
     if (input == NULL || history_h == NULL || i2h_w == NULL || h2h_w == NULL ||
                         i2h_bias == NULL || h2h_bias == NULL || output == NULL || hidden_o == NULL ||
                         params == NULL || input->dptr_ == 0 || i2h_w->dptr_ == 0 ||
                         h2h_w->dptr_ == 0 || i2h_bias->dptr_ == 0 || h2h_bias->dptr_ == 0 ||
                         output->dptr_ == 0 || hidden_o->dptr_ == 0 ||
                         (history_h->shape_.ndim_ != 0 && history_h->dptr_ == 0)) {
         return (T_ERR_INVALID_PARA);
     }
     if (input->dtype_ != Int8 || i2h_w->dtype_ != Int8 ||
                        h2h_w->dtype_ != Int8 || i2h_bias->dtype_ != Int32 ||
                        h2h_bias->dtype_ != Int32 || output->dtype_ != Int8 ||
                        hidden_o->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }
    if ((params->layout != 0 && params->layout != 1) || params->direction < 0 ||
                        params->direction > 1 || params->input_size <= 0 ||
                        params->hidden_size <= 0) {
        return (T_ERR_INVALID_PARA);
    }
    #endif
    int32_t batch_dim = params->layout == 0 ? 1 : 0;
    int32_t seq_dim = params->layout == 0 ? 0 : 1;
    #if THINKER_PARAM_CHECK
    if (input->shape_.ndim_ != 3 || input->shape_.dims_[batch_dim] != 1 ||
                         input->shape_.dims_[2] != params->input_size ||
                         output->shape_.ndim_ != 3 ||
                         output->shape_.dims_[seq_dim] != input->shape_.dims_[seq_dim] ||
                         output->shape_.dims_[batch_dim] != 1 ||
                         output->shape_.dims_[2] != params->hidden_size ||
                        hidden_o->shape_.ndim_ != 3 || hidden_o->shape_.dims_[0] != 1 ||
                        hidden_o->shape_.dims_[1] != 1 ||
                        hidden_o->shape_.dims_[2] != params->hidden_size) {
        return (T_ERR_INVALID_DATA);
    }
    if (i2h_w->shape_.ndim_ != 2 ||
                        i2h_w->shape_.dims_[0] != params->input_size ||
                        i2h_w->shape_.dims_[1] != params->hidden_size * 3 ||
                        h2h_w->shape_.ndim_ != 2 ||
                        h2h_w->shape_.dims_[0] != params->hidden_size ||
                        h2h_w->shape_.dims_[1] != params->hidden_size * 3 ||
                        i2h_bias->shape_.ndim_ != 1 ||
                        i2h_bias->shape_.dims_[0] != params->hidden_size * 3 ||
                        h2h_bias->shape_.ndim_ != 1 ||
                        h2h_bias->shape_.dims_[0] != params->hidden_size * 3) {
        return (T_ERR_INVALID_DATA);
    }
    #endif
    if (history_h->shape_.ndim_ != 0) {
        #if THINKER_PARAM_CHECK
        if (history_h->dtype_ != Int8 || history_h->shape_.ndim_ != 3 ||
                            history_h->shape_.dims_[0] != 1 ||
                            history_h->shape_.dims_[1] != 1 ||
                            history_h->shape_.dims_[2] != params->hidden_size) {
            return (T_ERR_INVALID_DATA);
        }
        #endif
    }
    #if THINKER_PARAM_CHECK
    if (input->scale_ + i2h_w->scale_ < 11 ||
                        input->scale_ + i2h_w->scale_ - 11 > 63 ||
                        hidden_o->scale_ + h2h_w->scale_ < 11 ||
                        hidden_o->scale_ + h2h_w->scale_ - 11 > 63 ||
                        15 - hidden_o->scale_ < 0 || 15 - hidden_o->scale_ > 63 ||
                        output->scale_ != hidden_o->scale_) {
        return (T_ERR_INVALID_PARA);
    }
    if (ALIGN(params->input_size, 8) *
                            ALIGN(params->hidden_size * 3, 4) > 32768 ||
                        ALIGN(params->hidden_size, 8) *
                            ALIGN(params->hidden_size * 3, 4) > 32768) {
        return (T_ERR_NO_IMPLEMENTED);
    }
    #endif
    #if THINKER_RUNTIME_CHECK
    if (input->mem_.type_ != 2 || output->mem_.type_ != 2 ||
                          hidden_o->mem_.type_ != 2 || i2h_w->mem_.type_ != 2 ||
                          h2h_w->mem_.type_ != 2 || i2h_bias->mem_.type_ != 2 ||
                          h2h_bias->mem_.type_ != 2 ||
                          (history_h->shape_.ndim_ != 0 && history_h->mem_.type_ != 2)) {
        return (T_ERR_NO_SUPPORT_OP);
    }
    #endif
    #if THINKER_RUNTIME_CHECK
    if (workspace == NULL || workspace->dptr_ == 0 ||
                          workspace->shape_.ndim_ == 0 ||
                          workspace->mem_.type_ != 2 ||
                          workspace->shape_.dims_[0] < params->hidden_size * 24) {
        return (T_ERR_NO_WORKSPACE);
    }
    #endif

    int32_t seq_len = 0, batch_size = 0;
    if (params->layout == 0) {
        seq_len = input->shape_.dims_[0];
        batch_size = input->shape_.dims_[1];
    } else {
        seq_len = input->shape_.dims_[1];
        batch_size = input->shape_.dims_[0];
    }

    gru_param_t gru_param = {0};
    gru_param.go_forward = params->direction;
    gru_param.input_size = params->input_size;
    gru_param.hidden_size = params->hidden_size;
    gru_param.batch_size = batch_size;
    gru_param.iw_size = getTensorSize(i2h_w);
    gru_param.hw_size = getTensorSize(h2h_w);
    gru_param.ib_size = getTensorSize(i2h_bias);
    gru_param.hb_size = getTensorSize(h2h_bias);
    gru_param.q_i = (int32_t)input->scale_;
    gru_param.q_h = (int32_t)hidden_o->scale_;
    gru_param.q_iw = (int32_t)i2h_w->scale_;
    gru_param.q_hw = (int32_t)h2h_w->scale_;
    gru_param.q_ib = gru_param.q_i + gru_param.q_iw;
    gru_param.q_hb = gru_param.q_h + gru_param.q_hw;
    gru_param.q_o = (int32_t)output->scale_;
    gru_param.p_h_in = (void *)hidden_o->dptr_;
    gru_param.p_iw = (void *)i2h_w->dptr_;
    gru_param.p_hw = (void *)h2h_w->dptr_;
    gru_param.p_ib = (void *)i2h_bias->dptr_;
    gru_param.p_hb = (void *)h2h_bias->dptr_;

    int32_t go_forward = gru_param.go_forward;
    int32_t step_size = gru_param.input_size * gru_param.batch_size;
    int32_t out_step_size = gru_param.hidden_size * gru_param.batch_size;
    int8_t *p_input = (int8_t *)input->dptr_;
    int8_t *p_out = (int8_t *)output->dptr_;
    int8_t *p_tmp = (int8_t *)workspace->dptr_;
    int32_t tmp_size = getTensorSize(workspace) * workspace->byte_;

    if(history_h->shape_.ndim_ == 0)
    {
        memset(gru_param.p_h_in, 0, gru_param.hidden_size * gru_param.batch_size * hidden_o->byte_);
    }
    else{
        memcpy(gru_param.p_h_in, (void *)history_h->dptr_,
               gru_param.hidden_size * gru_param.batch_size * hidden_o->byte_);
    }

    if (go_forward == 1) {
        for (int32_t b = 0; b < batch_size; b++) {
            for (int32_t t = 0; t < seq_len; t++) {
                THINKER_RET_CHECK(gru_luna_inner(&gru_param, t, p_input + step_size * t + b * step_size * seq_len,
                              p_out + out_step_size * t + b * out_step_size * seq_len, p_tmp, tmp_size), "gru_luna_inner");
            }
        }
    } else {
        for (int32_t b = 0; b < batch_size; b++) {
            for (int32_t t = seq_len - 1; t >= 0; t--) {
                THINKER_RET_CHECK(gru_luna_inner(&gru_param, seq_len - t - 1, p_input + step_size * t + b * step_size * seq_len,
                              p_out + out_step_size * t + b * out_step_size * seq_len, p_tmp, tmp_size), "gru_luna_inner");
            }
        }
    }

    return T_SUCCESS;
}

#endif  // __GRUINT_H__
