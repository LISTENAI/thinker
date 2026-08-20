#ifndef __GRUINT_H__
#define __GRUINT_H__

#include <math.h>
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "thinker_status.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#include "luna/luna_matrix_math.h"
#define API_LIB(api) luna_##api
#endif


/**
 * @brief GRU operation parameters
 */
typedef struct _bigru_param {
    int32_t go_forward;    // Direction of processing
    int32_t batch_size;    // Batch size
    int32_t hidden_size;   // Size of hidden state
    int32_t input_size;    // Size of input
    int32_t iw_size;       // Size of input-to-hidden weights
    int32_t hw_size;       // Size of hidden-to-hidden weights
    int32_t ib_size;       // Size of input bias
    int32_t hb_size;       // Size of hidden bias
    int32_t q_i;           // Input quantization scale
    int32_t q_iw;          // Input-to-hidden weight quantization scale
    int32_t q_h;           // Hidden state quantization scale
    int32_t q_hw;          // Hidden-to-hidden weight quantization scale
    int32_t q_ib;          // Input bias quantization scale
    int32_t q_hb;          // Hidden bias quantization scale
    int32_t q_o;           // Output quantization scale
    void *p_h_in;          // Pointer to hidden state input
    void *p_iw;            // Pointer to input-to-hidden weights
    void *p_hw;            // Pointer to hidden-to-hidden weights
    void *p_ib;            // Pointer to input bias
    void *p_hb;            // Pointer to hidden bias
} gru_param_t;

static int32_t requant_i32_inplace(int32_t *data, int32_t size, int32_t src_q, int32_t dst_q) {
    int32_t shift = dst_q - src_q;
    if (shift == 0) {
        return T_SUCCESS;
    }
    uint32_t left_shift = shift > 0 ? (uint32_t)shift : 0U;
    uint32_t right_shift = shift > 0 ? 0U : (uint32_t)(-shift);
    #if THINKER_PARAM_CHECK
    if (left_shift > 30 || right_shift > 63) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    uint32_t multiplier = left_shift == 0 ? 1U : (1UL << left_shift);
    return API_LIB(scale_i32i32o32)(data, multiplier, data, size, right_shift);
}

/**
 * @brief GRU inner computation function
 * @param params GRU parameters
 * @param t Current time step
 * @param p_input Input data pointer
 * @param p_output Output data pointer
 * @param p_tmp Temporary workspace pointer
 * @param tmp_size Size of temporary workspace
 * @return int32_t Operation status
 */
static int32_t gru_luna_inner(gru_param_t *params, int32_t t, int8_t *p_input, int8_t *p_output, int8_t *p_tmp, int32_t tmp_size) {
    // Define quantization parameters
    const int32_t active_q_in = 27;
    const int32_t active_q_out = 31;

    gru_param_t *p_gru_param = params;
    int32_t input_size = p_gru_param->input_size;
    int32_t hidden_size = p_gru_param->hidden_size;
    int32_t batch_size = p_gru_param->batch_size;

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

    // Properly allocate memory for all intermediate calculations
    int32_t *p_out1 = (int32_t *)p_tmp;
    int32_t *p_out2 = p_out1 + hidden_size * 3 * batch_size;

    // Compute input contributions for all gates [W_ir, W_iz, W_in] * x_t
    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o32)(p_iw_weight, p_in, p_ib_bias, p_out1, hidden_size * 3, input_size, batch_size, 0), "luna_split_mat_mul_bias_i8i8i32o32");
    
    // Adjust quantization for input contributions
    int32_t input_scale_diff = active_q_in - ib_q;
    if (input_scale_diff > 0) {
        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(p_out1, 1 << input_scale_diff, p_out1, hidden_size * 3 * batch_size, 0), "luna_scale_i32i32o32");
    } else {
        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(p_out1, 1, p_out1, hidden_size * 3 * batch_size, -input_scale_diff), "luna_scale_i32i32o32");
    }

    // Compute hidden contributions for all gates [W_hr, W_hz, W_hn] * h_{t-1}
    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o32)(p_hw_weight, p_h_in, p_hb_bias, p_out2, hidden_size * 2, hidden_size, batch_size, 0), "luna_split_mat_mul_bias_i8i8i32o32");

    // Adjust quantization for hidden contributions
    int32_t hidden_scale_diff = active_q_in - hb_q;
    if (hidden_scale_diff > 0) {
        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(p_out2, 1 << hidden_scale_diff, p_out2, hidden_size * 2 * batch_size, 0), "luna_scale_i32i32o32");
    } else {
        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(p_out2, 1, p_out2, hidden_size * 2 * batch_size, -hidden_scale_diff), "luna_scale_i32i32o32");
    }

    // Calculate reset gate: r_t = σ(W_ir·x_t + b_ir + W_hr·h_{t-1} + b_hr)
    int32_t *p_reset_gate  = (int32_t *)p_tmp;
    int32_t *p_hidden_state_input = p_out1 + hidden_size * 2 * batch_size;
    THINKER_RET_CHECK(API_LIB(add_i32i32o32)(p_out1, p_out2, p_reset_gate, hidden_size * 2 * batch_size, 0), "luna_add_i32i32o32");     // active_q_in + active_q_in => active_q_in
    THINKER_RET_CHECK(API_LIB(sigmoid_i32o32)(p_reset_gate, p_reset_gate, hidden_size * batch_size), "luna_sigmoid_i32o32");             // active_q_in => active_q_out

    // Calculate update gate: z_t = σ(W_iz·x_t + b_iz + W_hz·h_{t-1} + b_hz)
    int32_t *p_update_gate = (int32_t *)p_tmp + hidden_size * batch_size;
    THINKER_RET_CHECK(API_LIB(sigmoid_i32o32)(p_update_gate, p_update_gate, hidden_size * batch_size), "luna_sigmoid_i32o32");           // active_q_in => active_q_out

    // Step 1: Compute r_t ⊙ h_{t-1} (element-wise multiplication)
    int32_t *p_h_in_int32 = p_out2;
    int8_t *p_h_reset = (int8_t *)(p_out2 + hidden_size * batch_size);
    THINKER_RET_CHECK(API_LIB(scale_i8i8o32)(p_h_in, 1, p_h_in_int32, hidden_size * batch_size, 0), "luna_scale_i8i8o32");
    THINKER_RET_CHECK(API_LIB(mul_i32i32o8)(p_reset_gate, p_h_in_int32, p_h_reset, hidden_size * batch_size, active_q_out), "luna_mul_i32i32o8");  // active_q_out + h_q - active_q_out => h_q
    // THINKER_RET_CHECK(API_LIB(scale_i32i32o8)(p_h_in_int32, 1, p_h_reset, hidden_size, 0);
    // if (ret != 0) return ret;
    
    // Step 2: Compute W_hn·(r_t ⊙ h_{t-1}) + b_hn
    int32_t *p_hidden_state_prehidden = (int32_t *)p_out2;
    THINKER_RET_CHECK(API_LIB(split_mat_mul_bias_i8i8i32o32)(p_hw_weight + 2 * hidden_size * hidden_size, 
                                                            p_h_reset, p_hb_bias + 2 * hidden_size, 
                                                            p_hidden_state_prehidden, hidden_size, 
                                                            hidden_size, batch_size, 0), "luna_split_mat_mul_bias_i8i8i32o32"); // hb_q
    
    if (hidden_scale_diff > 0) {
        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(p_hidden_state_prehidden, 1 << hidden_scale_diff, p_hidden_state_prehidden, hidden_size * batch_size, 0), "luan_scale_i32i32o32");
    } else {
        THINKER_RET_CHECK(API_LIB(scale_i32i32o32)(p_hidden_state_prehidden, 1, p_hidden_state_prehidden, hidden_size * batch_size, -hidden_scale_diff), "luan_scale_i32i32o32"); // hb_q => active_q_in
    }

    // Step 3: Compute W_in·x_t + b_in + W_hn·(r_t ⊙ h_{t-1}) + b_hn
    int32_t *p_hidden_state = p_out2;
    THINKER_RET_CHECK(API_LIB(add_i32i32o32)(p_hidden_state_input, p_hidden_state_prehidden, p_hidden_state, hidden_size * batch_size, 0), "luan_add_i32i32o32");   // active_q_in => active_q_in

    // Step 4: Calculate candidate hidden state: n_t = tanh(W_in·x_t + b_in + W_hn·(r_t ⊙ h_{t-1}) + b_hn)
    THINKER_RET_CHECK(API_LIB(tanh_i32o32)(p_hidden_state, p_hidden_state, hidden_size * batch_size), "luan_tanh_i32o32");    // activate_q_in => active_q_out
    THINKER_RET_CHECK(requant_i32_inplace(p_hidden_state, hidden_size * batch_size, active_q_out, h_q), "luna_scale_i32i32o32");

    // Calculate final hidden state: h_t = n_t + z_t ⊙ (h_{t-1} - n_t)
    int32_t *p_final_hidden_delta = (int32_t *)p_tmp;
    THINKER_RET_CHECK(API_LIB(scale_i8i8o32)(p_h_in, 1, p_final_hidden_delta, hidden_size * batch_size, 0), "luna_scale_i8i8o32");
    THINKER_RET_CHECK(API_LIB(sub_i32i32o32)(p_final_hidden_delta, p_hidden_state, p_final_hidden_delta, hidden_size * batch_size, 0), "luna_sub_i32i32o32");
    THINKER_RET_CHECK(API_LIB(mul_i32i32o32)(p_update_gate, p_final_hidden_delta, p_final_hidden_delta, hidden_size * batch_size, active_q_out), "luna_mul_i32i32o32");
    THINKER_RET_CHECK(API_LIB(add_i32i32o8)(p_final_hidden_delta, p_hidden_state, p_h_in, hidden_size * batch_size, 0), "luna_add_i32i32o8");

    // Copy result to output
    THINKER_RET_CHECK(API_LIB(memcpy_i8o8)(p_out, p_h_in, hidden_size * batch_size), "luna_memcpy_i8o8");

    return T_SUCCESS;
}

/**
 * @brief GRU operation implementation
 * @param input Input tensor
 * @param history_h History hidden tensor
 * @param i2h_w Input-to-hidden weight tensor
 * @param h2h_w Hidden-to-hidden weight tensor
 * @param i2h_bias Input bias tensor
 * @param h2h_bias Hidden bias tensor
 * @param output Output tensor
 * @param hidden_o Hidden output tensor
 * @param params GRU operation attributes
 * @param workspace Workspace tensor
 * @return int32_t Operation status
 */
int32_t gruint_luna(tTensor *input, tTensor *history_h, tTensor *i2h_w, tTensor *h2h_w, tTensor *i2h_bias, tTensor *h2h_bias,
                    tTensor *output, tTensor *hidden_o, GRUIntAttrs *params, tTensor *workspace) {
    #if THINKER_PARAM_CHECK
    if (input == NULL || history_h == NULL || i2h_w == NULL ||
                        h2h_w == NULL || i2h_bias == NULL || h2h_bias == NULL ||
                        output == NULL || hidden_o == NULL || params == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if (input->dtype_ != Int8 || i2h_w->dtype_ != Int8 ||
                        h2h_w->dtype_ != Int8 || i2h_bias->dtype_ != Int32 ||
                        h2h_bias->dtype_ != Int32 || output->dtype_ != Int8 ||
                        hidden_o->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (params->layout != 0 || params->direction > 1) {
        return (T_ERR_INVALID_PARA);
    }

    if (input->shape_.dims_[1] != 1) {
        return (T_ERR_NO_IMPLEMENTED);
    }
#endif
    #if THINKER_RUNTIME_CHECK
    if (input->dptr_ == 0 || i2h_w->dptr_ == 0 ||
                          h2h_w->dptr_ == 0 || i2h_bias->dptr_ == 0 ||
                          h2h_bias->dptr_ == 0 || output->dptr_ == 0 ||
                          hidden_o->dptr_ == 0 || input->mem_.type_ != 2 ||
                          i2h_w->mem_.type_ != 2 || h2h_w->mem_.type_ != 2 ||
                          i2h_bias->mem_.type_ != 2 || h2h_bias->mem_.type_ != 2 ||
                          output->mem_.type_ != 2 || hidden_o->mem_.type_ != 2 ||
                          (history_h->shape_.ndim_ != 0 &&
                           (history_h->dptr_ == 0 || history_h->mem_.type_ != 2))) {
        return (T_ERR_NO_SUPPORT_OP);
    }
#endif
    size_t scratch_bytes = (size_t)params->hidden_size * 28;
    #if THINKER_RUNTIME_CHECK
    if (workspace == NULL || workspace->dptr_ == 0 ||
                          workspace->mem_.type_ != 2 || workspace->dtype_ != Int8 ||
                          workspace->byte_ != 1 ||
                          ((uintptr_t)workspace->dptr_ & 3U) != 0 ||
                          getTensorDataSize(workspace) < scratch_bytes) {
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
    gru_param.batch_size = batch_size;
    gru_param.input_size = params->input_size;
    gru_param.hidden_size = params->hidden_size;
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
    int8_t *p_tmp = NULL;
    int32_t tmp_size = 0;
    if (params->layout != 0 && batch_size != 1) {
        // [B,T,F] => [T,F,B]
        p_input = (int8_t *)workspace->dptr_;
        int32_t data_size = batch_size * seq_len * params->input_size;
        THINKER_RET_CHECK(API_LIB(mat_trans_i8o8)((int8_t *)input->dptr_, p_input, batch_size, seq_len * params->input_size), "luna_mat_trans_i8o8");
        p_tmp = p_input + data_size;
        tmp_size = workspace->shape_.dims_[0] - data_size;

    } else {
        p_tmp = (int8_t *)workspace->dptr_;
        tmp_size = workspace->shape_.dims_[0];
    }
    int8_t *p_out = (int8_t *)output->dptr_;

    int32_t t = 0;
    if (history_h->shape_.ndim_ == 0) {
        THINKER_RET_CHECK(luna_memset_i8o8(gru_param.p_h_in, 0, gru_param.hidden_size * gru_param.batch_size * hidden_o->byte_), "luna_memset_i8o8");
    } else {
        THINKER_RET_CHECK(API_LIB(memcpy_i8o8)((int8_t *)gru_param.p_h_in,
                          (int8_t *)history_h->dptr_,
                          params->hidden_size * batch_size),
                          "luna_memcpy_i8o8");
    }
    if (go_forward == 1) {
        for (t = 0; t < seq_len; t++) {
            THINKER_RET_CHECK(gru_luna_inner(&gru_param, t, p_input + step_size * t,
                                p_out + out_step_size * t, p_tmp, tmp_size), "gru_luna_inner");
        }
    } else {
        for (t = seq_len - 1; t >= 0; t--) {
            THINKER_RET_CHECK(gru_luna_inner(&gru_param, seq_len - t - 1, p_input + step_size * t,
                                p_out + out_step_size * t, p_tmp, tmp_size), "gru_luna_inner");
        }
    }

    if (params->layout != 0 && batch_size != 1) {
        // [T,H,B] => [B,T,H]
        THINKER_RET_CHECK(API_LIB(mat_trans_i8o8)(p_out, p_out, seq_len * params->hidden_size, batch_size), "luna_mat_trans_i8o8");
    }

    return T_SUCCESS;
}

#endif  // __GRUINT_H__
