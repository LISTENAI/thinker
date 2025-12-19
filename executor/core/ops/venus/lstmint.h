#ifndef _LSTMINT_LUNA_H_
#define _LSTMINT_LUNA_H_

#include <stdint.h>
#include <stdio.h>
#include <string.h>
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

/**
 * @brief LSTM parameters structure
 */
typedef struct _luna_lstm_param {
    int32_t go_forward;     // Direction of processing (forward or backward)
    int32_t hidden_size;    // Number of hidden units
    int32_t input_size;     // Input feature size
    int32_t iw_size;        // Size of input-to-hidden weights
    int32_t hw_size;        // Size of hidden-to-hidden weights
    int32_t ib_size;        // Size of input bias
    int32_t hb_size;        // Size of hidden bias
    int32_t q_i;            // Input quantization factor
    int32_t q_iw;           // Input-to-hidden weight quantization factor
    int32_t q_h;            // Hidden state quantization factor
    int32_t q_hw;           // Hidden-to-hidden weight quantization factor
    int32_t q_ib;           // Input bias quantization factor
    int32_t q_hb;           // Hidden bias quantization factor
    int32_t q_o;            // Output quantization factor
    void *p_h_in;           // Pointer to input hidden state
    void *p_c_in;           // Pointer to input cell state
    void *p_iw;             // Pointer to input-to-hidden weights
    void *p_hw;             // Pointer to hidden-to-hidden weights
    void *p_ib;             // Pointer to input bias
    void *p_hb;             // Pointer to hidden bias
} luna_lstm_param_t;

/**
 * @brief Ceiling function for integer division
 * @param x Input integer
 * @param shift Number of bits to shift
 * @return int32_t Result after ceiling operation
 */
static int32_t luna_ceil(int32_t x, int32_t shift) {
    if (x & ~(0xFFFFFFFF << shift)) {
        return (x >> shift) + 1;
    } else {
        return (x >> shift);
    }
}

/**
 * @brief Calculate the number of splits for matrix multiplication
 * @param M Number of rows
 * @param N Number of columns
 * @param L Number of elements in the third dimension
 * @param byte Data type size in bytes
 * @return int32_t Number of splits
 */
static int32_t calc_mat_mul_split_num(int M, int N, int L, int byte) {
    const int32_t right_limit = 32 * 1024;
    int32_t split_num = 1;
    int32_t split_L = L / split_num;

    while ((luna_ceil(N, 3) << 3) * (luna_ceil(split_L, 2) << 2) * byte > right_limit ||
           (L % split_num != 0)) {
        split_num++;
        split_L = L / split_num;
    }

    return split_num;
}

/**
 * @brief Core LSTM computation function for quantized integer data
 * @param params LSTM parameters
 * @param t Current time step
 * @param p_input Input data pointer
 * @param p_output Output data pointer
 * @param p_tmp Temporary workspace pointer
 * @return int32_t Operation status
 */
static int32_t luna_lstm_q7_int8_inner(luna_lstm_param_t *params, int32_t t,
                                       int8_t *p_input, int8_t *p_output,
                                       int8_t *p_tmp) {
    int32_t ret = -1;
    const int32_t active_q_in = 11;
    const int32_t active_q_out = 7;

    // Pointer assignments
    int8_t *p_in = p_input;
    int8_t *p_out = p_output;
    int8_t *p_h_in = (int8_t *)params->p_h_in;
    int16_t *p_cell_in = (int16_t *)params->p_c_in;
    int8_t *p_iw_weight = (int8_t *)params->p_iw;
    int8_t *p_hw_weight = (int8_t *)params->p_hw;
    int32_t *p_ib_bias = (int32_t *)params->p_ib;
    int32_t *p_hb_bias = (int32_t *)params->p_hb;

    // Quantization factors
    int32_t i_q = params->q_i;
    int32_t h_q = params->q_h;
    int32_t iw_q = params->q_iw;
    int32_t hw_q = params->q_hw;
    int32_t ib_q = params->q_ib;
    int32_t hb_q = params->q_hb;

    // Step 1: Compute input gates
    int32_t *p_out1 = (int32_t *)p_tmp;
    int32_t split_num = calc_mat_mul_split_num(1, params->input_size, params->hidden_size * 4, 1);
    ret = API_LIB(split_mat_mul_q7_int32)(p_in, p_iw_weight, p_out1, split_num, 1,
                                          params->input_size, params->hidden_size * 4, 0);
    ret = API_LIB(add_q31_int32)(p_out1, p_ib_bias, p_out1, params->hidden_size * 4, 0);

    // Step 2: Compute hidden gates
    int32_t *p_out2 = (int32_t *)p_tmp + params->hidden_size * 4;
    split_num = calc_mat_mul_split_num(1, params->hidden_size, params->hidden_size * 4, 1);
    ret = API_LIB(split_mat_mul_q7_int32)(p_h_in, p_hw_weight, p_out2, split_num, 1,
                                          params->hidden_size, params->hidden_size * 4, 0);
    ret = API_LIB(add_q31_int32)(p_out2, p_hb_bias, p_out2, params->hidden_size * 4, 0);

    // Step 3: Scale and add gates
    if (active_q_in > ib_q && active_q_in > hb_q) {
        ret = API_LIB(scale_q31_int32)(p_out1, 1 << (active_q_in - ib_q), p_out1,
                                       params->hidden_size * 4, 0);
        ret = API_LIB(scale_q31_int32)(p_out2, 1 << (active_q_in - hb_q), p_out2,
                                       params->hidden_size * 4, 0);
    } else if (active_q_in > ib_q) {
        int32_t shift2 = hb_q - active_q_in;
        ret = API_LIB(scale_q31_int32)(p_out1, 1 << (active_q_in - ib_q), p_out1,
                                       params->hidden_size * 4, 0);
        ret = API_LIB(scale_q31_int32)((const q31_t *)p_out2, 1, (int32_t *)p_out2,
                                       params->hidden_size * 4, shift2);
    } else if (active_q_in > hb_q) {
        int32_t shift1 = ib_q - active_q_in;
        ret = API_LIB(scale_q31_int32)((const q31_t *)p_out1, 1, (int32_t *)p_out1,
                                       params->hidden_size * 4, shift1);
        ret = API_LIB(scale_q31_int32)(p_out2, 1 << (active_q_in - hb_q), p_out2,
                                       params->hidden_size * 4, 0);
    } else {
        int32_t shift1 = ib_q - active_q_in;
        int32_t shift2 = hb_q - active_q_in;
        ret = API_LIB(scale_q31_int32)((const q31_t *)p_out1, 1, (int32_t *)p_out1,
                                       params->hidden_size * 4, shift1);
        ret = API_LIB(scale_q31_int32)((const q31_t *)p_out2, 1, (int32_t *)p_out2,
                                       params->hidden_size * 4, shift2);
    }
    ret = API_LIB(add_q31_int16)((const q31_t *)p_out1, (q31_t *)p_out2,
                                 (int16_t *)p_out1, params->hidden_size * 4, 0);

    // Step 4: Compute gates using activation functions
    int16_t *G_i = (int16_t *)p_out1;
    int16_t *G_f = G_i + params->hidden_size;
    int16_t *G_c = G_f + params->hidden_size;
    int16_t *G_o = G_c + params->hidden_size;

    int8_t *g_i = (int8_t *)p_out2;
    int8_t *g_f = g_i + params->hidden_size;
    int8_t *g_c = g_f + params->hidden_size;
    int8_t *g_o = g_c + params->hidden_size;

    ret = API_LIB(sigmoid_int8)(G_i, g_i, params->hidden_size);
    ret = API_LIB(sigmoid_int8)(G_f, g_f, params->hidden_size);
    ret = API_LIB(tanh_int8)(G_c, g_c, params->hidden_size);
    ret = API_LIB(sigmoid_int8)(G_o, g_o, params->hidden_size);

    // Step 5: Compute cell state and hidden state
    int32_t *p_out3 = (int32_t *)p_out2 + params->hidden_size * 4;
    int32_t *p_out4 = p_out3 + params->hidden_size;

    ret = API_LIB(scale_q7_int16)(g_f, 1, G_f, params->hidden_size, 0);
    ret = API_LIB(mul_q15_int32)(p_cell_in, G_f, p_out3, params->hidden_size, 0);
    ret = API_LIB(mul_q7_int32)(g_i, g_c, p_out4, params->hidden_size, 0);
    ret = API_LIB(add_q31_int32)(p_out3, p_out4, p_out3, params->hidden_size, 0);
    ret = API_LIB(scale_q31_int16)(p_out3, 1, p_cell_in, params->hidden_size, active_q_out);

    ret = API_LIB(scale_q31_int16)(p_out3, 1, G_o, params->hidden_size,
                                   active_q_out + active_q_out - active_q_in);
    ret = API_LIB(tanh_int8)(G_o, g_i, params->hidden_size);
    ret = API_LIB(mul_q7_int8)(g_o, g_i, p_h_in, params->hidden_size,
                               active_q_out + active_q_out - h_q);
    ret = API_LIB(scale_q7_int8)(p_h_in, 1, p_out, params->hidden_size, 0);

    return ret;
}

/**
 * @brief Main LSTM function for quantized integer data
 * @param data Input tensor
 * @param history_h Previous hidden state tensor
 * @param history_c Previous cell state tensor
 * @param i2h_weight Input-to-hidden weights tensor
 * @param h2h_weight Hidden-to-hidden weights tensor
 * @param i2h_bias Input bias tensor
 * @param h2h_bias Hidden bias tensor
 * @param mask Mask tensor (optional)
 * @param out Output tensor
 * @param hidden_o Output hidden state tensor
 * @param cell_o Output cell state tensor
 * @param params LSTM attributes
 * @param workspace Temporary workspace tensor
 * @return int32_t Operation status
 */
int32_t lstmint_luna(const tTensor *data, const tTensor *history_h,
                     const tTensor *history_c, const tTensor *i2h_weight,
                     const tTensor *h2h_weight, const tTensor *i2h_bias,
                     const tTensor *h2h_bias, const tTensor *mask,
                     const tTensor *out, const tTensor *hidden_o,
                     const tTensor *cell_o, const LstmIntAttrs *params,
                     const tTensor *workspace) {
    int32_t ret = -1;
    if (data->dtype_ != Int8) {
        return -1;
    }

    int32_t seq_len = 0, batch_size = 0;
    if (params->layout == 0) {
        seq_len = data->shape_.dims_[0];
        batch_size = data->shape_.dims_[1];
    } else {
        seq_len = data->shape_.dims_[1];
        batch_size = data->shape_.dims_[0];
    }

    if (mask) {
        seq_len = (int32_t)(*(int32_t *)mask->dptr_);
    }

    luna_lstm_param_t p_lstm_param;
    p_lstm_param.go_forward = (params->direction) ^ 1;
    p_lstm_param.input_size = params->input_size;
    p_lstm_param.hidden_size = params->hidden_size;
    p_lstm_param.iw_size = getTensorSize(i2h_weight);
    p_lstm_param.hw_size = getTensorSize(h2h_weight);
    p_lstm_param.ib_size = getTensorSize(i2h_bias);
    p_lstm_param.hb_size = getTensorSize(h2h_bias);
    p_lstm_param.q_i = (int32_t)data->scale_;
    p_lstm_param.q_h = (int32_t)hidden_o->scale_;
    p_lstm_param.q_iw = (int32_t)i2h_weight->scale_;
    p_lstm_param.q_hw = (int32_t)h2h_weight->scale_;
    p_lstm_param.q_ib = p_lstm_param.q_i + p_lstm_param.q_iw;
    p_lstm_param.q_hb = p_lstm_param.q_h + p_lstm_param.q_hw;
    p_lstm_param.q_o = (int32_t)out->scale_;
    p_lstm_param.p_h_in = (void *)hidden_o->dptr_;
    p_lstm_param.p_c_in = (void *)cell_o->dptr_;
    p_lstm_param.p_iw = (void *)i2h_weight->dptr_;
    p_lstm_param.p_hw = (void *)h2h_weight->dptr_;
    p_lstm_param.p_ib = (void *)i2h_bias->dptr_;
    p_lstm_param.p_hb = (void *)h2h_bias->dptr_;

    int32_t step_size = p_lstm_param.input_size;
    int32_t out_step_size = p_lstm_param.hidden_size;
    int8_t *p_input = (int8_t *)data->dptr_;
    int8_t *p_out = (int8_t *)out->dptr_;
    int8_t *p_tmp = (int8_t *)workspace->dptr_;
    int32_t t = 0;

    if (history_c != NULL) {
        API_LIB(memcpy)(p_lstm_param.p_c_in, (void *)history_c->dptr_,
                        history_c->byte_ * p_lstm_param.hidden_size);
    } else {
        memset(p_lstm_param.p_c_in, 0, p_lstm_param.hidden_size * cell_o->byte_);
    }

    if (history_h != NULL) {
        API_LIB(memcpy)(p_lstm_param.p_h_in, (void *)history_h->dptr_,
                        history_h->byte_ * p_lstm_param.hidden_size);
    } else {
        memset(p_lstm_param.p_h_in, 0, p_lstm_param.hidden_size * hidden_o->byte_);
    }

    if (p_lstm_param.go_forward == 1) {
        for (t = 0; t < seq_len; t++) {
            ret = luna_lstm_q7_int8_inner(&p_lstm_param, t,
                                         p_input + step_size * t,
                                         p_out + out_step_size * t, p_tmp);
        }
    } else {
        for (t = seq_len - 1; t >= 0; t--) {
            ret = luna_lstm_q7_int8_inner(&p_lstm_param, seq_len - t - 1,
                                         p_input + step_size * t,
                                         p_out + out_step_size * t, p_tmp);
        }
    }

    return ret;
}

#endif