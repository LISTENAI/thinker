#ifndef _LSTMINT_LUNA_H_
#define _LSTMINT_LUNA_H_
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "luna/luna_math.h"

typedef struct _luna_lstm_param {
  int32_t go_forward;
  int32_t hidden_size;
  int32_t input_size;
  // int32_t num_layers;
  int32_t iw_size;
  int32_t hw_size;
  int32_t ib_size;
  int32_t hb_size;
  int32_t q_i;
  int32_t q_iw;
  int32_t q_h;
  int32_t q_hw;
  int32_t q_ib;
  int32_t q_hb;
  int32_t q_o;
  void *p_h_in;
  void *p_c_in;
  void *p_iw;
  void *p_hw;
  void *p_ib;
  void *p_hb;
} luna_lstm_param_t;

static int32_t luna_lstm_q7_int8_inner(luna_lstm_param_t *params, int32_t t,
                                       int8_t *p_input, int8_t *p_output,
                                       int8_t *p_tmp) {
  int32_t ret = -1;
  const int32_t split_num = 4;
  const int32_t active_q_in = 11;
  const int32_t active_q_out = 7;

  luna_lstm_param_t *p_lstm_param = params;
  int32_t input_size = p_lstm_param->input_size;
  int32_t hidden_size = p_lstm_param->hidden_size;

  int32_t iw_size = p_lstm_param->iw_size;
  int32_t hw_size = p_lstm_param->hw_size;
  int32_t ib_size = p_lstm_param->ib_size;
  int32_t hb_size = p_lstm_param->hb_size;

  int8_t *p_in = (int8_t *)p_input;
  int8_t *p_out = (int8_t *)p_output;
  int8_t *p_h_in = (int8_t *)p_lstm_param->p_h_in;
  int16_t *p_cell_in = (int16_t *)p_lstm_param->p_c_in;
  int8_t *p_iw_weight = (int8_t *)p_lstm_param->p_iw;
  int8_t *p_hw_weight = (int8_t *)p_lstm_param->p_hw;
  int32_t *p_ib_bias = (int32_t *)p_lstm_param->p_ib;
  int32_t *p_hb_bias = (int32_t *)p_lstm_param->p_hb;

  int32_t i_q = p_lstm_param->q_i;
  int32_t h_q = p_lstm_param->q_h;
  int32_t iw_q = p_lstm_param->q_iw;
  int32_t hw_q = p_lstm_param->q_hw;
  int32_t ib_q = p_lstm_param->q_ib;
  int32_t hb_q = p_lstm_param->q_hb;

  // step1: [Gi_i, Gf_i, Gc_i, Go_i] =  [Wi_i, Wf_i, Wc_i, Wo_i] * i + Bias_i
  int32_t *p_out1 = (int32_t *)p_tmp;
  ret = luna_split_mat_mul_q7_int32(p_input, p_iw_weight, p_out1, split_num, 1,
                                    input_size, hidden_size * 4, 0);
  ret = luna_add_q31_int32(p_out1, p_ib_bias, p_out1, hidden_size * 4, 0);

  // step2: [Gi_h, Gf_h, Gc_h, Go_h] = Hp * [Wi_h, Wf_h, Wc_h, Wo_h] + Bias_h
  int32_t *p_out2 = (int32_t *)p_tmp + p_lstm_param->hidden_size * 4;
  ret = luna_split_mat_mul_q7_int32(p_h_in, p_hw_weight, p_out2, split_num, 1,
                                    hidden_size, hidden_size * 4, 0);
  ret = luna_add_q31_int32(p_out2, p_hb_bias, p_out2, hidden_size * 4, 0);

  // step3:[Gi_i, Gf_i, Gc_i, Go_i] + [Gi_h, Gf_h, Gc_h, Go_h] = [G_i, G_f, G_c,
  // G_o];
  if ((active_q_in > ib_q) && (active_q_in > hb_q)) {
    ret = luna_scale_q31_int32(p_out1, 1 << (active_q_in - ib_q), p_out1,
                               hidden_size * 4, 0);
    ret = luna_scale_q31_int32(p_out2, 1 << (active_q_in - hb_q), p_out2,
                               hidden_size * 4, 0);
    ret = luna_add_q31_int16(p_out1, p_out2, (int16_t *)p_out1, hidden_size * 4,
                             0);  // Q11
  } else {
    int32_t shift1 = ib_q - active_q_in;
    int32_t shift2 = hb_q - active_q_in;
    ret = luna_scale_q31_int32((const q31_t *)p_out1, (1), (int32_t *)p_out1,
                               hidden_size * 4, shift1);
    ret = luna_scale_q31_int32((const q31_t *)p_out2, (1), (int32_t *)p_out2,
                               hidden_size * 4, shift2);
    ret = luna_add_q31_int16((const q31_t *)p_out1, (q31_t *)p_out2,
                             (int16_t *)p_out1, hidden_size * 4, 0);
  }

  // step4:sigmod(G_i, G_f, G_o)
  int16_t *G_i = (int16_t *)p_out1;
  int16_t *G_f = (int16_t *)p_out1 + hidden_size;
  int16_t *G_c = (int16_t *)p_out1 + hidden_size * 2;
  int16_t *G_o = (int16_t *)p_out1 + hidden_size * 3;

  int8_t *g_i = (int8_t *)p_out2;
  int8_t *g_f = (int8_t *)p_out2 + hidden_size;
  int8_t *g_c = (int8_t *)p_out2 + hidden_size * 2;
  int8_t *g_o = (int8_t *)p_out2 + hidden_size * 3;

  ret = luna_sigmoid_int8(G_i, g_i, hidden_size);  // Q7
  ret = luna_sigmoid_int8(G_f, g_f, hidden_size);  // Q7
  ret = luna_tanh_int8(G_c, g_c, hidden_size);     // Q7
  ret = luna_sigmoid_int8(G_o, g_o, hidden_size);  // Q7

  // step4: C_t = g_f .* C_t_1 + g_i * g_c
  int32_t *p_out3 = (int32_t *)p_out2 + hidden_size * 4;
  int32_t *p_out4 = (int32_t *)p_out3 + hidden_size;
  ret = luna_scale_q7_int16(g_f, 1, G_f, hidden_size, 0);
  ret = luna_mul_q15_int32(p_cell_in, G_f, p_out3, hidden_size, 0);

  ret = luna_mul_q7_int32(g_i, g_c, p_out4, hidden_size, 0);

  ret = luna_add_q31_int32(p_out3, p_out4, p_out3, hidden_size,
                           0);  // Q7 * Q7 => Q14

  ret = luna_scale_q31_int16(p_out3, 1, p_cell_in, hidden_size, active_q_out);

  // step5: h_t = g_o .* tanh(C_t)
  // tanh覆盖源操作数
  ret = luna_scale_q31_int16(p_out3, 1, G_o, hidden_size,
                             active_q_out + active_q_out - active_q_in);  // Q11
  ret = luna_tanh_int8(G_o, g_i, hidden_size);                            // Q7

  ret = luna_mul_q7_int8(g_o, g_i, p_h_in, hidden_size,
                         active_q_out + active_q_out - h_q);  // Q7 * Q7 => h_q
  ret = luna_scale_q7_int8(p_h_in, 1, p_out, hidden_size, 0);

  return ret;
}

int32_t lstmint_luna(const tTensor *data, const tTensor *history_h,
                     const tTensor *history_c, const tTensor *i2h_weight,
                     const tTensor *h2h_weight, const tTensor *i2h_bias,
                     const tTensor *h2h_bias, const tTensor *mask,
                     const tTensor *out, const tTensor *hidden_o,
                     const tTensor *cell_o, const LstmIntAttrs *params,
                     const tTensor *workspace) {
  // gru default num_directions forward
  int32_t ret = -1;
  if (data->dtype_ != Int8) {
    return -1;
  }
  int32_t seq_len = 0, batch_size = 0;
  if (params->layout == 0) {
    // T B D
    seq_len = data->shape_.dims_[0];
    batch_size = data->shape_.dims_[1];
  } else {
    // B T D
    seq_len = data->shape_.dims_[1];
    batch_size = data->shape_.dims_[0];
  }

  ///////////////////////////////////////////////////
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

  int32_t go_forward = p_lstm_param.go_forward;
  int32_t step_size = p_lstm_param.input_size;
  int32_t out_step_size = p_lstm_param.hidden_size;
  int8_t *p_input = (int8_t *)data->dptr_;
  int8_t *p_out = (int8_t *)out->dptr_;
  int8_t *p_tmp = (int8_t *)workspace->dptr_;
  int32_t t = 0;
  memset(p_lstm_param.p_h_in, 0, p_lstm_param.hidden_size * hidden_o->byte_);
  memset(p_lstm_param.p_c_in, 0, p_lstm_param.hidden_size * cell_o->byte_);
  if (go_forward == 1) {
    for (t = 0; t < seq_len; t++) {
      ret = luna_lstm_q7_int8_inner(&p_lstm_param, t, p_input + step_size * t,
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
