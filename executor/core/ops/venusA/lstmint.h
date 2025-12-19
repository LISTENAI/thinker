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
  int32_t go_forward;
  int32_t hidden_size;
  int32_t input_size;
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

/**
 * @brief Calculate ceiling value
 * @param x Input value
 * @param shift Shift amount
 * @return Ceiling result
 */
static  int32_t luna_ceil(int32_t x, int32_t shift) {
  if (x & ~(0xFFFFFFFF << shift)) {
    return (x >> shift) + 1;
  } else {
    return (x >> shift);
  }
}

/**
 * @brief Calculate matrix multiplication split number
 * @param M Matrix M rows
 * @param N Matrix N columns
 * @param L Matrix L size
 * @param byte Byte size
 * @return Split number
 */
static int calc_mat_mul_split_num(int M,int N,int L,int byte)
{
    int32_t split_num = 1;

    const int32_t right_limit = 32 * 1024;
    int32_t split_L = L / split_num;

    int32_t int8_condition_r =(luna_ceil(N, 3) << 3) * (luna_ceil(split_L, 2) << 2) * byte;  // right:8x4
    while (int8_condition_r > right_limit || (0 != (L % split_num))) 
    {
        split_num++;
        split_L = L / split_num;
        int8_condition_r = (luna_ceil(N, 3) << 3) *(luna_ceil(split_L, 2) << 2) * byte;  // right:8x4
    }
    return split_num;
}

/**
 * @brief LSTM kernel function (quantized version)
 * @param params LSTM parameters
 * @param t Time step
 * @param p_input Input pointer
 * @param p_output Output pointer
 * @param p_tmp Temporary buffer
 * @return Execution status
 */
static int32_t luna_lstm_q7_int8_inner(luna_lstm_param_t *params, int32_t t, int8_t *p_input, int8_t *p_output, int8_t *p_tmp) 
{
    int32_t ret = -1;
    const int32_t active_q_in = 27;
    const int32_t active_q_out = 31;
    
    // Initialize parameters
    luna_lstm_param_t *p_lstm_param = params;
    int32_t input_size = p_lstm_param->input_size;
    int32_t hidden_size = p_lstm_param->hidden_size;
    int32_t iw_size = p_lstm_param->iw_size;
    int32_t hw_size = p_lstm_param->hw_size;
    int32_t ib_size = p_lstm_param->ib_size;
    int32_t hb_size = p_lstm_param->hb_size;
    
    // Initialize pointers
    int8_t *p_in = (int8_t *)p_input;
    int8_t *p_out = (int8_t *)p_output;
    int8_t *p_h_in = (int8_t *)p_lstm_param->p_h_in;
    int32_t *p_cell_in = (int32_t *)p_lstm_param->p_c_in;
    int8_t *p_iw_weight = (int8_t *)p_lstm_param->p_iw;
    int8_t *p_hw_weight = (int8_t *)p_lstm_param->p_hw;
    int32_t *p_ib_bias = (int32_t *)p_lstm_param->p_ib;
    int32_t *p_hb_bias = (int32_t *)p_lstm_param->p_hb;
    
    // Get quantization parameters
    int32_t i_q = p_lstm_param->q_i;
    int32_t h_q = p_lstm_param->q_h;
    int32_t iw_q = p_lstm_param->q_iw;
    int32_t hw_q = p_lstm_param->q_hw;
    int32_t ib_q = p_lstm_param->q_ib;
    int32_t hb_q = p_lstm_param->q_hb;
    
    // Main computation steps
    int32_t *gates_input = (int32_t *)p_tmp;
    ret = API_LIB(split_mat_mul_bias_i8i8i32o32)(p_iw_weight, p_in, p_ib_bias, gates_input, hidden_size * 4, input_size, 1, 0);
    
    int32_t *gates_hidden = (int32_t *)p_tmp + hidden_size * 4;
    ret = API_LIB(split_mat_mul_bias_i8i8i32o32)(p_hw_weight, p_h_in, p_hb_bias, gates_hidden, hidden_size * 4, hidden_size, 1, 0);
    
    if ((active_q_in > ib_q) && (active_q_in > hb_q)) {
        ret = API_LIB(scale_i32i32o32)(gates_input, 1 << (active_q_in - ib_q), gates_input, hidden_size * 4, 0);
        ret = API_LIB(scale_i32i32o32)(gates_hidden, 1 << (active_q_in - hb_q), gates_hidden, hidden_size * 4, 0);
    } else if (active_q_in > ib_q) {
        int32_t shift2 = hb_q - active_q_in;
        ret = API_LIB(scale_i32i32o32)(gates_input, 1 << (active_q_in - ib_q), gates_input, hidden_size * 4, 0);
        ret = API_LIB(scale_i32i32o32)(gates_hidden, 1, gates_hidden, hidden_size * 4, shift2);
    } else if (active_q_in > hb_q) {
        int32_t shift1 = ib_q - active_q_in;
        ret = API_LIB(scale_i32i32o32)(gates_input, 1, gates_input, hidden_size * 4, shift1);
        ret = API_LIB(scale_i32i32o32)(gates_hidden, 1 << (active_q_in - hb_q), gates_hidden, hidden_size * 4, 0);
    } else {
        int32_t shift1 = ib_q - active_q_in;
        int32_t shift2 = hb_q - active_q_in;
        ret = API_LIB(scale_i32i32o32)(gates_input, 1, gates_input, hidden_size * 4, shift1);
        ret = API_LIB(scale_i32i32o32)(gates_hidden, 1, gates_hidden, hidden_size * 4, shift2);
    }
    
    ret = API_LIB(add_i32i32o32)(gates_input, gates_hidden, gates_input, hidden_size * 4, 0);
    
    int32_t *G_i = (int32_t *)gates_input;
    int32_t *G_f = (int32_t *)gates_input + hidden_size;
    int32_t *G_c = (int32_t *)gates_input + hidden_size * 2;
    int32_t *G_o = (int32_t *)gates_input + hidden_size * 3;
    
    ret = API_LIB(sigmoid_i32o32)(G_i, G_i, hidden_size);
    ret = API_LIB(sigmoid_i32o32)(G_f, G_f, hidden_size);
    ret = API_LIB(tanh_i32o32)(G_c, G_c, hidden_size);
    ret = API_LIB(sigmoid_i32o32)(G_o, G_o, hidden_size);
    
    ret = API_LIB(scale_i32i32o32)(G_i, 1, G_i, hidden_size, 16);
    ret = API_LIB(scale_i32i32o32)(G_f, 1, G_f, hidden_size, 16);
    ret = API_LIB(scale_i32i32o32)(G_c, 1, G_c, hidden_size, 16);
    ret = API_LIB(scale_i32i32o32)(G_o, 1, G_o, hidden_size, 16);
    
    int32_t *cell_new = (int32_t *)p_cell_in;
    int32_t *hidden_new = (int32_t *)p_h_in;
    
    ret = API_LIB(mul_i32i32o32)(p_cell_in, G_f, cell_new, hidden_size, 0);
    ret = API_LIB(mul_i32i32o32)(G_i, G_c, hidden_new, hidden_size, 0);
    ret = API_LIB(add_i32i32o32)(cell_new, hidden_new, p_cell_in, hidden_size, 30 - active_q_in);
    
    ret = API_LIB(tanh_i32o32)(p_cell_in, hidden_new, hidden_size);
    ret = API_LIB(scale_i32i32o32)(p_cell_in, 1, p_cell_in, hidden_size, 12);
    ret = API_LIB(scale_i32i32o32)(hidden_new, 1, hidden_new, hidden_size, 16);
    ret = API_LIB(mul_i32i32o8)(G_o, hidden_new, p_h_in, hidden_size, 30 - h_q);
    ret = API_LIB(scale_i8i8o8)(p_h_in, 1, p_out, hidden_size, 0);
    
    return ret;
}

static int32_t luna_lstm_q7_int8_inner2(luna_lstm_param_t *params, int32_t t, int8_t *p_input, int8_t *p_output, int8_t *p_tmp, tDMA_List *list)
{
  int32_t ret = -1;
  int32_t split_num = 1;
  const int32_t active_q_in = 27;
  const int32_t active_q_out = 31;

  luna_lstm_param_t *p_lstm_param = params;
  int32_t input_size      = p_lstm_param->input_size;
  int32_t hidden_size     = p_lstm_param->hidden_size;

  int32_t iw_size         = p_lstm_param->iw_size;
  int32_t hw_size         = p_lstm_param->hw_size;
  int32_t ib_size         = p_lstm_param->ib_size;
  int32_t hb_size         = p_lstm_param->hb_size;

  int8_t *p_in            = (int8_t *)p_input;
  int8_t *p_out           = (int8_t *)p_output;
  int8_t *p_h_in          = (int8_t *)p_lstm_param->p_h_in;
  int32_t *p_cell_in      = (int32_t *)p_lstm_param->p_c_in;
  int8_t *p_iw_weight     = (int8_t *)p_lstm_param->p_iw;
  int8_t *p_hw_weight     = (int8_t *)p_lstm_param->p_hw;
  int32_t *p_ib_bias      = (int32_t *)p_lstm_param->p_ib;
  int32_t *p_hb_bias      = (int32_t *)p_lstm_param->p_hb;

  int32_t i_q             = p_lstm_param->q_i;
  int32_t h_q             = p_lstm_param->q_h;
  int32_t iw_q            = p_lstm_param->q_iw;
  int32_t hw_q            = p_lstm_param->q_hw;
  int32_t ib_q            = p_lstm_param->q_ib;
  int32_t hb_q            = p_lstm_param->q_hb;

  // step1: [Gi_i, Gf_i, Gc_i, Go_i] =  [Wi_i, Wf_i, Wc_i, Wo_i] * i + Bias_i
  int32_t *p_out1 = (int32_t *)p_tmp;
  int8_t *weight_temp	= (int8_t *)p_iw_weight;
  int32_t *bias_temp	= (int32_t *)(p_iw_weight + hidden_size * input_size * 2);
  ret = API_LIB(split_mat_mul_bias_i8i8i32o32)(weight_temp, p_input, bias_temp, p_out1, hidden_size * 2, input_size, 1, 0);
  getWeightData(list, 0);
  weight_temp	= (int8_t *)p_ib_bias;
  bias_temp		= (int32_t *)((int8_t *)p_ib_bias + hidden_size * input_size * 2);
  ret = API_LIB(split_mat_mul_bias_i8i8i32o32)(weight_temp, p_input, bias_temp, p_out1 + hidden_size * 2, hidden_size * 2, input_size, 1, 0);

  // step2: [Gi_h, Gf_h, Gc_h, Go_h] = Hp * [Wi_h, Wf_h, Wc_h, Wo_h] + Bias_h
  getWeightData(list, 0);
  weight_temp	= (int8_t *)p_hw_weight;
  bias_temp		= (int32_t *)(p_hw_weight + hidden_size * hidden_size * 2);
  int32_t *p_out2 = (int32_t *)p_tmp + p_lstm_param->hidden_size * 4;
  ret = API_LIB(split_mat_mul_bias_i8i8i32o32)(weight_temp, p_h_in, bias_temp, p_out2, hidden_size * 2, hidden_size, 1, 0);
  getWeightData(list, 0);
  weight_temp	= (int8_t *)p_hb_bias;
  bias_temp		= (int32_t *)((int8_t *)p_hb_bias + hidden_size * hidden_size * 2);
  ret = API_LIB(split_mat_mul_bias_i8i8i32o32)(weight_temp, p_h_in, bias_temp, p_out2 + hidden_size * 2, hidden_size * 2, hidden_size, 1, 0);

  // step3:[Gi_i, Gf_i, Gc_i, Go_i] + [Gi_h, Gf_h, Gc_h, Go_h] = [G_i, G_f, G_c, G_o];
  if ((active_q_in > ib_q) && (active_q_in > hb_q)) {
    ret = API_LIB(scale_i32i32o32)(p_out1, 1 << (active_q_in - ib_q), p_out1, hidden_size * 4, 0);
    ret = API_LIB(scale_i32i32o32)(p_out2, 1 << (active_q_in - hb_q), p_out2, hidden_size * 4, 0);
  }
  else if (active_q_in > ib_q) {
    int32_t shift2 = hb_q - active_q_in;
    ret = API_LIB(scale_i32i32o32)(p_out1, 1 << (active_q_in - ib_q), p_out1, hidden_size * 4, 0);
    ret = API_LIB(scale_i32i32o32)(p_out2, (1), (int32_t *)p_out2, hidden_size * 4, shift2);
  }
  else if (active_q_in > hb_q) {
    int32_t shift1 = ib_q - active_q_in;
    ret = API_LIB(scale_i32i32o32)(p_out1, (1), (int32_t *)p_out1, hidden_size * 4, shift1);
    ret = API_LIB(scale_i32i32o32)(p_out2, 1 << (active_q_in - hb_q), p_out2, hidden_size * 4, 0);
  }
  else {
    int32_t shift1 = ib_q - active_q_in;
    int32_t shift2 = hb_q - active_q_in;
    ret = API_LIB(scale_i32i32o32)(p_out1, (1), (int32_t *)p_out1, hidden_size * 4, shift1);
    ret = API_LIB(scale_i32i32o32)(p_out2, (1), (int32_t *)p_out2, hidden_size * 4, shift2);
  }
  ret = API_LIB(add_i32i32o32)(p_out1, (int32_t *)p_out2, (int32_t *)p_out1, hidden_size * 4, 0);

  // step4:sigmod(G_i, G_f, G_o)
  int32_t *G_i = (int32_t *)p_out1;
  int32_t *G_f = (int32_t *)p_out1 + hidden_size;
  int32_t *G_c = (int32_t *)p_out1 + hidden_size * 2;
  int32_t *G_o = (int32_t *)p_out1 + hidden_size * 3;

  ret = API_LIB(sigmoid_i32o32)(G_i, G_i, hidden_size);  // Q27=>Q31
  ret = API_LIB(sigmoid_i32o32)(G_f, G_f, hidden_size);  // Q27=>Q31
  ret = API_LIB(tanh_i32o32)(G_c, G_c, hidden_size);     // Q27=>Q31
  ret = API_LIB(sigmoid_i32o32)(G_o, G_o, hidden_size);  // Q27=>Q31

  ret = API_LIB(scale_i32i32o32)(G_i, 1, G_i, hidden_size, 16);  // Q31=>Q15
  ret = API_LIB(scale_i32i32o32)(G_f, 1, G_f, hidden_size, 16);  // Q31=>Q15
  ret = API_LIB(scale_i32i32o32)(G_c, 1, G_c, hidden_size, 16);  // Q31=>Q15
  ret = API_LIB(scale_i32i32o32)(G_o, 1, G_o, hidden_size, 16);  // Q31=>Q15

  // step4: C_t = g_f .* C_t_1 + g_i * g_c
  int32_t *p_out3 = (int32_t *)p_out1 + hidden_size;    // g_f .* C_t_1
  int32_t *p_out4 = (int32_t *)p_out1;                  // g_i * g_c

  ret = API_LIB(mul_i32i32o32)(p_cell_in, G_f, p_out3, hidden_size, 0); // Q15 + Q15 => Q30
  ret = API_LIB(mul_i32i32o32)(G_i, G_c, p_out4, hidden_size, 0);       // Q15 + Q15 => Q30
  ret = API_LIB(add_i32i32o32)(p_out3, p_out4, p_cell_in, hidden_size, 30 - active_q_in);

  // step5: h_t = g_o .* tanh(C_t)
  // tanh覆盖源操作数
  ret = API_LIB(tanh_i32o32)(p_cell_in, p_out4, hidden_size);              // Q27 => Q31
  ret = API_LIB(scale_i32i32o32)(p_cell_in, 1, p_cell_in, hidden_size, 12); // Q27 => Q15
  ret = API_LIB(scale_i32i32o32)(p_out4, 1, p_out4, hidden_size, 16);   // Q31 => Q15
  ret = API_LIB(mul_i32i32o8)(G_o, p_out4, p_h_in, hidden_size, 30 - h_q);  // Q15 + Q15 => h_q
  ret = API_LIB(scale_i8i8o8)(p_h_in, 1, p_out, hidden_size, 0);

  return ret;
}

int32_t lstmint_luna2(const tTensor *data, const tTensor *history_h, const tTensor *history_c, const tTensor *i2h_weight,
                     const tTensor *h2h_weight, const tTensor *i2h_bias, const tTensor *h2h_bias, const tTensor *mask,
                     const tTensor *out, const tTensor *hidden_o, const tTensor *cell_o, const LstmIntAttrs *params,
                     const tTensor *workspace, tDMA_List *list) {
  // gru default num_directions forward
  int32_t ret = T_ERR_NO_IMPLEMENTED;
  if (data->dtype_ != Int8) {
    return T_ERR_INVALID_DATATYPE;
  }
  int32_t seq_len = 0, batch_size = 0;
  if (params->layout == 0) {
    // T B D
    seq_len = data->shape_.dims_[1];
    batch_size = data->shape_.dims_[2];
  } 
  else {
    // B T D
    seq_len = data->shape_.dims_[2];
    batch_size = data->shape_.dims_[1];
  }
  if (mask)
  {
    seq_len = (int32_t)(*(int32_t *)mask->dptr_);
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
  int8_t *p_tmp = (NULL != workspace) ? (int8_t *)workspace->dptr_ : NULL;
  int32_t workspace_size = (NULL != workspace) ? workspace->shape_.dims_[0] : 0;
  int32_t used_size = 0;

 if(history_c != NULL)
 {
	  ret = API_LIB(memcpy_i8o8)((int8_t *)p_lstm_param.p_c_in, (int8_t *)history_c->dptr_, history_c->byte_ * p_lstm_param.hidden_size);
 }
 else{
	  ret = API_LIB(memset_i8o8)((int8_t *)p_lstm_param.p_c_in, 0, cell_o->byte_ * p_lstm_param.hidden_size);
 }
 if(history_h != NULL)
 {
	  ret = API_LIB(memcpy_i8o8)((int8_t *)p_lstm_param.p_h_in, (int8_t *)history_h->dptr_, history_h->byte_ * p_lstm_param.hidden_size);
 }
 else{
    ret = API_LIB(memset_i8o8)((int8_t *)p_lstm_param.p_h_in, 0, hidden_o->byte_ * p_lstm_param.hidden_size);
 }


  if (go_forward == 1) {
    for (int32_t t = 0; t < seq_len; t++) {
    	ret |= luna_lstm_q7_int8_inner2(&p_lstm_param, t, p_input + step_size * t, p_out + out_step_size * t, p_tmp + used_size, list);
    }
  }
  else {
    for (int32_t t = seq_len - 1; t >= 0; t--) {
      ret |= luna_lstm_q7_int8_inner2(&p_lstm_param, seq_len - t - 1,  p_input + step_size * t, p_out + out_step_size * t, p_tmp + used_size, list);
    }
  }

  return ret;
}

int32_t lstmint_luna(const tTensor *data, const tTensor *history_h, const tTensor *history_c, const tTensor *i2h_weight,
                     const tTensor *h2h_weight, const tTensor *i2h_bias, const tTensor *h2h_bias, const tTensor *mask,
                     const tTensor *out, const tTensor *hidden_o, const tTensor *cell_o, const LstmIntAttrs *params,
                     const tTensor *workspace) {
  // gru default num_directions forward
  int32_t ret = T_ERR_NO_IMPLEMENTED;
  if (data->dtype_ != Int8) {
    return T_ERR_INVALID_DATATYPE;
  }
  int32_t seq_len = 0, batch_size = 0;
  if (params->layout == 0) {
    // T B D
    seq_len     = data->shape_.dims_[1];
    batch_size  = data->shape_.dims_[2];
  }
  else {
    // B T D
    seq_len     = data->shape_.dims_[2];
    batch_size  = data->shape_.dims_[1];
  }
  if (mask)
  {
    seq_len     = (int32_t)(*(int32_t *)mask->dptr_);
  }

  ///////////////////////////////////////////////////
  luna_lstm_param_t p_lstm_param;
  p_lstm_param.go_forward   = (params->direction) ^ 1;
  p_lstm_param.input_size   = params->input_size;
  p_lstm_param.hidden_size  = params->hidden_size;
  p_lstm_param.iw_size      = getTensorSize(i2h_weight);
  p_lstm_param.hw_size      = getTensorSize(h2h_weight);
  p_lstm_param.ib_size      = getTensorSize(i2h_bias);
  p_lstm_param.hb_size      = getTensorSize(h2h_bias);
  p_lstm_param.q_i          = (int32_t)data->scale_;
  p_lstm_param.q_h          = (int32_t)hidden_o->scale_;
  p_lstm_param.q_iw         = (int32_t)i2h_weight->scale_;
  p_lstm_param.q_hw         = (int32_t)h2h_weight->scale_;
  p_lstm_param.q_ib         = p_lstm_param.q_i + p_lstm_param.q_iw;
  p_lstm_param.q_hb         = p_lstm_param.q_h + p_lstm_param.q_hw;
  p_lstm_param.q_o          = (int32_t)out->scale_;
  p_lstm_param.p_h_in       = (void *)hidden_o->dptr_;
  p_lstm_param.p_c_in       = (void *)cell_o->dptr_;
  p_lstm_param.p_iw         = (void *)i2h_weight->dptr_;
  p_lstm_param.p_hw         = (void *)h2h_weight->dptr_;
  p_lstm_param.p_ib         = (void *)i2h_bias->dptr_;
  p_lstm_param.p_hb         = (void *)h2h_bias->dptr_;

  int32_t go_forward        = p_lstm_param.go_forward;
  int32_t step_size         = p_lstm_param.input_size;
  int32_t out_step_size     = p_lstm_param.hidden_size;
  int8_t *p_input           = (int8_t *)data->dptr_;
  int8_t *p_out             = (int8_t *)out->dptr_;
  int8_t *p_tmp             = (NULL != workspace) ? (int8_t *)workspace->dptr_ : NULL;
  int32_t workspace_size    = (NULL != workspace) ? workspace->shape_.dims_[0] : 0;
  int32_t used_size         = 0;

 if(history_c != NULL)
 {
	  ret = API_LIB(memcpy_i8o8)((int8_t *)p_lstm_param.p_c_in, (int8_t *)history_c->dptr_, history_c->byte_ * p_lstm_param.hidden_size);
 }
 else{
	  ret = API_LIB(memset_i8o8)((int8_t *)p_lstm_param.p_c_in, 0, cell_o->byte_ * p_lstm_param.hidden_size);
 }
 if(history_h != NULL)
 {
	  ret = API_LIB(memcpy_i8o8)((int8_t *)p_lstm_param.p_h_in, (int8_t *)history_h->dptr_, history_h->byte_ * p_lstm_param.hidden_size);
 }
 else{
    ret = API_LIB(memset_i8o8)((int8_t *)p_lstm_param.p_h_in, 0, hidden_o->byte_ * p_lstm_param.hidden_size);
 }

  int32_t flag = 0;
  int32_t last_workspace = workspace_size - p_lstm_param.hidden_size * 32 - p_lstm_param.ib_size * 4 - p_lstm_param.hb_size * 4;
  if (last_workspace >= p_lstm_param.iw_size + p_lstm_param.hw_size)
      flag = 15;
  else {
    if (p_lstm_param.iw_size >= p_lstm_param.hw_size) {
      if (last_workspace >= p_lstm_param.iw_size)
        flag = 13;
      else if (last_workspace >= p_lstm_param.hw_size)
        flag = 7;
      else if (last_workspace >= 0)
        flag = 5;
    }
    else {
      if (last_workspace >= p_lstm_param.hw_size)
        flag = 7;
      else if (last_workspace >= p_lstm_param.iw_size)
        flag = 13;
      else if (last_workspace >= 0)
        flag = 5;
    }
	}

  if (flag & 0x08) {
    p_lstm_param.p_iw = p_tmp;
    ret = API_LIB(memcpy_i8o8)((int8_t *)p_lstm_param.p_iw, (int8_t *)i2h_weight->dptr_, p_lstm_param.iw_size);
    used_size += p_lstm_param.iw_size;
  }

  if (flag & 0x04) {
    p_lstm_param.p_ib =  p_tmp + used_size;
    ret = API_LIB(memcpy_i8o8)((int8_t *)p_lstm_param.p_ib, (int8_t *)i2h_bias->dptr_, p_lstm_param.ib_size * 4);
    used_size += p_lstm_param.ib_size * 4;
  }

  if (flag & 0x02) {
    p_lstm_param.p_hw = p_tmp + used_size;
    ret = API_LIB(memcpy_i8o8)((int8_t *)p_lstm_param.p_hw, (int8_t *)h2h_weight->dptr_, p_lstm_param.hw_size);
    used_size += p_lstm_param.hw_size;
  }

  if (flag & 0x01) {
    p_lstm_param.p_hb = p_tmp + used_size;
    ret = API_LIB(memcpy_i8o8)((int8_t *)p_lstm_param.p_hb, (int8_t *)h2h_bias->dptr_, p_lstm_param.hb_size * 4);
    used_size += p_lstm_param.hb_size * 4;
  }
//   }

  if (go_forward == 1) {
    for (int32_t t = 0; t < seq_len; t++) {
      ret |= luna_lstm_q7_int8_inner(&p_lstm_param, t, p_input + step_size * t, p_out + out_step_size * t, p_tmp + used_size);
    }
  }
  else {
    for (int32_t t = seq_len - 1; t >= 0; t--) {
      ret |= luna_lstm_q7_int8_inner(&p_lstm_param, seq_len - t - 1,  p_input + step_size * t, p_out + out_step_size * t, p_tmp + used_size);
    }
  }

  return ret;
}

#endif
