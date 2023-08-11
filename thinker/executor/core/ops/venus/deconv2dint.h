#ifndef _DECONV2DINT_LUNA_H_
#define _DECONV2DINT_LUNA_H_

#include <math.h>
#include <stdint.h>

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "luna/luna_math.h"
#include "thinker_status.h"

static void deconv2dint_venus_para_init(ConvTranspose2dIntAttrs *attrs,
                                        s_conv_struct *conv_attrs, tTensor *X,
                                        tTensor *W, tTensor *Bias, tTensor *Y) {
  memset(conv_attrs, 0, sizeof(s_conv_struct));
  if (NULL != Bias) {
    conv_attrs->is_bias = 1;
  }
  conv_attrs->input_c = X->shape_.dims_[1];
  conv_attrs->input_h = X->shape_.dims_[2];
  conv_attrs->input_w = X->shape_.dims_[3];
  conv_attrs->output_c = Y->shape_.dims_[1];
  conv_attrs->output_h = Y->shape_.dims_[2];
  conv_attrs->output_w = Y->shape_.dims_[3];
  conv_attrs->weight_h = attrs->kernel[0];
  conv_attrs->weight_w = attrs->kernel[1];
  conv_attrs->stride_h = attrs->stride[0];
  conv_attrs->stride_w = attrs->stride[1];
  conv_attrs->padding_h_up = attrs->kernel[0] - attrs->pad[0] - 1;
  conv_attrs->padding_h_down = attrs->kernel[0] - attrs->pad[2] -1 + attrs->output_padding[0];
  conv_attrs->padding_w_left = attrs->kernel[1] - attrs->pad[1] - 1;
  conv_attrs->padding_w_right = attrs->kernel[1] - attrs->pad[3] - 1 + attrs->output_padding[1];
  conv_attrs->input_h_after_padding =
      (conv_attrs->input_h - 1) * conv_attrs->stride_h + 1 +
      conv_attrs->padding_h_up + conv_attrs->padding_h_down;
  conv_attrs->input_w_after_padding =
      (conv_attrs->input_w - 1) * conv_attrs->stride_w + 1 +
      conv_attrs->padding_w_left + conv_attrs->padding_w_right;
  ;
  if (1 == attrs->act_type) {
    conv_attrs->activation_type = RELU;
  } else if (2 == attrs->act_type) {
    conv_attrs->activation_type = PRELU;
  } else {
    conv_attrs->activation_type = NO_ACTIVE;
  }
  int32_t q_x = (int32_t)X->scale_;
  int32_t q_w = (int32_t)W->scale_;
  int32_t q_y = (int32_t)Y->scale_;
  conv_attrs->positive_shift_type = ShiftType_FloorX05;
  conv_attrs->positive_shift_value = q_x + q_w - q_y;
  conv_attrs->negative_shift_type = ShiftType_FloorX05;
  conv_attrs->negative_shift_value = conv_attrs->positive_shift_value;
  conv_attrs->batch_num = attrs->group;
}

static int32_t calc_deconv_luna(int32_t w_dtype, int32_t y_dtype, int8_t *input,
                                int8_t *weight, int32_t *bias, void *output,
                                s_conv_struct *conv_attrs) {
  int32_t ret = 0;
  switch (w_dtype) {
    case Int4:
      switch (y_dtype) {
        case Int8:
          ret = luna_deconv_intx_int8((const int8_t *)input, (int8_t *)weight,
                                      (int32_t *)bias, (int8_t *)output,
                                      conv_attrs, 4);
          break;
        case Int16:
          ret = luna_deconv_intx_int16((const int8_t *)input, (int8_t *)weight,
                                       (int32_t *)bias, (int16_t *)output,
                                       conv_attrs, 4);
          break;
        case Int32:
          ret = luna_deconv_intx_int32((const int8_t *)input, (int8_t *)weight,
                                       (int32_t *)bias, (int32_t *)output,
                                       conv_attrs, 4);
          break;
      }
      break;
    case Int8:
      switch (y_dtype) {
        case Int8:
          ret = luna_deconv_q7_int8((const int8_t *)input, (int8_t *)weight,
                                    (int32_t *)bias, (int8_t *)output,
                                    conv_attrs);
          break;
        case Int16:
          ret = luna_deconv_q7_int16((const int8_t *)input, (int8_t *)weight,
                                     (int32_t *)bias, (int16_t *)output,
                                     conv_attrs);
          break;
        case Int32:
          ret = luna_deconv_q7_int32((const int8_t *)input, (int8_t *)weight,
                                     (int32_t *)bias, (int32_t *)output,
                                     conv_attrs);
          break;
      }
      break;
  }

  return ret;
}

int32_t deconv2dint_venus(tTensor *X, tTensor *W, tTensor *Bias, tTensor *Y,
                          tTensor *Temp, ConvTranspose2dIntAttrs *attrs) {
  int32_t ret = T_ERR_FAIL;

  uint64_t paddr_b = 0;
  if (Bias != NULL) {
    paddr_b = Bias->dptr_;
  }

  s_conv_struct conv_attrs;
  deconv2dint_venus_para_init(attrs, &conv_attrs, X, W, Bias, Y);

  int32_t batch = X->shape_.dims_[0];
  int32_t group_num = attrs->group;
  int32_t in_c = conv_attrs.input_c;
  int32_t in_h = conv_attrs.input_h;
  int32_t in_w = conv_attrs.input_w;
  int32_t ou_c = conv_attrs.output_c;
  int32_t ou_h = conv_attrs.output_h;
  int32_t ou_w = conv_attrs.output_w;
  int32_t k_n = W->shape_.dims_[0];
  int32_t k_c = W->shape_.dims_[3];
  int32_t k_h = conv_attrs.weight_h;
  int32_t k_w = conv_attrs.weight_w;
  int32_t in_batch_size = in_c * in_h * in_w;
  int32_t ou_batch_size = ou_c * ou_h * ou_w * (Y->dtype_ & 0xF);
  int32_t n;

  if (1 == attrs->group) {  // deconv
    for (n = 0; n < batch; n++) {
      int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
      int8_t *p_weight = (int8_t *)W->dptr_;
      int32_t *p_bias = (0 == paddr_b) ? NULL : ((int32_t *)paddr_b);
      int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
      ret = calc_deconv_luna(W->dtype_, Y->dtype_, p_in, p_weight, p_bias,
                             (void *)p_out, &conv_attrs);
    }
  } else {  // group conv
    in_c = in_c / group_num;
    ou_c = ou_c / group_num;
    k_n = k_n / group_num;
    int32_t step_data_in = in_c * in_h * in_w * (0xF & X->dtype_);
    int32_t step_data_out = ou_c * ou_h * ou_w * (0xF & Y->dtype_);
    int32_t step_weight = k_n * k_c * k_h * k_w;
    int32_t step_bias = (0 == paddr_b) ? 0 : ou_c;
    int32_t i;
    conv_attrs.input_c = in_c;
    conv_attrs.output_c = ou_c;
    for (n = 0; n < batch; n++) {
      int32_t in_batch_size = in_c * in_h * in_w * group_num;
      int32_t ou_batch_size =
          ou_c * ou_h * ou_w * group_num * (0xF & Y->dtype_);
      int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
      int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
      int8_t *p_weight = (int8_t *)W->dptr_;
      int32_t *p_bias = (0 == paddr_b) ? NULL : ((int32_t *)paddr_b);
      for (i = 0; i < group_num; i++) {
        int8_t *p_in_tmp = p_in + i * step_data_in;
        int8_t *p_out_tmp = p_out + i * step_data_out;
        int8_t *p_weight_tmp = p_weight + i * step_weight;
        int32_t *p_bias_tmp = p_bias + i * step_bias;
        ret = calc_deconv_luna(W->dtype_, Y->dtype_, p_in_tmp, p_weight_tmp,
                               p_bias_tmp, (void *)p_out_tmp, &conv_attrs);
      }
    }
  }
  return ret;
}
#endif // _DEQCONV2DINT_VENUS_H_
