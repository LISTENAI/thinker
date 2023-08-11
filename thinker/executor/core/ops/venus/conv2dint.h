#ifndef _CONV2DINT_VENUS_H_
#define _CONV2DINT_VENUS_H_

#include <math.h>
#include <stdint.h>

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "luna/luna_math.h"
#include "thinker_status.h"

static int32_t luna_quant_ceil(int32_t x, int32_t shift) {
  if (x & ~(0xFFFFFFFF << shift)) {
    return (x >> shift) + 1;
  } else {
    return (x >> shift);
  }
}

static int32_t calc_conv_luna(int32_t w_dtype, int32_t y_dtype, int8_t *input,
                              int8_t *weight, int32_t *bias, void *output,
                              s_conv_struct *conv_attrs) {
  int32_t ret = 0;
  switch (w_dtype) {
    case Int4:
      switch (y_dtype) {
        case Int8:
          ret = luna_conv_intx_int8((const int8_t *)input, (int8_t *)weight,
                                    (int32_t *)bias, (int8_t *)output,
                                    conv_attrs, 4);
          break;
        case Int16:
          ret = luna_conv_intx_int16((const int8_t *)input, (int8_t *)weight,
                                     (int32_t *)bias, (int16_t *)output,
                                     conv_attrs, 4);
          break;
        case Int32:
          ret = luna_conv_intx_int32((const int8_t *)input, (int8_t *)weight,
                                     (int32_t *)bias, (int32_t *)output,
                                     conv_attrs, 4);
          break;
      }
      break;
    case Int8:
      switch (y_dtype) {
        case Int8:
          ret =
              luna_conv_q7_int8((const int8_t *)input, (int8_t *)weight,
                                (int32_t *)bias, (int8_t *)output, conv_attrs);
          break;
        case Int16:
          ret = luna_conv_q7_int16((const int8_t *)input, (int8_t *)weight,
                                   (int32_t *)bias, (int16_t *)output,
                                   conv_attrs);
          break;
        case Int32:
          ret = luna_conv_q7_int32((const int8_t *)input, (int8_t *)weight,
                                   (int32_t *)bias, (int32_t *)output,
                                   conv_attrs);
          break;
      }
      break;
  }

  return ret;
}

static int32_t calc_depthwise_luna(int32_t w_dtype, int32_t y_dtype,
                                   int8_t *input, int8_t *weight, int32_t *bias,
                                   void *output, s_conv_struct *conv_attrs) {
  int32_t ret = 0;
  switch (w_dtype) {
    case Int4:
      switch (y_dtype) {
        case Int8:
          ret = luna_depthwise_conv_intx_int8((const int8_t *)input,
                                              (int8_t *)weight, (int32_t *)bias,
                                              (int8_t *)output, conv_attrs, 4);
          break;
        case Int16:
          ret = luna_depthwise_conv_intx_int16(
              (const int8_t *)input, (int8_t *)weight, (int32_t *)bias,
              (int16_t *)output, conv_attrs, 4);
          break;
        case Int32:
          ret = luna_depthwise_conv_intx_int32(
              (const int8_t *)input, (int8_t *)weight, (int32_t *)bias,
              (int32_t *)output, conv_attrs, 4);
          break;
      }
      break;
    case Int8:
      switch (y_dtype) {
        case Int8:
          ret = luna_depthwise_conv_q7_int8((const int8_t *)input,
                                            (int8_t *)weight, (int32_t *)bias,
                                            (int8_t *)output, conv_attrs);
          break;
        case Int16:
          ret = luna_depthwise_conv_q7_int16((const int8_t *)input,
                                             (int8_t *)weight, (int32_t *)bias,
                                             (int16_t *)output, conv_attrs);
          break;
        case Int32:
          ret = luna_depthwise_conv_q7_int32((const int8_t *)input,
                                             (int8_t *)weight, (int32_t *)bias,
                                             (int32_t *)output, conv_attrs);
          break;
      }
      break;
  }
  return ret;
}

static int32_t calc_split_cnn_luna(int32_t w_dtype, int32_t y_dtype,
                                   int8_t *input, int8_t *weight, int32_t *bias,
                                   void *output, s_conv_struct *conv_attrs) {
  int32_t ret = 0;

  switch (w_dtype) {
    case Int8:
      switch (y_dtype) {
        case Int8:
          ret = luna_conv_split_q7_int8((const int8_t *)input, (int8_t *)weight,
                                        (int32_t *)bias, (int8_t *)output,
                                        conv_attrs);
          break;
        case Int16:
          ret = luna_conv_split_q7_int16((const int8_t *)input,
                                         (int8_t *)weight, (int32_t *)bias,
                                         (int16_t *)output, conv_attrs);
          break;
        case Int32:
          ret = luna_conv_split_q7_int32((const int8_t *)input,
                                         (int8_t *)weight, (int32_t *)bias,
                                         (int32_t *)output, conv_attrs);
          break;
      }
      break;
  }

  return ret;
}

static int32_t calc_pointwise_luna(int32_t w_dtype, int32_t y_dtype, int8_t *input,
                                  int8_t *weight, int32_t *bias, void *output, void *work_space,
                                  s_conv_struct *conv_attrs) {
  int32_t ret = -1;

  int32_t in_c = conv_attrs->input_c;
  int32_t in_h = conv_attrs->input_h;
  int32_t in_w = conv_attrs->input_w;
  int32_t ou_c = conv_attrs->output_c;
  int32_t ou_h = conv_attrs->output_h;
  int32_t ou_w = conv_attrs->output_w;
  int32_t k_n = ou_c;
  int32_t k_c = in_c;

  if (in_h != ou_h || in_w != ou_w)
  {
    return -1;
  }

  // img2col, Y = W * X
  const int32_t left_limit = 64 * 1024;
  const int32_t right_limit = 32 * 1024;
  int32_t M = k_n;
  int32_t N = k_c;
  int32_t L = in_h * in_w;
  int32_t shift = conv_attrs->positive_shift_value;
  int32_t is_relu = (conv_attrs->activation_type == RELU) ? 1 : 0;
  int32_t is_bias = conv_attrs->is_bias;

  switch (w_dtype) {
    case Int4:  //not support
      break;
    case Int8:
    {
      int32_t int8_condition_l = (luna_quant_ceil(M, 2) << 2) * (luna_quant_ceil(N, 3) << 3);  // right:4x8
      if (int8_condition_l > left_limit) {
        return ret;
      }
      int32_t int8_condition_r = (luna_quant_ceil(N, 3) << 3) * (luna_quant_ceil(L, 2) << 2);  // right:8x4
      if (int8_condition_r <= right_limit) {
        if (is_bias) {
          int32_t *p_tmp = (int32_t *)work_space;
          ret = luna_mat_mul_q7_int32(weight, input, (int32_t *)p_tmp, M, N, L, 0);
        } else {
          ret = luna_mat_mul_q7_int8(weight, input, (int8_t *)output, M, N, L, shift);
        }
      }
      else {  // big martrix split on col
        int32_t split_num = 2;
        int32_t split_L = L / split_num;
        int8_condition_r =
            (luna_quant_ceil(N, 3) << 3) * (luna_quant_ceil(split_L, 2) << 2);  // right:8x4
        while (int8_condition_r > right_limit || (0 != (L % split_num))) {
          split_num++;
          split_L = L / split_num;
          int8_condition_r = (luna_quant_ceil(N, 3) << 3) * (luna_quant_ceil(split_L, 2) << 2);  // right:8x4
        }
        {
          if (is_bias) {
            int32_t *p_tmp = (int32_t *)work_space;
            ret = luna_split_mat_mul_q7_int32(weight, input, (int32_t *)p_tmp, split_num, M, N, L, 0);
          } else {
            ret = luna_split_mat_mul_q7_int8(weight, input, (int8_t *)output, split_num, M, N, L, shift);
          }
        }
      }
      if (is_bias) {
        int32_t *p_tmp = (int32_t *)work_space + M * L;
        ret = luna_mat_trans_q31((int32_t*)work_space, (int32_t*)p_tmp, M, L);
        for (int32_t i = 0; i < L; i++)  // add bias
        {
          int32_t *tsrc1 = (int32_t *)p_tmp + i * M;
          int8_t *tdst = (int8_t *)work_space + i * M;
          ret |= luna_add_q31_int8(tsrc1, bias, tdst, M, shift);
        }
        ret = luna_mat_trans_q7((int8_t*)work_space, (int8_t*)output, L, M);
      }
      if (is_relu) {
        ret |= luna_relu_q7_int8((int8_t *)output, (int8_t *)output, M * L, 0);
      }
    }
      break;
  }

  return ret;
}

static void conv2dint_venus_para_init(Conv2dIntAttrs *attrs,
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
  conv_attrs->padding_h_up = attrs->pad[0];
  conv_attrs->padding_h_down = attrs->pad[2];
  conv_attrs->padding_w_left = attrs->pad[1];
  conv_attrs->padding_w_right = attrs->pad[3];
  conv_attrs->input_h_after_padding = conv_attrs->input_h +
                                      conv_attrs->padding_h_up +
                                      conv_attrs->padding_h_down;
  conv_attrs->input_w_after_padding = conv_attrs->input_w +
                                      conv_attrs->padding_w_left +
                                      conv_attrs->padding_w_right;
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

int32_t conv2dint_venus(tTensor *X, tTensor *W, tTensor *Bias, tTensor *Y,
                        tTensor *Temp, Conv2dIntAttrs *attrs) {
  int32_t ret = T_ERR_FAIL;

  uint64_t paddr_b = 0;
  if (Bias != NULL) {
    paddr_b = Bias->dptr_;
  }

  s_conv_struct conv_attrs;
  conv2dint_venus_para_init(attrs, &conv_attrs, X, W, Bias, Y);
  int32_t n = 0;
  int32_t batch = X->shape_.dims_[0];
  int32_t group_num = attrs->group;
  int32_t in_c = conv_attrs.input_c;
  int32_t in_h = conv_attrs.input_h;
  int32_t in_w = conv_attrs.input_w;
  int32_t k_n = W->shape_.dims_[0];
  int32_t k_c = W->shape_.dims_[3];
  int32_t k_h = conv_attrs.weight_h;
  int32_t k_w = conv_attrs.weight_w;
  int32_t ou_c = conv_attrs.output_c;
  int32_t ou_h = conv_attrs.output_h;
  int32_t ou_w = conv_attrs.output_w;
  int32_t s_h = conv_attrs.stride_h;
  int32_t s_w = conv_attrs.stride_w;
  int32_t padding_hd = conv_attrs.padding_h_down;
  int32_t padding_hu = conv_attrs.padding_h_up;
  int32_t padding_wl = conv_attrs.padding_w_left;
  int32_t padding_wr = conv_attrs.padding_w_right;
  int32_t dilation_h = attrs->dilation[0];
  int32_t dilation_w = attrs->dilation[1];
  int32_t log2n_stride_w = (conv_attrs.stride_w >> 1);
  int32_t input_condition =
      (luna_quant_ceil(in_c, 3) << 3) * in_h *
      (luna_quant_ceil(in_w, (3 + log2n_stride_w)) << (3 + log2n_stride_w));
  int32_t kernel_condition = (luna_quant_ceil(in_c, 3) << 3) * k_h * k_w *
                             (luna_quant_ceil(ou_c, 1) << 1);
  kernel_condition = (kernel_condition <= 32 * 1024) ? 1 : 0;
  input_condition = (input_condition <= 64 * 1024) ? 1 : 0;

  if (!ou_c || !ou_h || !ou_w) {
    return ret;
  }

  if (dilation_h > 1 || dilation_w > 1) //dilation conv
  {
    int8_t *p_src = (int8_t *)X->dptr_;
    if (dilation_h > 1)
    {
      int32_t in_hw = in_w * in_h;
      int32_t ou_hw = in_w * k_h;
      int32_t d_hw = dilation_h * in_w;
      for (int i = 0; i < in_c; i++)
      {
        int32_t i_off = i * in_hw;
        int32_t o_off = i * ou_hw;
        for (int j = 0; j < k_h; j++)
  		    memcpy(p_src + o_off + j * in_w, p_src + i_off + j * d_hw, in_w);
      }
      in_h = k_h;
      conv_attrs.input_h = k_h;
      conv_attrs.input_h_after_padding = conv_attrs.input_h + conv_attrs.padding_h_up + conv_attrs.padding_h_down;
      input_condition =
      (luna_quant_ceil(in_c, 3) << 3) * in_h *
      (luna_quant_ceil(in_w, (3 + log2n_stride_w)) << (3 + log2n_stride_w));
      input_condition = (input_condition <= 64 * 1024) ? 1 : 0;
    }
    else
    {
      // C*H*W -> C*W*H
      int8_t *p_tmp = (int8_t *)Temp->dptr_;
      uint32_t axis[3] = {0, 2, 1};
      uint32_t in_shape[3] = {in_c, in_h, in_w};
      luna_trans_axis_q7(p_src, p_tmp, in_shape, axis, 3);

      int32_t in_hw = in_w * in_h;
      int32_t ou_hw = in_h * k_w;
      int32_t d_hw = dilation_w * in_h;
      for (int i = 0; i < in_c; i++)
      {
        int32_t i_off = i * in_hw;
        int32_t o_off = i * ou_hw;
        for (int j = 0; j < k_w; j++)
        {
          memcpy(p_tmp + o_off + j * in_h, p_tmp + i_off + j * d_hw, in_h);
        }
      }
      in_w = k_w;
      conv_attrs.input_h = k_h;
      conv_attrs.input_h_after_padding = conv_attrs.input_h + conv_attrs.padding_h_up + conv_attrs.padding_h_down;
      uint32_t new_in_shape[3] = {in_c, in_w, in_h};
      luna_trans_axis_q7(p_tmp, p_src, in_shape, axis, 3);
    }
  }  
  
  if (1 == attrs->group) {                      // conv
    if (input_condition && kernel_condition) {  // no need split
      int32_t in_batch_size = in_c * in_h * in_w;
      int32_t ou_batch_size = ou_c * ou_h * ou_w * (Y->dtype_ & 0xF);
      if (X->dtype_)
      for (n = 0; n < batch; n++) {
        int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
        if (X->mem_.type_ != 2)
        {
         p_in = (int8_t *)Temp->dptr_;
         memcpy(p_in, (int8_t *)X->dptr_ +  n * in_batch_size, in_batch_size);
        }
        int8_t *p_weight = (int8_t *)W->dptr_;
        int32_t *p_bias = (0 == paddr_b) ? NULL : ((int32_t *)paddr_b);
        int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
        if (Y->mem_.type_ != 2)
        {
         p_out = (int8_t *)Temp->dptr_  + in_batch_size * (X->mem_.type_ != 2);
        }
        ret = calc_conv_luna(W->dtype_, Y->dtype_, p_in, p_weight, p_bias,
                             (void *)p_out, &conv_attrs);
        if (Y->mem_.type_ != 2)
        {
         memcpy((int8_t *)Y->dptr_ +  n * ou_batch_size, p_out, ou_batch_size);
        }
      }
    } 
    else if (input_condition && !kernel_condition) {  // split weight N
      int32_t c_in_align_8 = (((in_c + 7) >> 3) << 3);
      int32_t c_out_max = (32768 / (c_in_align_8 * k_h * k_w)) &
                          0xFFFFFFFE;  // max kernel size is 32KB
      int32_t split_num = (ou_c + c_out_max - 1) / c_out_max;
      int32_t tmp_ou_c = ((ou_c + 2 * split_num - 1) / split_num) & 0xFFFFFFFE;
      int32_t step_data_out = tmp_ou_c * ou_h * ou_w * (0xF & Y->dtype_);
      int32_t step_weight = (k_n / split_num) * k_c * k_h * k_w;
      int32_t step_bias = (0 == paddr_b) ? 0 : tmp_ou_c;
      int32_t i;
      for (n = 0; n < batch; n++) {
        conv_attrs.output_c = tmp_ou_c;
        int32_t in_batch_size = in_c * in_h * in_w;
        int32_t ou_batch_size = ou_c * ou_h * ou_w * (0xF & Y->dtype_);
        int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
        int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
        int8_t *p_weight = (int8_t *)W->dptr_;
        int32_t *p_bias = (0 == paddr_b) ? NULL : ((int32_t *)paddr_b);
        for (i = 0; i < split_num; i++) {
          int8_t *p_in_tmp = p_in;
          if (X->mem_.type_ != 2)
          {
            p_in_tmp = (int8_t *)Temp->dptr_;
            memcpy(p_in_tmp, (int8_t *)p_in, in_batch_size);
          }
          int8_t *p_out_tmp = p_out + i * step_data_out;
          if (Y->mem_.type_ != 2)
          {
            p_out_tmp = (int8_t *)Temp->dptr_  + in_batch_size * (X->mem_.type_ != 2);
          }
          int8_t *p_weight_tmp = p_weight + i * step_weight;
          int32_t *p_bias_tmp = p_bias + i * step_bias;
          if ((ou_c != k_n) && (i == (split_num - 1))) {
            conv_attrs.output_c = tmp_ou_c - (k_n - ou_c);
          }
          ret = calc_conv_luna(W->dtype_, Y->dtype_, p_in_tmp, p_weight_tmp,
                               p_bias_tmp, (void *)p_out_tmp, &conv_attrs);
          if (Y->mem_.type_ != 2)
          {
            memcpy(p_out + + i * step_data_out + n * ou_batch_size, p_out_tmp, step_data_out);
          }
        }
      }
    } 
    else if (!input_condition && kernel_condition) {  // split input H/W
      if ((in_h * in_w >= 64 * 1024) || (X->mem_.type_ != 2) || (Y->mem_.type_ != 2)){
        /////only support H
        int32_t input_limit_without_h =
            (luna_quant_ceil(in_c, 3) << 3) *
            (luna_quant_ceil(in_w, (3 + log2n_stride_w))
             << (3 + log2n_stride_w));
        int32_t split_num = 1;
        int32_t tmp_in_h = in_h;
        while ((tmp_in_h * input_limit_without_h > 65536) ||
               ((ou_h % split_num) != 0)) {
          split_num += 1;
          tmp_in_h = (ou_h * s_h) / split_num + k_h - s_h;
          if ((split_num > in_h) || (split_num > ou_h)) {
            break;
          }
        }
        int32_t cal_var0 = 0;
        while (padding_hd) {
          cal_var0 = in_h + padding_hu + padding_hd - k_h + s_h;
          if (cal_var0 % s_h) {
            padding_hd = padding_hd - 1;
          } else {
            break;
          }
        }
        int32_t tmp_ou_h = ou_h / split_num;
        int32_t in_h_1st = tmp_in_h - padding_hu;
        int32_t in_h_last = tmp_in_h - padding_hd;
        int32_t pad_h_down_1st = 0;
        int32_t pad_h_down_last = 0;
        int32_t pad_h_down_mid = 0;
        int32_t in_addr_offset_1st = in_w * (tmp_in_h - k_h + s_h - padding_hu);
        int32_t in_addr_offset = in_w * (tmp_in_h - k_h + s_h);
        int32_t ou_addr_offset = ou_c * ou_w * tmp_ou_h * (0xF & Y->dtype_);
        if ((in_h_1st + padding_hu) < ((tmp_ou_h - 1) * s_h + k_h)) {
          pad_h_down_1st = (tmp_ou_h - 1) * s_h + k_h - in_h_1st - padding_hu;
        }
        if (in_h_last < ((tmp_ou_h - 1) * s_h + k_h)) {
          pad_h_down_last = (tmp_ou_h - 1) * s_h + k_h - in_h_last;
        }
        if (tmp_in_h < ((tmp_ou_h - 1) * s_h + k_h)) {
          pad_h_down_mid = (tmp_ou_h - 1) * s_h + k_h - tmp_in_h;
        }
        ///////////////
        int32_t in_batch_size = in_c * in_h * in_w;
        int32_t ou_batch_size = ou_c * ou_h * ou_w * (Y->dtype_ & 0xF);
        int32_t i, j;
        for (n = 0; n < batch; n++) {
          int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
          int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
          int8_t *p_weight = (int8_t *)W->dptr_;
          int32_t *p_bias = (0 == paddr_b) ? NULL : ((int32_t *)paddr_b);
          int8_t *p_in_tmp = p_in;
          int8_t *p_out_tmp = p_out;
          int8_t *p_tmp = (int8_t *)Temp->dptr_;
          for (i = 0; i < split_num; i++) {
            if (i == 0) {
              conv_attrs.input_h = in_h_1st;
              conv_attrs.padding_h_up = padding_hu;
              conv_attrs.padding_h_down = pad_h_down_1st;
              p_in_tmp = p_in;
            } else if (i == (split_num - 1)) {
              conv_attrs.input_h = in_h_last;
              conv_attrs.padding_h_up = 0;
              conv_attrs.padding_h_down = pad_h_down_last;
              p_in_tmp = p_in + in_addr_offset_1st + (i - 1) * in_addr_offset;
            } else {
              conv_attrs.input_h = tmp_in_h;
              conv_attrs.padding_h_up = 0;
              conv_attrs.padding_h_down = pad_h_down_mid;
              p_in_tmp = p_in + in_addr_offset_1st + (i - 1) * in_addr_offset;
            }
            conv_attrs.input_h_after_padding = conv_attrs.input_h +
                                               conv_attrs.padding_h_up +
                                               conv_attrs.padding_h_down;
            conv_attrs.output_h = tmp_ou_h;
            p_out_tmp = p_out + i * ou_addr_offset;

            int32_t c;
            int32_t o_offset = in_w * conv_attrs.input_h;
            int32_t i_offset = in_w * in_h;
            for (c = 0; c < in_c; c++) {
              memcpy(p_tmp + c * o_offset, p_in_tmp + c * i_offset, o_offset);
            }
            ret = calc_conv_luna(W->dtype_, Y->dtype_, p_tmp, p_weight, p_bias,
                                 (void *)p_tmp, &conv_attrs);

            int32_t one_channel_ou_offset = ou_w * tmp_ou_h * (0xF & Y->dtype_);
            int32_t ou_hw = ou_h * ou_w;
            o_offset = i * one_channel_ou_offset;
            for (j = 0; j < ou_c; j++) {
              memcpy(p_out + o_offset + j * ou_hw, p_tmp + j * one_channel_ou_offset, one_channel_ou_offset);
            }
          }
        }
      } 
      else {
        /////only support H
        int32_t in_batch_size = in_c * in_h * in_w;
        int32_t ou_batch_size = ou_c * ou_h * ou_w * (Y->dtype_ & 0xF);
        for (n = 0; n < batch; n++) {
          int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
          int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
          int8_t *p_weight = (int8_t *)W->dptr_;
          int32_t *p_bias = (0 == paddr_b) ? NULL : ((int32_t *)paddr_b);
          ret = calc_split_cnn_luna(W->dtype_, Y->dtype_, p_in, p_weight,
                                    p_bias, (void *)p_out, &conv_attrs);
        }
      }
    }
  } 
  else if (attrs->group == conv_attrs.input_c &&
             attrs->group == conv_attrs.output_c) {  // depthwise
    kernel_condition = (luna_quant_ceil(in_c, 4) << 4) * k_h * k_w;
    kernel_condition = (kernel_condition <= 32 * 1024) ? 1 : 0;
    if (input_condition && kernel_condition) {  // no need split
      int32_t in_batch_size = in_c * in_h * in_w;
      int32_t ou_batch_size = ou_c * ou_h * ou_w * (Y->dtype_ & 0xF);
      for (n = 0; n < batch; n++) {
        int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
        if (X->mem_.type_ != 2)
        {
         p_in = (int8_t *)Temp->dptr_;
         memcpy(p_in, (int8_t *)X->dptr_ +  n * in_batch_size, in_batch_size);
        }
        int8_t *p_weight = (int8_t *)W->dptr_;
        int32_t *p_bias = (0 == paddr_b) ? NULL : ((int32_t *)paddr_b);
        int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
        if (Y->mem_.type_ != 2)
        {
         p_out = (int8_t *)Temp->dptr_  + in_batch_size * (X->mem_.type_ != 2);
        }
        ret = calc_depthwise_luna(W->dtype_, Y->dtype_, p_in, p_weight, p_bias,
                                  (void *)p_out, &conv_attrs);
        if (Y->mem_.type_ != 2)
        {
         memcpy((int8_t *)Y->dptr_ +  n * ou_batch_size, p_out, ou_batch_size);
        }
      }
    } 
    else if (!input_condition && kernel_condition) {  // split input H/W
      /////only support H
      int32_t input_limit_without_h =
          (luna_quant_ceil(in_c, 3) << 3) *
          (luna_quant_ceil(in_w, (3 + log2n_stride_w)) << (3 + log2n_stride_w));
      int32_t split_num = 1;
      int32_t tmp_in_h = in_h;
      while ((tmp_in_h * input_limit_without_h > 65536) ||
             ((ou_h % split_num) != 0)) {
        split_num += 1;
        tmp_in_h = (ou_h * s_h) / split_num + k_h - s_h;
        if ((split_num > in_h) || (split_num > ou_h)) {
          break;
        }
      }
      int32_t cal_var0 = 0;
      while (padding_hd) {
        cal_var0 = in_h + padding_hu + padding_hd - k_h + s_h;
        if (cal_var0 % s_h) {
          padding_hd = padding_hd - 1;
        } else {
          break;
        }
      }
      int32_t tmp_ou_h = ou_h / split_num;
      int32_t in_h_1st = tmp_in_h - padding_hu;
      int32_t in_h_last = tmp_in_h - padding_hd;
      int32_t pad_h_down_1st = 0;
      int32_t pad_h_down_last = 0;
      int32_t pad_h_down_mid = 0;
      int32_t in_addr_offset_1st = in_w * (tmp_in_h - k_h + s_h - padding_hu);
      int32_t in_addr_offset = in_w * (tmp_in_h - k_h + s_h);
      int32_t ou_addr_offset = ou_c * ou_w * tmp_ou_h * (0xF & Y->dtype_);
      if ((in_h_1st + padding_hu) < ((tmp_ou_h - 1) * s_h + k_h)) {
        pad_h_down_1st = (tmp_ou_h - 1) * s_h + k_h - in_h_1st - padding_hu;
      }
      if (in_h_last < ((tmp_ou_h - 1) * s_h + k_h)) {
        pad_h_down_last = (tmp_ou_h - 1) * s_h + k_h - in_h_last;
      }
      if (tmp_in_h < ((tmp_ou_h - 1) * s_h + k_h)) {
        pad_h_down_mid = (tmp_ou_h - 1) * s_h + k_h - tmp_in_h;
      }
      ///////////////
      int32_t in_batch_size = in_c * in_h * in_w;
      int32_t ou_batch_size = ou_c * ou_h * ou_w * (Y->dtype_ & 0xF);
      int32_t i, j;
      for (n = 0; n < batch; n++) {
        int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
        int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
        int8_t *p_weight = (int8_t *)W->dptr_;
        int32_t *p_bias = (0 == paddr_b) ? NULL : ((int32_t *)paddr_b);
        int8_t *p_in_tmp = p_in;
        int8_t *p_out_tmp = p_out;
        int8_t *p_tmp = (int8_t *)Temp->dptr_;
        for (i = 0; i < split_num; i++) {
          if (i == 0) {
            conv_attrs.input_h = in_h_1st;
            conv_attrs.padding_h_up = padding_hu;
            conv_attrs.padding_h_down = pad_h_down_1st;
            p_in_tmp = p_in;
          } else if (i == (split_num - 1)) {
            conv_attrs.input_h = in_h_last;
            conv_attrs.padding_h_up = 0;
            conv_attrs.padding_h_down = pad_h_down_last;
            p_in_tmp = p_in + in_addr_offset_1st + (i - 1) * in_addr_offset;
          } else {
            conv_attrs.input_h = tmp_in_h;
            conv_attrs.padding_h_up = 0;
            conv_attrs.padding_h_down = pad_h_down_mid;
            p_in_tmp = p_in + in_addr_offset_1st + (i - 1) * in_addr_offset;
          }
          conv_attrs.input_h_after_padding = conv_attrs.input_h +
                                             conv_attrs.padding_h_up +
                                             conv_attrs.padding_h_down;
          conv_attrs.output_h = tmp_ou_h;

          p_out_tmp = p_out + i * ou_addr_offset;
          if (Y->mem_.type_ != 2)
            p_out_tmp = p_tmp + in_c * in_w * tmp_in_h + i * ou_addr_offset;

          int32_t c;
          int32_t o_offset = in_w * conv_attrs.input_h;
          int32_t i_offset = in_w * in_h;
          for (c = 0; c < in_c; c++) {
            memcpy(p_tmp + c * o_offset, p_in_tmp + c * i_offset, o_offset);
          }
          ret = calc_depthwise_luna(W->dtype_, Y->dtype_, p_tmp, p_weight,
                                    p_bias, (void *)p_out_tmp, &conv_attrs);
        }

        int32_t one_channel_ou_offset = ou_w * tmp_ou_h * (0xF & Y->dtype_);
        if (Y->mem_.type_ != 2){        
          for (j = 0; j < ou_c; j++) {
              int32_t i_offset = in_c * in_w * tmp_in_h + j * one_channel_ou_offset;
              int32_t o_offset = j * ou_w * ou_h;
            for (i = 0; i < split_num; i++) {
              memcpy(p_out + o_offset + i * one_channel_ou_offset, p_tmp + i_offset + i * ou_addr_offset, one_channel_ou_offset);
            }
          }
        }
        else{
          for (j = 0; j < ou_c; j++) {
            for (i = 0; i < split_num; i++) {
              int32_t i_offset = i * ou_addr_offset + j * one_channel_ou_offset;
              int32_t o_offset = i * one_channel_ou_offset + j * ou_w * ou_h;
              memcpy(p_tmp + o_offset, p_out + i_offset, one_channel_ou_offset);
            }
          }
          memcpy(p_out, p_tmp, ou_c * ou_h * ou_w);
        }
      }
    }
  } 
  else {  // group conv
    in_c = in_c / group_num;
    ou_c = ou_c / group_num;
    k_n = k_n / group_num;
    input_condition =
        (luna_quant_ceil(in_c, 3) << 3) * in_h *
        (luna_quant_ceil(in_w, (3 + log2n_stride_w) << (3 + log2n_stride_w)));
    kernel_condition = (luna_quant_ceil(in_c, 3) << 3) * k_h * k_w *
                       (luna_quant_ceil(ou_c, 1) << 1);
    kernel_condition = (kernel_condition <= 32 * 1024) ? 1 : 0;
    input_condition = (input_condition <= 64 * 1024) ? 1 : 0;
    if (input_condition && kernel_condition) {
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
          ret = calc_conv_luna(W->dtype_, Y->dtype_, p_in_tmp, p_weight_tmp,
                               p_bias_tmp, (void *)p_out_tmp, &conv_attrs);
        }
      }
    } 
    else if (!input_condition && kernel_condition) {  // split input H/W
      /////only support H
      int32_t input_limit_without_h =
          (luna_quant_ceil(in_c, 3) << 3) *
          (luna_quant_ceil(in_w, (3 + log2n_stride_w)) << (3 + log2n_stride_w));
      int32_t split_num = 1;
      int32_t tmp_in_h = in_h;
      while ((tmp_in_h * input_limit_without_h > 65536) ||
             ((ou_h % split_num) != 0)) {
        split_num += 1;
        tmp_in_h = (ou_h * s_h) / split_num + k_h - s_h;
        if ((split_num > in_h) || (split_num > ou_h)) {
          break;
        }
      }
      int32_t cal_var0 = 0;
      while (padding_hd) {
        cal_var0 = in_h + padding_hu + padding_hd - k_h + s_h;
        if (cal_var0 % s_h) {
          padding_hd = padding_hd - 1;
        } else {
          break;
        }
      }
      int32_t tmp_ou_h = ou_h / split_num;
      int32_t in_h_1st = tmp_in_h - padding_hu;
      int32_t in_h_last = tmp_in_h - padding_hd;
      int32_t pad_h_down_1st = 0;
      int32_t pad_h_down_last = 0;
      int32_t pad_h_down_mid = 0;
      int32_t in_addr_offset_1st = in_w * (tmp_in_h - k_h + s_h - padding_hu);
      int32_t in_addr_offset = in_w * (tmp_in_h - k_h + s_h);
      int32_t ou_addr_offset = ou_c * ou_w * tmp_ou_h * (0xF & Y->dtype_);
      if ((in_h_1st + padding_hu) < ((tmp_ou_h - 1) * s_h + k_h)) {
        pad_h_down_1st = (tmp_ou_h - 1) * s_h + k_h - in_h_1st - padding_hu;
      }
      if (in_h_last < ((tmp_ou_h - 1) * s_h + k_h)) {
        pad_h_down_last = (tmp_ou_h - 1) * s_h + k_h - in_h_last;
      }
      if (tmp_in_h < ((tmp_ou_h - 1) * s_h + k_h)) {
        pad_h_down_mid = (tmp_ou_h - 1) * s_h + k_h - tmp_in_h;
      }
      ///////////////
      int32_t step_data_in = in_c * in_h * in_w * (0xF & X->dtype_);
      int32_t step_data_out = ou_c * ou_h * ou_w * (0xF & Y->dtype_);
      int32_t step_weight = k_n * k_c * k_h * k_w;
      int32_t step_bias = (0 == paddr_b) ? 0 : ou_c;
      int32_t i, j, k;
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
        for (k = 0; k < group_num; k++) {
          int8_t *p_in_group = p_in + k * step_data_in;
          int8_t *p_out_group = p_out + k * step_data_out;
          int8_t *p_weight_group = p_weight + k * step_weight;
          int32_t *p_bias_group = p_bias + k * step_bias;
          int8_t *p_tmp = (int8_t *)Temp->dptr_;
          int8_t *p_in_tmp = NULL;
          int8_t *p_out_tmp = NULL;
          for (i = 0; i < split_num; i++) {
            if (i == 0) {
              conv_attrs.input_h = in_h_1st;
              conv_attrs.padding_h_up = padding_hu;
              conv_attrs.padding_h_down = pad_h_down_1st;
              p_in_tmp = p_in_group;
            } else if (i == (split_num - 1)) {
              conv_attrs.input_h = in_h_last;
              conv_attrs.padding_h_up = 0;
              conv_attrs.padding_h_down = pad_h_down_last;
              p_in_tmp =
                  p_in_group + in_addr_offset_1st + (i - 1) * in_addr_offset;
            } else {
              conv_attrs.input_h = tmp_in_h;
              conv_attrs.padding_h_up = 0;
              conv_attrs.padding_h_down = pad_h_down_mid;
              p_in_tmp =
                  p_in_group + in_addr_offset_1st + (i - 1) * in_addr_offset;
            }
            conv_attrs.input_h_after_padding = conv_attrs.input_h +
                                               conv_attrs.padding_h_up +
                                               conv_attrs.padding_h_down;
            conv_attrs.output_h = tmp_ou_h;
            p_out_tmp = p_out_group + i * ou_addr_offset;

            int32_t c;
            int32_t o_offset = in_w * conv_attrs.input_h;
            int32_t i_offset = in_w * in_h;
            for (c = 0; c < in_c; c++) {
              memcpy(p_tmp + c * o_offset, p_in_tmp + c * i_offset, o_offset);
            }
            ret = calc_conv_luna(W->dtype_, Y->dtype_, p_tmp, p_weight_group,
                                 p_bias_group, (void *)p_out_tmp, &conv_attrs);
          }

          int32_t one_channel_ou_offset = ou_w * tmp_ou_h * (0xF & Y->dtype_);
          for (j = 0; j < ou_c; j++) {
            int32_t i_offset = j * one_channel_ou_offset;
            int32_t o_offset = j * ou_w * ou_h;
            for (i = 0; i < split_num; i++) {
              memcpy(p_tmp + i * one_channel_ou_offset, p_out_group + i * ou_addr_offset,
                    one_channel_ou_offset);
            }
          }
          memcpy(p_out_group, p_tmp, ou_c * ou_h * ou_w);
        }
      }
    }
  }
  return ret;
}
#endif  //_CONV2DINT_VENUS_H_
