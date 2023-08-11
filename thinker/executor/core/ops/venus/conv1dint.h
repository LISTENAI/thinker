#ifndef _CONV1DINT_VENUS_H_
#define _CONV1DINT_VENUS_H_

#include <math.h>
#include <stdint.h>

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "luna/luna_math.h"
#include "thinker_status.h"

typedef int32_t (*CONV1D_MAT_MUL_LUNA_API)(int8_t *src1, int8_t *src2, void *dst,
                                       int32_t row, int32_t col, int32_t col2,
                                       int32_t shift);
typedef int32_t (*CONV1D_SPLIT_MAT_MUL_LUNA_API)(int8_t *src1, int8_t *src2,
                                             void *dst, int32_t split_num,
                                             int32_t row, int32_t col,
                                             int32_t col2, int32_t shift);
typedef int32_t (*CONV1D_VEC_ADD_LUNA_API)(void *src1, void *src2, void *dst,
                                       int32_t size, int32_t shift);

struct conv1d_luna_api_item {
  void *luna_api;
};

struct conv1d_luna_api_item conv1d_luna_api_list[][3] = {
    {{luna_mat_mul_q7_int8}, {luna_mat_mul_q7_int16}, {luna_mat_mul_q7_int32}},
    {{luna_split_mat_mul_q7_int8},
     {luna_split_mat_mul_q7_int16},
     {luna_split_mat_mul_q7_int32}},
    {{luna_add_q7_int8}, {luna_add_q7_int16}, {luna_add_q7_int32}},
    {{luna_add_q15_int8}, {luna_add_q15_int16}, {luna_add_q15_int32}},
    {{luna_add_q31_int8}, {luna_add_q31_int16}, {luna_add_q31_int32}}};

static int32_t luna_quant_ceil(int32_t x, int32_t shift) {
  if (x & ~(0xFFFFFFFF << shift)) {
    return (x >> shift) + 1;
  } else {
    return (x >> shift);
  }
}

void dump_int8(const char *name, int8_t *data, int height, int width)
{
	//cwh
	//width大小包含了width_history
	FILE* fp;
	int c, h, w, i;

	fp = fopen(name, "w");

		for (h = 0; h < height; h++)
		{
			for (w = 0; w < width; w++)
			{
					fprintf(fp, "%d\t", data[h * width + w]);
			}
			fprintf(fp, "\n");
		}

	fclose(fp);
}

static int32_t img2col(int8_t *src, int8_t *dst, int32_t channel, int32_t height, int32_t kernel, int32_t stride){
  int32_t ret = luna_mat_trans_q7(src, src, channel, height);
  int32_t num = (height - kernel) / stride + 1;
  int32_t data_size = channel * kernel;
  int32_t step_size = channel * stride;
  for(int32_t i = 0; i < num; i++)
  {
    memcpy(dst + i * data_size, src + i *step_size, data_size);
  }
  return ret;
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

static void conv1dint_para_init(Conv1dIntAttrs *attrs,
                                s_conv_struct *conv_attrs, tTensor *X,
                                tTensor *W, tTensor *Bias, tTensor *Y) {
  memset(conv_attrs, 0, sizeof(s_conv_struct));
  if (NULL != Bias) {
    conv_attrs->is_bias = 1;
  }
  conv_attrs->input_c = X->shape_.dims_[1];
  conv_attrs->input_h = X->shape_.dims_[2];
  conv_attrs->input_w = 1;
  conv_attrs->output_c = Y->shape_.dims_[1];
  conv_attrs->output_h = Y->shape_.dims_[2];
  conv_attrs->output_w = 1;
  conv_attrs->weight_h = attrs->kernel;
  conv_attrs->weight_w = 1;
  conv_attrs->stride_h = attrs->stride;
  conv_attrs->stride_w = 1;
  conv_attrs->padding_h_up = attrs->pad[0];
  conv_attrs->padding_h_down = attrs->pad[1];
  conv_attrs->padding_w_left = 0;
  conv_attrs->padding_w_right = 0;
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

int32_t conv1dint_venus(tTensor *X, tTensor *W, tTensor *Bias, tTensor *Y,
                        tTensor *Temp, Conv1dIntAttrs *attrs) {
  int32_t ret = T_ERR_FAIL;

  uint64_t paddr_b = 0;
  int32_t has_bias = 0;
  int32_t bias_idx = 0;
  int32_t ou_idx = (Y->dtype_ & 0xF) >> 1;
  if (Bias != NULL) {
    bias_idx = (Bias->dtype_ & 0xF) >> 1;
    paddr_b = Bias->dptr_;
    has_bias = 1;
  }
  int32_t kernel = attrs->kernel;
  if (kernel < 6){
    s_conv_struct conv_attrs;
    conv1dint_para_init(attrs, &conv_attrs, X, W, Bias, Y);
    int32_t n = 0;
    int32_t batch = X->shape_.dims_[0];
    int32_t group_num = attrs->group;
    int32_t in_c = conv_attrs.input_c;
    int32_t in_h = conv_attrs.input_h;
    int32_t in_w = conv_attrs.input_w;
    int32_t k_n = W->shape_.dims_[0];
    int32_t k_c = W->shape_.dims_[1];
    int32_t k_h = conv_attrs.weight_h;
    int32_t k_w = conv_attrs.weight_w;
    int32_t ou_c = conv_attrs.output_c;
    int32_t ou_h = conv_attrs.output_h;
    int32_t ou_w = conv_attrs.output_w;
    int32_t s_h = conv_attrs.stride_h;
    // int32_t s_w = conv_attrs.stride_w;
    int32_t padding_hd = conv_attrs.padding_h_down;
    int32_t padding_hu = conv_attrs.padding_h_up;
    int32_t log2n_stride_w = (conv_attrs.stride_w >> 1);
    int32_t input_condition =
        (luna_quant_ceil(in_c, 3) << 3) * in_h *
        (luna_quant_ceil(in_w, (3 + log2n_stride_w)) << (3 + log2n_stride_w));
    int32_t kernel_condition = (luna_quant_ceil(in_c, 3) << 3) * k_h * k_w *
                              (luna_quant_ceil(ou_c, 1) << 1);
    kernel_condition = (kernel_condition <= 32 * 1024) ? 1 : 0;
    input_condition = (input_condition <= 64 * 1024) ? 1 : 0;

    if (!ou_c || !ou_h || !ou_w) {
      return 0;
    }

    if (1 == attrs->group) {                      // conv
      if (input_condition && kernel_condition) {  // no need split
        int32_t in_batch_size = in_c * in_h * in_w;
        int32_t ou_batch_size = ou_c * ou_h * ou_w * (Y->dtype_ & 0xF);
        for (n = 0; n < batch; n++) {
          int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
          int8_t *p_weight = (int8_t *)W->dptr_;
          int32_t *p_bias = (0 == paddr_b) ? NULL : ((int32_t *)paddr_b);
          int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
          ret = calc_conv_luna(W->dtype_, Y->dtype_, p_in, p_weight, p_bias,
                              (void *)p_out, &conv_attrs);
        }
      } else if (input_condition && !kernel_condition) {  // split weight N
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
            int8_t *p_out_tmp = p_out + i * step_data_out;
            int8_t *p_weight_tmp = p_weight + i * step_weight;
            int32_t *p_bias_tmp = p_bias + i * step_bias;
            if ((ou_c != k_n) && (i == (split_num - 1))) {
              conv_attrs.output_c = tmp_ou_c - (k_n - ou_c);
            }
            ret = calc_conv_luna(W->dtype_, Y->dtype_, p_in_tmp, p_weight_tmp,
                                p_bias_tmp, (void *)p_out_tmp, &conv_attrs);
          }
        }
      } else if (!input_condition && kernel_condition) {  // split input H/W
        if (in_h * in_w >= 64 * 1024) {
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
              for (j = 0; j < ou_c; j++) {
                int32_t i_offset = j * one_channel_ou_offset;
                int32_t o_offset = i * one_channel_ou_offset + j * ou_w * ou_h;
                memcpy(p_out + o_offset, p_tmp + i_offset, one_channel_ou_offset);
              }
            }
          }
        } else {
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
    } else if (attrs->group == conv_attrs.input_c &&
              attrs->group == conv_attrs.output_c) {  // depthwise
      kernel_condition = (luna_quant_ceil(in_c, 4) << 4) * k_h * k_w;
      kernel_condition = (kernel_condition <= 32 * 1024) ? 1 : 0;
      if (input_condition && kernel_condition) {  // no need split
        int32_t in_batch_size = in_c * in_h * in_w;
        int32_t ou_batch_size = ou_c * ou_h * ou_w * (Y->dtype_ & 0xF);
        for (n = 0; n < batch; n++) {
          int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
          int8_t *p_weight = (int8_t *)W->dptr_;
          int32_t *p_bias = (0 == paddr_b) ? NULL : ((int32_t *)paddr_b);
          int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
          ret = calc_depthwise_luna(W->dtype_, Y->dtype_, p_in, p_weight, p_bias,
                                    (void *)p_out, &conv_attrs);
        }
      } else if (!input_condition && kernel_condition) {  // split input H/W
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
    } else {  // group conv
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
      } else if (!input_condition && kernel_condition) {  // split input H/W
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
              for (i = 0; i < split_num; i++) {
                int32_t i_offset = i * ou_addr_offset + j * one_channel_ou_offset;
                int32_t o_offset = i * one_channel_ou_offset + j * ou_w * ou_h;
                memcpy(p_tmp + o_offset, p_out_group + i_offset,
                      one_channel_ou_offset);
              }
            }
            memcpy(p_out_group, p_tmp, ou_c * ou_h * ou_w);
          }
        }
      }
    }
  }
  else{// img2col
    int32_t c = X->shape_.dims_[1];
    int32_t h = X->shape_.dims_[2];
    int32_t stride = attrs->stride;
    int32_t pad_head = attrs->pad[0];
    int32_t pad_tail = attrs->pad[1];
    int32_t h_after_pad = h + pad_head + pad_tail;
    int32_t M = Y->shape_.dims_[2];
    int32_t N = c * kernel;
    int32_t L = Y->shape_.dims_[1];

    int32_t q_x = (int32_t)X->scale_;
    int32_t q_w = (int32_t)W->scale_;
    int32_t q_y = (int32_t)Y->scale_;
    int32_t shift = q_x + q_w - q_y;

    int8_t *data = (int8_t *)X->dptr_;
    int8_t *p_weight = (int8_t *)W->dptr_;
    uint32_t temp_size = 0;
    uint32_t workspace_size = Temp->shape_.dims_[0];



    const int32_t left_limit = 64 * 1024;
    const int32_t right_limit = 32 * 1024;
    // int8_t *p_in = data_temp2;
    int8_t *p_out = (int8_t *)Y->dptr_;
    int32_t int8_condition_l = (luna_quant_ceil(M, 2) << 2) * (luna_quant_ceil(N, 3) << 3);  // right:4x8
    int32_t int8_condition_r = (luna_quant_ceil(N, 3) << 3) * (luna_quant_ceil(L, 2) << 2);  // right:8x4
    if (int8_condition_l > left_limit) {  //split left martrix
      int s_num = 2;
      int32_t split_M = (0 == (M % s_num)) ? (M / s_num) : (M / s_num + 1);
      int32_t final_s_M = 0;
      int8_condition_l = (luna_quant_ceil(split_M, 2) << 2) * (luna_quant_ceil(N, 3) << 3);  // right:4x8
      while (int8_condition_l > left_limit) {
        s_num++;
        split_M = (0 == (M % s_num)) ? (M / s_num) : (M / s_num + 1);
        int8_condition_l = (luna_quant_ceil(split_M, 2) << 2) * (luna_quant_ceil(N, 3) << 3);  // right:4x8
      }
      // final_s_M = (0 == (M % s_num)) ? (split_M) : ( M - (split_M * (s_num - 1)));

      int32_t split_left_size = split_M * N;
      int32_t split_out_size = split_M * L;


      int8_t * data_temp2 = (int8_t *)Temp->dptr_ + temp_size;
      temp_size += split_left_size;

      int8_t *data_temp = (int8_t *)Temp->dptr_+ temp_size;
      if ((pad_head !=0) || (pad_tail !=0)){
        memset(data_temp, 0, c * h_after_pad);
        for(int32_t i = 0; i < c ; i++)
          memcpy(data_temp + pad_head + i * h_after_pad, data + i * h, h);
        temp_size += c * h_after_pad;
      }
      else{
        memcpy(data_temp, data, c * h);
        temp_size += c * h;
      }


      if (int8_condition_r <= right_limit) {
        CONV1D_MAT_MUL_LUNA_API luna_mat_mul_api;
        img2col(data_temp, data_temp2, c, h, kernel, stride);//c_in h_in w_in ==> hout * (kernel * c_in)
        
        for (int32_t i = 0; i < s_num; i++) {
          final_s_M = (s_num - 1 == i) ? (split_M) : ( M - (split_M * (s_num - 1)));

          int8_t *p_tmp_in = (int8_t *)data_temp2 + split_left_size * i;
          int8_t *p_tmp_ou = (int8_t *)p_out + split_out_size * i;
          int32_t in_oft = split_left_size * i;
          int32_t ou_oft = split_out_size * i;
          if (i == (s_num - 1)) {
            split_M = final_s_M;
            split_left_size = split_M * N;
            split_out_size = split_M * L;
          }
          if (has_bias) {
            int32_t *p_out_tmp = (int32_t *)((int8_t *)Temp->dptr_ + temp_size);
            temp_size += split_M * L * (Bias->dtype_ & 0xF);
            if (temp_size > workspace_size)
              return -1;
            luna_mat_mul_api = (CONV1D_MAT_MUL_LUNA_API)conv1d_luna_api_list[0][bias_idx].luna_api;
            CONV1D_VEC_ADD_LUNA_API luna_add_api =
                  (CONV1D_VEC_ADD_LUNA_API)conv1d_luna_api_list[2 + bias_idx][ou_idx].luna_api;
            // dump_int8("p_tmp_in.txt", p_tmp_in, split_M, N);
            // dump_int8("p_weight.txt", p_weight, N, L);
            ret = luna_mat_mul_api(p_tmp_in, p_weight, p_out_tmp, split_M, N, L, 0);
            for(int32_t j = 0; j < split_M; j++)
              ret |= luna_add_api(p_out_tmp + j * L, (void *)paddr_b, p_tmp_ou + j * L, L, shift);
          } 
          else {
            luna_mat_mul_api = (CONV1D_MAT_MUL_LUNA_API)conv1d_luna_api_list[0][ou_idx].luna_api;
            ret = luna_mat_mul_api(p_tmp_in, p_weight, p_tmp_ou, split_M, N, L, shift);
          }
        }
      }
      else {  // big martrix split on col
        printf("do not support!\n");
        return T_ERR_INVALID_PARA;
      }
    }
    else {  // left martrix <= 64KB

      int8_t * data_temp2 = (int8_t *)Temp->dptr_ + temp_size;
      temp_size += M * N;
      int8_t *data_temp = (int8_t *)Temp->dptr_+ temp_size;
      if ((pad_head !=0) || (pad_tail !=0)){
        memset(data_temp, 0, c * h_after_pad);
        for(int32_t i = 0; i < c ; i++)
          memcpy(data_temp + pad_head + i * h_after_pad, data + i * h, h);
        // temp_size += c * h_after_pad;
      }
      else{
        data_temp = data;
      }
      memset(data_temp2, 0, M * N);
      for(int32_t i = 0; i < M; i++){
        for(int32_t j = 0; j < c; j++){
            memcpy(data_temp2 + i * c * kernel + j * kernel, data_temp + i * stride + j * h_after_pad, kernel);
        }
      }

      int8_t *p_tmp_in = (int8_t *)data_temp2;
      int8_t *p_tmp_ou = (int8_t *)p_out;
      
      if (int8_condition_r <= right_limit) {
        CONV1D_MAT_MUL_LUNA_API luna_mat_mul_api;
        if (has_bias) {
          int32_t *p_tmp = (int32_t *)((int8_t *)Temp->dptr_ + temp_size);
          temp_size += M * L * 4; //sizeof(int32_t)
          if (temp_size > workspace_size)
            return -1;
          luna_mat_mul_api =
              (CONV1D_MAT_MUL_LUNA_API)conv1d_luna_api_list[0][bias_idx].luna_api;
          ret = luna_mat_mul_api(p_tmp_in, p_weight, p_tmp, M, N, L, 0);
          for (int32_t i = 0; i < M; i++)  // add bias
          {
            CONV1D_VEC_ADD_LUNA_API luna_add_api =
                (CONV1D_VEC_ADD_LUNA_API)conv1d_luna_api_list[2 + bias_idx][ou_idx].luna_api;
            int8_t *tsrc1 = (int8_t *)p_tmp + i * L * (Bias->dtype_ & 0xF);
            int8_t *tdst = (int8_t *)p_tmp_ou + i * L * (Y->dtype_ & 0xF);
            ret |= luna_add_api(tsrc1, (void *)paddr_b, tdst, L, shift);
          }
        } else {
          luna_mat_mul_api =
              (CONV1D_MAT_MUL_LUNA_API)conv1d_luna_api_list[0][ou_idx].luna_api;
          ret = luna_mat_mul_api(p_tmp_in, p_weight, p_tmp_ou, M, N, L, shift);
        }
      }
      else { // big martrix split on col
        int32_t split_num = 2;
        int32_t split_L = L / split_num;
        int8_condition_r =
            (luna_quant_ceil(N, 3) << 3) * (luna_quant_ceil(split_L, 2) << 2);  // right:8x4
        while (int8_condition_r > right_limit || (0 != (L % split_num))) {
          split_num++;
          split_L = L / split_num;
          int8_condition_r = (luna_quant_ceil(N, 3) << 3) *
                            (luna_quant_ceil(split_L, 2) << 2);  // right:8x4
        }
        {
          CONV1D_SPLIT_MAT_MUL_LUNA_API luna_mat_mul_api;
          if (has_bias) {
            int32_t *p_tmp = (int32_t *)((int8_t *)Temp->dptr_ + temp_size);
            temp_size += M * L * 4; //sizeof(int32_t)
            if (temp_size > workspace_size)
              return -1;
            luna_mat_mul_api = (CONV1D_SPLIT_MAT_MUL_LUNA_API)conv1d_luna_api_list[1][bias_idx].luna_api;
            ret = luna_mat_mul_api(p_tmp_in, p_weight, p_tmp, split_num, M, N, L, 0);
            for (int32_t i = 0; i < M; i++)  // add bias
            {
              CONV1D_VEC_ADD_LUNA_API luna_add_api =
                  (CONV1D_VEC_ADD_LUNA_API)conv1d_luna_api_list[2 + bias_idx][ou_idx].luna_api;
              int8_t *tsrc1 = (int8_t *)p_tmp + i * L * (Bias->dtype_ & 0xF);
              int8_t *tdst = (int8_t *)p_tmp_ou + i * L * (Y->dtype_ & 0xF);
              ret |= luna_add_api(tsrc1, (void *)paddr_b, tdst, L, shift);
            }
          } else {
            luna_mat_mul_api =
                (CONV1D_SPLIT_MAT_MUL_LUNA_API)conv1d_luna_api_list[1][ou_idx].luna_api;
            ret = luna_mat_mul_api(p_tmp_in, p_weight, p_tmp_ou, split_num, M, N, L, shift);
          }
        }
      }      
    }
    luna_mat_trans_q7(p_out, p_out, M, L);
  }
  return ret;
}
#endif  //_CONV1DINT_VENUS_H_
