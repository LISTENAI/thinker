#ifndef _CONV2DINT_VENUS_H_
#define _CONV2DINT_VENUS_H_

#include <math.h>
#include <stdint.h>
#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#include "luna/luna_cnn_tools.h"
#define API_LIB(api) luna_##api
#endif
#include "thinker_status.h"


// Quantization ceiling function
static int32_t luna_quant_ceil(int32_t x, int32_t shift) {
  if (x & ~(0xFFFFFFFF << shift)) {
    return (x >> shift) + 1;
  } else {
    return (x >> shift);
  }
}

// Initialize convolution parameters
static void conv2dint_luna_para_init(Conv2dIntAttrs *attrs, conv_struct_t *conv_attrs, tTensor *X, tTensor *W, tTensor *Bias, tTensor *Y) {
  memset(conv_attrs, 0, sizeof(conv_struct_t));
  conv_attrs->is_bias = (Bias != NULL);

  // Input dimensions
  conv_attrs->input_c = X->shape_.dims_[1];
  conv_attrs->input_h = X->shape_.dims_[2];
  conv_attrs->input_w = X->shape_.dims_[3];

  // Output dimensions
  conv_attrs->output_c = Y->shape_.dims_[1];
  conv_attrs->output_h = Y->shape_.dims_[2];
  conv_attrs->output_w = Y->shape_.dims_[3];

  // Kernel parameters
  conv_attrs->weight_h = attrs->kernel[0];
  conv_attrs->weight_w = attrs->kernel[1];
  conv_attrs->stride_h = attrs->stride[0];
  conv_attrs->stride_w = attrs->stride[1];
  conv_attrs->padding_h_up = attrs->pad[0];
  conv_attrs->padding_h_down = attrs->pad[2];
  conv_attrs->padding_w_left = attrs->pad[1];
  conv_attrs->padding_w_right = attrs->pad[3];
  conv_attrs->dilation_h = attrs->dilation[0];
  conv_attrs->dilation_w = attrs->dilation[1];


  // Activation type
  switch(attrs->act_type) {
      case 1: conv_attrs->activation_type = RELU; break;
      case 2: conv_attrs->activation_type = PRELU; break;
      default: conv_attrs->activation_type = NO_ACTIVE; break;
  }

  // Quantization parameters
  int32_t q_x = (int32_t)X->scale_;
  int32_t q_w = (int32_t)W->scale_;
  int32_t q_y = (int32_t)Y->scale_;
  conv_attrs->positive_shift_type = attrs->quant_type == 0 ? ShiftType_Floor : ShiftType_FloorX05;
  conv_attrs->positive_shift_value = q_x + q_w - q_y;
  conv_attrs->negative_shift_type = attrs->quant_type == 0 ? ShiftType_Floor : ShiftType_FloorX05;
  conv_attrs->negative_shift_value = conv_attrs->positive_shift_value;

  // Memory and data type configuration
  uint8_t data_mem_type = (X->mem_.type_ &  0x0F) + 1;
  data_mem_type = (data_mem_type == 3) ? 0 : data_mem_type;
  uint8_t weight_mem_type = (W->mem_.type_ & 0x0F) + 1;
  weight_mem_type = (weight_mem_type == 3) ? 0 : weight_mem_type;
  conv_attrs->data_mem_type = (data_mem_type << 4) | weight_mem_type;
  conv_attrs->ou_bits = Y->byte_ * 8;
  conv_attrs->weight_bits = W->dtype_ == Int4 ? 4 : 8;
  conv_attrs->out_padding_h = 0;
  conv_attrs->out_padding_w = 0;
  conv_attrs->group = attrs->group;
}

// Main convolution function
int32_t conv2dint_luna(tTensor *X, tTensor *W, tTensor *Bias, tTensor *Y, tTensor *Temp, Conv2dIntAttrs *attrs) {
  int8_t *p_src   = (int8_t *)(X->dptr_);
  int8_t *p_weight= (int8_t *)(W->dptr_);
  int32_t *p_bias = Bias ? (int32_t *)(Bias->dptr_) : NULL;
  int8_t *p_dst   = (int8_t *)(Y->dptr_);
  int8_t *p_tmp   = Temp ? (int8_t *)Temp->dptr_ : NULL;
  int32_t workspace_size = Temp ? Temp->shape_.dims_[0] : 0;

  #if THINKER_PARAM_CHECK
  if (X->dtype_ != Int8) {
      return (T_ERR_INVALID_DATATYPE);
  }

  if (W->dtype_ != Int4 && W->dtype_ != Int8) {
      return (T_ERR_INVALID_DATATYPE);
  }

  if (Y->dtype_ != Int8) {
      return (T_ERR_INVALID_DATATYPE);
  }

  if (Bias != NULL && getTensorSize(Bias) != (size_t)Y->shape_.dims_[1]) {
      return (T_ERR_INVALID_PARA);
  }
#endif

  conv_struct_t conv_attrs;
  luna_cnn_static_para_t conv_static_para;
  conv2dint_luna_para_init(attrs, &conv_attrs, X, W, Bias, Y);

  uint32_t in_c      = conv_attrs.input_c;
  uint32_t in_h      = conv_attrs.input_h;
  uint32_t in_w      = conv_attrs.input_w;
  uint32_t ou_c      = conv_attrs.output_c;
  uint32_t ou_h      = conv_attrs.output_h;
  uint32_t ou_w      = conv_attrs.output_w;
  uint32_t ou_size   = ou_c * ou_h * ou_w;
  uint32_t group     = conv_attrs.group;
  uint32_t k_h       = conv_attrs.weight_h;
  uint32_t k_w       = conv_attrs.weight_w;
  uint32_t s_h       = conv_attrs.stride_h;
  uint32_t s_w       = conv_attrs.stride_w;
  uint32_t padding_hd = conv_attrs.padding_h_down;
  uint32_t padding_hu = conv_attrs.padding_h_up;
  uint32_t padding_wl = conv_attrs.padding_w_left;
  uint32_t padding_wr = conv_attrs.padding_w_right;
  uint32_t dilation_h = attrs->dilation[0];
  uint32_t dilation_w = attrs->dilation[1];
  uint32_t log2n_stride_w = (conv_attrs.stride_w >> 1);

  int32_t in_is_psram = (X->mem_.type_ != 2) ? 1 : 0;
  int32_t ou_is_psram = (Y->mem_.type_ != 2) ? 1 : 0;
  int32_t shift = (int32_t)X->scale_ + (int32_t)W->scale_ - (int32_t)Y->scale_;
  #if THINKER_PARAM_CHECK
  if (shift < 0 || shift > 63) {
      return (T_ERR_INVALID_PARA);
  }
#endif
  #if THINKER_RUNTIME_CHECK
  if (ou_is_psram && (p_tmp == NULL || workspace_size <= 0)) {
      return (T_ERR_NO_WORKSPACE);
  }

  if (k_h < 1 || k_w < 1 || k_h > 12 || k_w > 12) {
      return (T_ERR_INVALID_PARA);
  }

  if ((s_h != 1 && s_h != 2 && s_h != 4) ||
                      (s_w != 1 && s_w != 2 && s_w != 4)) {
      return (T_ERR_INVALID_PARA);
  }

  if (padding_hu >= k_h || padding_hd >= k_h ||
                      padding_wl >= k_w || padding_wr >= k_w) {
      return (T_ERR_INVALID_PARA);
  }

  if ((dilation_h != 1 && dilation_h != 2 && dilation_h != 4 && dilation_h != 8) ||
                      (dilation_w != 1 && dilation_w != 2 && dilation_w != 4 && dilation_w != 8)) {
      return (T_ERR_INVALID_PARA);
  }

  if ((dilation_h != 1 || dilation_w != 1) && (k_h > 5 || k_w > 5)) {
      return (T_ERR_INVALID_PARA);
  }

  if (in_h + padding_hu + padding_hd < (k_h - 1) * dilation_h + 1 ||
                      in_w + padding_wl + padding_wr < (k_w - 1) * dilation_w + 1) {
      return (T_ERR_INVALID_PARA);
  }

  if (attrs->group != 1 &&
      !(attrs->group == conv_attrs.input_c && attrs->group == conv_attrs.output_c)) {
      return (T_ERR_INVALID_PARA);
  }
#endif

  if (1 == attrs->group) { // common conv2d
        int32_t weight_size_align = (luna_quant_ceil(ou_c, 2) << 2) * (luna_quant_ceil(in_c, 3) << 3) * k_h * k_w; // <=32KB
        bool is_split_weight = (weight_size_align  > 32768);
        if (ou_is_psram) {
          int32_t data_size_align_withouth =(luna_quant_ceil(in_c, 3) << 3) * 
          (luna_quant_ceil(in_w, (3 + log2n_stride_w)) << (3 + log2n_stride_w));
          int32_t target_elements = CONV_IN_CONDITION / data_size_align_withouth;
          int32_t split_max_ou_h = (target_elements - ((k_h - 1) * dilation_h + 1) + s_h) / s_h;
          split_max_ou_h = (split_max_ou_h > 1) ? split_max_ou_h : 1;
          int32_t split_in_num = (0 == (ou_h % split_max_ou_h)) ? (ou_h / split_max_ou_h) : (ou_h / split_max_ou_h + 1);

          int32_t skip_load_weight = 0;
          int32_t tmp_ou_h = split_max_ou_h;
          if (split_in_num != 1) {
            for (int32_t i = 0; i < split_in_num; i++) {
              int32_t cur_ou_h = (i == split_in_num - 1) ? (ou_h - split_max_ou_h * (split_in_num - 1)) : split_max_ou_h;
              int32_t tmp_need = ou_c * cur_ou_h * ou_w * Y->byte_;
              #if THINKER_RUNTIME_CHECK
              if (cur_ou_h <= 0 || workspace_size < tmp_need) {
                  return (T_ERR_NO_WORKSPACE);
              }
#endif
              conv_attrs.reserved = skip_load_weight | ((i + 1) << 16);
              skip_load_weight = is_split_weight ? 0 : 1 << 8;
              THINKER_RET_CHECK(luna_split_conv_para_pack(&conv_attrs, &conv_static_para, LUNA_CONV), "luna_split_conv_para_pack");
              if (Int4 == W->dtype_)
                THINKER_RET_CHECK(API_LIB(conv2d_i8i4o8)(p_src, p_weight, p_bias, (int8_t *)p_tmp, &conv_static_para), "luna_conv2d_i8i4o8");
              else if (Int8 == W->dtype_)
                THINKER_RET_CHECK(API_LIB(conv2d_i8i8o8)(p_src, p_weight, p_bias, (int8_t *)p_tmp, &conv_static_para), "luna_conv2d_i8i8o8");

              tmp_ou_h = cur_ou_h;
              int32_t one_channel_ou_offset = ou_w * split_max_ou_h * Y->byte_;
              int32_t size = ou_w * tmp_ou_h * Y->byte_;
              for (int32_t j = 0; j < ou_c; j++) {
                int8_t *out_once = p_dst + i * one_channel_ou_offset + j * ou_w * ou_h * Y->byte_;
                opi_psram_cpy_out(out_once, p_tmp + j * size, size);
              }
            }
          }
          else {
            int32_t tmp_need = ou_c * ou_h * ou_w * Y->byte_;
            #if THINKER_RUNTIME_CHECK
            if (workspace_size < tmp_need) {
                return (T_ERR_NO_WORKSPACE);
            }
#endif
            THINKER_RET_CHECK(luna_split_conv_para_pack(&conv_attrs, &conv_static_para, LUNA_CONV), "luna_split_conv_para_pack");
            if (Int4 == W->dtype_)
              THINKER_RET_CHECK(API_LIB(conv2d_i8i4o8)(p_src, p_weight, p_bias, (int8_t *)p_tmp, &conv_static_para), "luna_conv2d_i8i4o8");
            else if (Int8 == W->dtype_)
              THINKER_RET_CHECK(API_LIB(conv2d_i8i8o8)(p_src, p_weight, p_bias, (int8_t *)p_tmp, &conv_static_para), "luna_conv2d_i8i8o8");
            opi_psram_cpy_out(p_dst, p_tmp, ou_c * ou_h * ou_w * Y->byte_);
          }
#if !(defined(WIN32) || defined(linux))
          HAL_FlushInvalidateDCache_by_Addr((uint32_t *)(Y->dptr_), ou_c * ou_h * ou_w * Y->byte_);
#endif
        }
        else {
          conv_attrs.reserved = 0;
          THINKER_RET_CHECK(luna_split_conv_para_pack(&conv_attrs, &conv_static_para, LUNA_CONV), "luna_split_conv_para_pack");
          if (Int4 == W->dtype_)
            THINKER_RET_CHECK(API_LIB(conv2d_i8i4o8)(p_src, p_weight, p_bias, p_dst, &conv_static_para), "luna_conv2d_i8i4o8");
          else if (Int8 == W->dtype_)
            THINKER_RET_CHECK(API_LIB(conv2d_i8i8o8)(p_src, p_weight, p_bias, p_dst, &conv_static_para), "luna_conv2d_i8i8o8");
        }
    }
    else  // depthwise
    {
          // int32_t weight_size_align = (luna_quant_ceil(ou_c, 2) << 2) * k_h * k_w; <=32KB
        if (ou_is_psram) {
          int32_t h_eff = in_h + (padding_hu != 0 ? 1 : 0);
          int32_t data_size_align_withouth =(luna_quant_ceil(in_c, 2) << 2) *
            (luna_quant_ceil(in_w, (3 + log2n_stride_w)) << (3 + log2n_stride_w));
          int32_t dw_data_size_align = data_size_align_withouth * h_eff;
          #if THINKER_RUNTIME_CHECK
          if (data_size_align_withouth <= 0 || data_size_align_withouth * k_h > DW_IN_CONDITION) {
              return (T_ERR_INVALID_PARA);
          }
#endif
          int32_t need_split_input = (dw_data_size_align > DW_IN_CONDITION);
          #if THINKER_RUNTIME_CHECK
          if (p_tmp == NULL || workspace_size <= 0) {
              return (T_ERR_NO_WORKSPACE);
          }
#endif
          int32_t target_elements = DW_IN_CONDITION / data_size_align_withouth;
          int32_t split_max_ou_h = (target_elements - k_h - conv_attrs.padding_h_up + s_h) / s_h;
          split_max_ou_h = (split_max_ou_h > 1) ? split_max_ou_h : 1;
          int32_t split_in_num = (0 == (ou_h % split_max_ou_h)) ? (ou_h / split_max_ou_h) : (ou_h / split_max_ou_h + 1);
          int32_t skip_load_weight = 0;
          int32_t tmp_ou_h = split_max_ou_h;
          for (int32_t i = 0; i < split_in_num; i++) {
            int32_t cur_ou_h = (i == split_in_num - 1) ? (ou_h - split_max_ou_h * (split_in_num - 1)) : split_max_ou_h;
            int32_t tmp_need = ou_c * cur_ou_h * ou_w * Y->byte_;
            #if THINKER_RUNTIME_CHECK
            if (cur_ou_h <= 0 || workspace_size < tmp_need) {
                return (T_ERR_NO_WORKSPACE);
            }
#endif
            conv_attrs.reserved = skip_load_weight | ((i + 1) << 16);
            skip_load_weight = 1 << 8;
            THINKER_RET_CHECK(luna_split_conv_para_pack(&conv_attrs, &conv_static_para, LUNA_DEPTHWISE), "luna_split_conv_para_pack");

            if (Int4 == W->dtype_)
              THINKER_RET_CHECK(API_LIB(depthwise2d_i8i4o8)(p_src, p_weight, p_bias, (int8_t *)p_tmp, &conv_static_para), "luna_depthwise2d_i8i4o8");
            else if (Int8 == W->dtype_)
              THINKER_RET_CHECK(API_LIB(depthwise2d_i8i8o8)(p_src, p_weight, p_bias, (int8_t *)p_tmp, &conv_static_para), "luna_depthwise2d_i8i8o8");

            tmp_ou_h = cur_ou_h;
            int32_t one_channel_ou_offset = ou_w * split_max_ou_h * Y->byte_;
            int32_t size = ou_w * tmp_ou_h * Y->byte_;
            for (int32_t j = 0; j < ou_c; j++) {
              int8_t *out_once = p_dst + i * one_channel_ou_offset + j * ou_w * ou_h * Y->byte_;
              opi_psram_cpy_out(out_once, p_tmp + j * size, size);
            }
          }
  #if !(defined(WIN32) || defined(linux))
            HAL_FlushInvalidateDCache_by_Addr((uint32_t *)(Y->dptr_), ou_c * ou_h * ou_w * Y->byte_);
  #endif
          }
          else {
            conv_attrs.reserved = 0;
            THINKER_RET_CHECK(luna_split_conv_para_pack(&conv_attrs, &conv_static_para, LUNA_DEPTHWISE), "luna_split_conv_para_pack");
            if (Int4 == W->dtype_)
              THINKER_RET_CHECK(API_LIB(depthwise2d_i8i4o8)(p_src, p_weight, p_bias, p_dst, &conv_static_para), "luna_depthwise2d_i8i4o8");
            else if (Int8 == W->dtype_)
              THINKER_RET_CHECK(API_LIB(depthwise2d_i8i8o8)(p_src, p_weight, p_bias, p_dst, &conv_static_para), "luna_depthwise2d_i8i8o8");
          }
      }

  return T_SUCCESS;
}
#endif  //_CONV2DINT_VENUS_H_
