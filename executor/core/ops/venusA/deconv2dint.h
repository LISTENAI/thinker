#ifndef _DECONV2DINT_VENUSA_H_
#define _DECONV2DINT_VENUSA_H_

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

/// Initialize deconvolution parameters
static void deconv2dint_luna_para_init(ConvTranspose2dIntAttrs *attrs, conv_struct_t *conv_attrs, 
                                      tTensor *X, tTensor *W, tTensor *Bias, tTensor *Y) {
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
    conv_attrs->padding_h_up = attrs->kernel[0] - attrs->pad[0] - 1;
    conv_attrs->padding_h_down = attrs->kernel[0] - attrs->pad[2] - 1 + attrs->output_padding[0];
    conv_attrs->padding_w_left = attrs->kernel[1] - attrs->pad[1] - 1;
    conv_attrs->padding_w_right = attrs->kernel[1] - attrs->pad[3] - 1 + attrs->output_padding[1];
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
    conv_attrs->negative_shift_value = q_x + q_w - q_y;
    
    // Memory and data type configuration
    uint8_t data_mem_type = ((X->mem_.type_ & 0x0F) + 1) % 3;
    uint8_t weight_mem_type = ((W->mem_.type_ & 0x0F) + 1) % 3;
    conv_attrs->data_mem_type = (data_mem_type << 4) | weight_mem_type;
    conv_attrs->ou_bits = Y->byte_ * 8;
    conv_attrs->weight_bits = (W->dtype_ == Int4) ? 4 : 8;
    conv_attrs->group = attrs->group;
}

/// Main deconvolution function
int32_t deconv2dint_luna(tTensor *X, tTensor *W, tTensor *Bias, tTensor *Y, 
                       tTensor *Temp, ConvTranspose2dIntAttrs *attrs) {
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
    deconv2dint_luna_para_init(attrs, &conv_attrs, X, W, Bias, Y);

    // Kernel/stride/pad checks from VenusA CNN constraints.
    #if THINKER_PARAM_CHECK
    if (attrs->kernel[0] < 1 || attrs->kernel[1] < 1 ||
                        attrs->kernel[0] > 12 || attrs->kernel[1] > 12) {
        return (T_ERR_INVALID_PARA);
    }

    if ((attrs->stride[0] != 1 && attrs->stride[0] != 2 && attrs->stride[0] != 4) ||
                        (attrs->stride[1] != 1 && attrs->stride[1] != 2 && attrs->stride[1] != 4)) {
        return (T_ERR_INVALID_PARA);
    }

    if (attrs->pad[0] >= attrs->kernel[0] || attrs->pad[2] >= attrs->kernel[0] ||
                        attrs->pad[1] >= attrs->kernel[1] || attrs->pad[3] >= attrs->kernel[1]) {
        return (T_ERR_INVALID_PARA);
    }

    if ((attrs->stride[0] == 2 && (attrs->kernel[0] < 2 || attrs->kernel[0] > 5)) ||
                        (attrs->stride[1] == 2 && (attrs->kernel[1] < 2 || attrs->kernel[1] > 5)) ||
                        (attrs->stride[0] == 4 && (attrs->kernel[0] < 4 || attrs->kernel[0] > 5)) ||
                        (attrs->stride[1] == 4 && (attrs->kernel[1] < 4 || attrs->kernel[1] > 5))) {
        return (T_ERR_INVALID_PARA);
    }

    if (attrs->dilation[0] != 1 || attrs->dilation[1] != 1) {
        return (T_ERR_INVALID_PARA);
    }

    if (attrs->group != 1) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    int32_t in_is_psram = (X->mem_.type_ != 2) ? 1 : 0;
    int32_t ou_is_psram = (Y->mem_.type_ != 2) ? 1 : 0;
    // Common deconvolution
    if(attrs->group == 1) {
        if (ou_is_psram) {
            #if THINKER_RUNTIME_CHECK
            if (p_tmp == NULL || workspace_size <= 0) {
                return (T_ERR_NO_WORKSPACE);
            }
#endif
            int32_t in_c = conv_attrs.input_c;
            int32_t in_h = conv_attrs.input_h;
            int32_t in_w = conv_attrs.input_w;
            int32_t ou_c = conv_attrs.output_c;
            int32_t ou_w = conv_attrs.output_w;
            int32_t ou_h = conv_attrs.output_h;
            int32_t k_h = conv_attrs.weight_h;
            int32_t k_w = conv_attrs.weight_w;
            int32_t s_h = conv_attrs.stride_h;
            int32_t s_w = conv_attrs.stride_w;
            int32_t pad_ht = conv_attrs.padding_h_up;
            int32_t pad_hb = conv_attrs.padding_h_down;
            int32_t ou_bits = conv_attrs.ou_bits;
            uint32_t log2n_s_w = s_w >> 1;
            uint32_t log2n_s_h = s_h >> 1;

            uint32_t split_in_num = 1;
            uint32_t tmp_in_h = in_h;
            uint32_t in_without_h = (luna_quant_ceil(in_c, 3) << 3) * (luna_quant_ceil(in_w, 3 + log2n_s_w) << (3 + log2n_s_w));

            int32_t overlap_num = luna_quant_ceil((k_h - 1), log2n_s_h);;

            // if (cnn_layer_type < LUNA_CONV1D)	//CNN1D no support split
            // {
            int32_t tmp_ou_h_1st = ou_h;
            int32_t tmp_ou_h_last = ou_h;
            int32_t ou_without_h = ou_c * ou_w;
            int32_t p_ht_1st = pad_ht;
            int32_t p_ht	 = pad_ht;
            if (0 == ((k_h - 1) & (s_h - 1)))
            {
              p_ht = s_h - 1;
            }
            else
            {
              p_ht = ((k_h - 1) & (s_h - 1)) - 1;
            }
            while ((tmp_in_h * in_without_h > CONV_IN_CONDITION) || (tmp_ou_h_1st * ou_without_h > CONV_IN_CONDITION) || (tmp_ou_h_last * ou_without_h > CONV_IN_CONDITION) || ((in_h % split_in_num) != 0))
            {
              split_in_num += 1;
              tmp_in_h = in_h / split_in_num;
              tmp_ou_h_1st = (tmp_in_h - 1) * s_h + 1 + p_ht_1st - k_h + 1;
              tmp_ou_h_last = (tmp_in_h + overlap_num - 1) * s_h + 1 + p_ht + pad_hb - k_h + 1;
              if ((split_in_num >= in_h) || (split_in_num >= ou_h))
              {
                break;
              }
            }
            /////////////////check split condition////////////////
            int32_t condition_1 = ((in_h % split_in_num) == 0);
            int32_t condition_2 = (((in_h / split_in_num - 1) * s_h + 1) >= k_h);
            if (!condition_1 || !condition_2 || (((1 == s_h) && (1 == s_w)) && split_in_num != 1) || ((tmp_in_h + overlap_num) * in_without_h > CONV_IN_CONDITION))
            {
              printf("[%s][%d]deconv input split invalid, split_num:%d \r\n", __func__, __LINE__, split_in_num);
              return -1;
            }

            int32_t s_h_d = s_h;
            int32_t s_w_d = s_w;
            s_h = 1;
            s_w = 1;
            int32_t pad_ht_1st = pad_ht;
            int32_t pad_ht_middle = 0;
            if (0 == ((k_h - 1) & (s_h_d - 1)))
            {
                pad_ht_middle = s_h_d - 1;
            }
            else
            {
                pad_ht_middle = ((k_h - 1) & (s_h_d - 1)) - 1;
            }
          int32_t split_in_h = in_h / split_in_num;
          int32_t skip_load_weight = 0;
          int32_t one_channel_ou_offset = 0;
          int32_t tmp_ou_h = ou_h;
          if (split_in_num != 1) {
            for (int32_t i = 0; i < split_in_num; i++) {
              if (i == 0) {
                tmp_ou_h = (split_in_h - 1) * s_h_d - k_h + pad_ht_1st + 2;
              }
              else if(i == split_in_num - 1) {
                tmp_ou_h = (split_in_h + overlap_num - 1) * s_h_d - k_h + pad_ht_middle + pad_hb + 2;
              }
              else {
                tmp_ou_h = (split_in_h + overlap_num - 1) * s_h_d - k_h + pad_ht_middle + 2;
              }
              int32_t tmp_need = ou_c * tmp_ou_h * ou_w * Y->byte_;
              #if THINKER_RUNTIME_CHECK
              if (tmp_ou_h <= 0 || workspace_size < tmp_need) {
                  return (T_ERR_NO_WORKSPACE);
              }
#endif
              conv_attrs.reserved = skip_load_weight | ((i + 1) << 16);
              skip_load_weight = 1 << 8;
              THINKER_RET_CHECK(luna_split_conv_para_pack(&conv_attrs, &conv_static_para, LUNA_DECONV), "luna_split_conv_para_pack");
              if (Int4 == W->dtype_)
                THINKER_RET_CHECK(API_LIB(deconv2d_i8i4o8)(p_src, p_weight, p_bias, (int8_t *)p_tmp, &conv_static_para), "luna_deconv2d_i8i4o8");
              else if (Int8 == W->dtype_)
                THINKER_RET_CHECK(API_LIB(deconv2d_i8i8o8)(p_src, p_weight, p_bias, (int8_t *)p_tmp, &conv_static_para), "luna_deconv2d_i8i8o8");
            //   tmp_ou_h = (i == split_in_num - 1) ? (ou_h - tmp_ou_h * (split_in_num - 1)) : split_in_h;
              int32_t size = ou_w * tmp_ou_h * Y->byte_;
              for (int32_t j = 0; j < ou_c; j++) {
                opi_psram_cpy_out(p_dst + one_channel_ou_offset + j * ou_w * ou_h * Y->byte_, p_tmp + j * size, size);
              }
              one_channel_ou_offset += size;
            }
          }
          else {
            #if THINKER_RUNTIME_CHECK
            if (workspace_size < ou_c * ou_h * ou_w * Y->byte_) {
                return (T_ERR_NO_WORKSPACE);
            }
#endif
            THINKER_RET_CHECK(luna_split_conv_para_pack(&conv_attrs, &conv_static_para, LUNA_DECONV), "luna_split_conv_para_pack");
            if (Int4 == W->dtype_)
              THINKER_RET_CHECK(API_LIB(deconv2d_i8i4o8)(p_src, p_weight, p_bias, (int8_t *)p_tmp, &conv_static_para), "luna_deconv2d_i8i4o8");
            else if (Int8 == W->dtype_)
              THINKER_RET_CHECK(API_LIB(deconv2d_i8i8o8)(p_src, p_weight, p_bias, (int8_t *)p_tmp, &conv_static_para), "luna_deconv2d_i8i8o8");
            opi_psram_cpy_out(p_dst, p_tmp, ou_c * ou_h * ou_w * Y->byte_);
          }
#if !(defined(WIN32) || defined(linux))
          HAL_FlushInvalidateDCache_by_Addr((uint32_t *)(Y->dptr_), ou_c * ou_h * ou_w * Y->byte_);
#endif
        }
        else {
          conv_attrs.reserved = 0;
          THINKER_RET_CHECK(luna_split_conv_para_pack(&conv_attrs, &conv_static_para, LUNA_DECONV), "luna_split_conv_para_pack");
          if (Int4 == W->dtype_)
            THINKER_RET_CHECK(API_LIB(deconv2d_i8i4o8)(p_src, p_weight, p_bias, p_dst, &conv_static_para), "luna_deconv2d_i8i4o8");
          else if (Int8 == W->dtype_)  
            THINKER_RET_CHECK(API_LIB(deconv2d_i8i8o8)(p_src, p_weight, p_bias, p_dst, &conv_static_para), "luna_deconv2d_i8i8o8");
        }
    }

    return T_SUCCESS;
}

#endif  //_DECONV2DINT_VENUSA_H_
