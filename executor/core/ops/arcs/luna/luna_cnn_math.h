/***************************************************************************
 * .h                                                  *
 *                                                                         *
 * Copyright (C) 2020 listenai Co.Ltd                   			       *
 * All rights reserved.                                                    *
 ***************************************************************************/
#ifndef __LUNA_CNN_MATH_H__
#define __LUNA_CNN_MATH_H__

#include "luna_math_types.h"

#ifndef MASK_SIM_TYPEDEF
#define MASK_SIM_TYPEDEF

typedef enum activation_type
{
	RELU	 = 0,
	PRELU    = 1,
	RELUx	 = 2,

	NO_ACTIVE = 128,
}e_activation_type;

typedef enum shift_type
{
	ShiftType_FloorX05 = 0,	//(floor(x+0.5))
	ShiftType_Floor,
}e_shift_type;

typedef struct conv_struct_
{
	uint32_t input_c;
	uint32_t input_w;
	uint32_t input_h;
	uint32_t output_c;
	uint32_t output_w;
	uint32_t output_h;
	uint8_t padding_w_left;
	uint8_t padding_w_right;
	uint8_t padding_h_up;
	uint8_t padding_h_down;
	uint8_t weight_w;
	uint8_t weight_h;
	uint8_t stride_w;
	uint8_t stride_h;
	uint8_t activation_type;
	uint8_t positive_shift_value;
	uint8_t positive_shift_type;
	uint8_t negative_shift_value;
	uint8_t negative_shift_type;
	uint8_t is_bias;
	uint8_t is_per_chn;
	uint8_t data_mem_type;	// weight(low 4bit) 0:sram, 1:flash, 2:psram, input(high 4bit) 0:sram, 1:flash, 2:psram
	uint8_t ou_bits;
	uint8_t weight_bits;	//support 4 and 8bit
	uint8_t dilation_w;
	uint8_t dilation_h;
	uint8_t out_padding_w;
	uint8_t out_padding_h;
	uint8_t batch_num;
	uint8_t group;
	uint8_t reserved;
}conv_struct_t;

typedef struct luna_cnn_static_para
{
	// load weight
	// offset 0 bytes
	uint16_t r8_c0_W;
	uint16_t r9_c0_W;
	uint16_t r8_c0_W_last;
	uint16_t r9_c0_W_last;
	uint32_t r18_W;
	uint32_t r18_W_last;

	// load feature map
	// offset 4 * 4 bytes
	uint32_t r8_c0_c2_1st_last;
	uint32_t r9_c0_c2_1st_last;
	uint32_t r8_c0_c2_1st;
	uint32_t r9_c0_c2_1st;
	uint32_t r8_c0_c2_last;
	uint32_t r9_c0_c2_last;
	uint32_t r8_c0_c2_mid;
	uint32_t r9_c0_c2_mid;
	uint16_t r8_c4;
	uint16_t r9_c4;

	// offset 13 * 4 bytes
	uint16_t r10_saddr_h;
	uint16_t r11_saddr_l;
	uint16_t r12_saddr_h;
	uint16_t r13_saddr_l;
	uint32_t r18_w_l;
	uint32_t r19_h_l_1st_last;
	uint32_t r19_h_l_1st;
	uint32_t r19_h_l_last;
	uint32_t r19_h_l_mid;
	uint16_t r20_c_l;
	uint16_t r21_sw_h;

	// calculation
	// offset 21 * 4 bytes
	uint32_t r8_st_rw_0;
	uint32_t r9_st_rw_0;
	uint32_t r8_st_rw_1;
	uint32_t r9_st_rw_1;
	uint32_t r8_st_rw_2;
	uint32_t r9_st_rw_2;
	uint32_t r8_st_rw_3;
	uint32_t r9_st_rw_3;
	uint32_t r8_st_rw_4;	// fot deconv_split
	uint32_t r9_st_rw_4;
	uint32_t r8_st_rw_5;
	uint32_t r9_st_rw_5;

	// offset 33 * 4 bytes
	// infmap P0 address
	uint32_t r10_c_kh_0;
	uint32_t r11_kw_w_0;
	uint32_t r12_c_kh_0;
	uint32_t r13_kw_w_0;
	uint32_t r14_p0_addr;
	uint32_t r10_h_g_0;
	uint32_t r11_gs_0;
	uint32_t r12_h_g_0;
	uint32_t r13_gs_0;

	// offset 42 * 4 bytes
	// infmap P1 address
	uint32_t r10_c_kh_1;
	uint32_t r11_kw_w_1;
	uint32_t r12_c_kh_1;
	uint32_t r13_kw_w_1;
	uint32_t r14_p1_addr;
	uint32_t r10_h_g_1;
	uint32_t r11_gs_1;
	uint32_t r12_h_g_1;
	uint32_t r13_gs_1;

	// offset 51 * 4 bytes
	uint32_t r17;
	uint32_t r18_1st_last;
	uint32_t r18_1st;	//
	uint32_t r18_last;
	uint32_t r18_mid;
	uint32_t r16_h;
	uint32_t r4_judge_sw;
	
	// offset 58 * 4 bytes
	uint32_t k_n_loop_num;
	uint32_t in_loop_num;
	uint32_t i_addr_inc_1st;
	uint32_t i_addr_inc;
	uint32_t o_addr_inc_1st;
	uint32_t o_addr_inc;
	uint32_t k_o_addr_inc;
	uint32_t k_addr_inc;
	uint32_t b_addr_inc;

	// offset 67 * 4 bytes
	uint32_t r23_shift;
	uint32_t r22_ow;
	uint32_t r27_integr;
	uint32_t r18_ch_len;
	uint32_t r18_ch_len_1st;
	uint32_t r18_ch_len_last;
	uint32_t r18_ch_len_mid;
	uint32_t r19_ch_num;	//kernel_n
	uint32_t r19_ch_num_last;	//kernel_n_last
	uint32_t r21_interval;

	// // offset 77 * 4 bytes
	uint32_t infmap_intv;

	// offset 78 * 4 bytes
	uint32_t is_bias;
	uint32_t is_per_chn;
	uint32_t data_mem_type;
	uint32_t weight_bits;
	uint32_t ou_bits;

	uint32_t r14_p0_addr_1st;	//for cnn_split

	// offset 84 * 4 bytes
	uint16_t in_c;
	uint16_t in_h;
	uint16_t in_w;
	uint16_t ou_c;
	uint16_t ou_h;
	uint16_t ou_w;
	uint8_t k_h;
	uint8_t k_w;
	uint8_t s_h;
	uint8_t s_w;
	uint8_t pad_wl;
	uint8_t pad_wr;
	uint8_t pad_ht;
	uint8_t pad_hb;
	uint8_t dilation_w;
	uint8_t dilation_h;
	uint8_t out_padding_w;
	uint8_t out_padding_h;
}luna_cnn_static_para_t;

typedef struct luna_cnn_para
{
	luna_cnn_static_para_t *cnn_static_para;

	uint32_t input_addr_oft;
	uint32_t weight_addr_oft;
	uint32_t bias_addr_oft;
	uint32_t output_addr_oft;
}luna_cnn_para_t;
#endif //MASK_SIM_TYPEDEF

int32_t luna_conv2d_i8i4o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_conv2d_i8i4o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_conv2d_i8i8o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_conv2d_i8i8o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);

int32_t luna_depthwise2d_i8i4o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_depthwise2d_i8i4o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_depthwise2d_i8i8o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_depthwise2d_i8i8o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);

int32_t luna_deconv2d_i8i4o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_deconv2d_i8i4o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_deconv2d_i8i8o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_deconv2d_i8i8o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);

int32_t luna_max_pooling2d_i8o8(const int8_t *p_in, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_mean_pooling2d_i8o8(const int8_t *p_in, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_mean_pooling2d_i8o32(const int8_t *p_in, int32_t *p_out, luna_cnn_static_para_t *conv_para_);

int32_t luna_conv1d_i8i4o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_conv1d_i8i4o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_conv1d_i8i8o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_conv1d_i8i8o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);

int32_t luna_depthwise1d_i8i4o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_depthwise1d_i8i4o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_depthwise1d_i8i8o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_depthwise1d_i8i8o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);

int32_t luna_deconv1d_i8i4o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_deconv1d_i8i4o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_deconv1d_i8i8o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_deconv1d_i8i8o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);

int32_t luna_max_pooling1d_i8o8(const int8_t *p_in, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_mean_pooling1d_i8o8(const int8_t *p_in, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_mean_pooling1d_i8o32(const int8_t *p_in, int32_t *p_out, luna_cnn_static_para_t *conv_para_);

#endif
