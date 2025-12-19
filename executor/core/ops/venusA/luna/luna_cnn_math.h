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
	uint8_t relu_x;

	uint32_t reserved; // 0-7:   skip_load_input(外部拆kernel)
					   // 8-15:  skip_load_weight(外部拆input)
					   // 16-23: ih_idx(内部拆输入)
					   // 24-31: ic_idx(内部拆kernel)
}conv_struct_t;

typedef struct luna_cnn_static_para
{
	// load weight
	// offset 0 bytes
	uint32_t lw_r12;	// ceil(infmap_c/8)*kernel_w*kernel_h*ceil(kernel_n/4))
	uint32_t lw_r12_last;
	uint32_t lw_r18;	// ceil(infmap_c/8)*kernel_w*kernel_h*ceil(kernel_n/4)*32)
	uint32_t lw_r18_last;

	// load feature map
	// offset 4 * 4 bytes
	uint32_t lf_r9_in_w;
	uint32_t lf_r10;	// infmap_w*infmap_h_all
	uint32_t lf_r11;	// for dw, in_w * in_h
	uint32_t lf_r12;	// ceil(infmap_w/32))
	uint32_t lf_r13;	// (in_c << 16) | in_h
	uint32_t lf_r13_1st;
	uint32_t lf_r13_last;
	uint32_t lf_r14;	//for dw, in_c
	uint32_t lf_r18;	// (in_h << 16) | in_w
	uint32_t lf_r18_1st;
	uint32_t lf_r18_last;
	uint32_t lf_r19;	// in_c
	uint32_t lf_stride;	// floor(stride_w/2)

	// calculation
	// offset 17 * 4 bytes
	uint32_t r8_st_0;	// (xc_map_cnt1 << 16) | c_map_cnt0
	uint32_t r9_rw_0; 	// (xkh_map_cnt5 << 16) | kh_map_cnt4
	uint32_t r8_st_1;	// (xkw_map_cnt3 << 16) | kw_map_cnt2
	uint32_t r9_rw_1;	// (xw_map_cnt7 << 16) | w_map_cnt6
	uint32_t r8_st_2;	// (xh_map_cnt9 << 16) | h_map_cnt8
	uint32_t r9_rw_2;	// (xg_map_cnt11 << 16) | g_map_cnt10
	uint32_t r8_st_3;	// for dw
	uint32_t r9_rw_3;
	uint32_t r8_st_4;	// reserved for deconv
	uint32_t r9_rw_4;

	// offset 27 * 4 bytes
	// infmap P1 address: read for featuremap
	uint32_t p1_r11_r10;	// (kh_stride_inf << 16) | c_stride_inf
	uint32_t p1_r13_r12;	// (w_stride_inf << 16) | kw_stride_inf
	uint32_t p1_r20;
	uint32_t p1_r20_1st;
	uint32_t p1_r21;
	uint32_t p1_r21_1st;
	uint32_t p1_r11_r10_1;	// (g_stride_inf << 16) | h_stride_inf
	uint32_t p1_r13_r12_1;	// for dw, gs_stride_inf

	// offset 35 * 4 bytes
	// infmap P0 address: read for kernel
	uint32_t p0_r11_r10;	// (kh_stride_ker << 16) | c_stride_ker
	uint32_t p0_r13_r12;	// (w_stride_ker << 16) | kw_stride_ker
	uint32_t p0_r20;
	uint32_t p0_r11_r10_1;	// (g_stride_ker << 16) | h_stride_ker
	uint32_t p0_r13_r12_1;	// for dw, gs_stride_ker
	uint32_t p0_r17;
	uint32_t p0_r18;
	uint32_t p0_r18_1st;
	uint32_t p0_r18_last;

	uint32_t p0_r16;	//	log2(stride_w)+2)*2^8+2+32

	// offset 45 * 4 bytes
	// pe config
	uint32_t pe_r8;		// relux_param
	uint32_t pe_r21;	// (ow_num << 16) | (ow_last_msk*2^8+ow_1st_msk)
	uint32_t pe_r23;	// (cnn_shift_total << 16) | (shift_per_ch_en*3)
	uint32_t pe_r26;	// (log2(o_precision)-3) << 16) | relux
	uint32_t pe_r27;	// (ceil(infmap_c/8)*kernel_h-1)

	// offset 50 * 4 bytes
	// iow config
	uint32_t iow_r18;	// oufmap_w*oufmap_h*o_precision/8
	uint32_t iow_r18_1st;
	uint32_t iow_r18_last;
	uint32_t iow_r19;	// kernel_n
	uint32_t iow_r19_last;
	uint32_t iow_r21;	// oufmap_w*oufmap_h*o_precision/8

	// offset 56 * 4 bytes
	uint32_t k_n_loop_num;
	uint32_t in_loop_num;
	uint32_t i_addr_inc_1st;
	uint32_t i_addr_inc;
	uint32_t o_addr_inc_1st;
	uint32_t o_addr_inc;
	uint32_t k_o_addr_inc;
	uint32_t k_addr_inc;
	uint32_t b_addr_inc;

	// offset 65 * 4 bytes
	uint32_t is_pad_ht_0;	// is_pad_ht_0
	uint32_t is_bias;
	uint32_t is_per_chn;
	uint32_t data_mem_type;
	uint32_t weight_bits;
	uint32_t ou_bits;
	uint32_t pool_type;	// 0:max_pool, 1:mean_pool

	uint32_t reserved;

	// offset 72 * 4 bytes
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
int32_t luna_conv2d_i8i4o16(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int16_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_conv2d_i8i4o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_conv2d_i8i8o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_conv2d_i8i8o16(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int16_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_conv2d_i8i8o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);

int32_t luna_depthwise2d_i8i4o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_depthwise2d_i8i4o16(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int16_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_depthwise2d_i8i4o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_depthwise2d_i8i8o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_depthwise2d_i8i8o16(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int16_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_depthwise2d_i8i8o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);

int32_t luna_deconv2d_i8i4o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_deconv2d_i8i4o16(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int16_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_deconv2d_i8i4o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_deconv2d_i8i8o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_deconv2d_i8i8o16(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int16_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_deconv2d_i8i8o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);

int32_t luna_max_pooling2d_i8o8(const int8_t *p_in, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_mean_pooling2d_i8o8(const int8_t *p_in, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_mean_pooling2d_i8o16(const int8_t *p_in, int16_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_mean_pooling2d_i8o32(const int8_t *p_in, int32_t *p_out, luna_cnn_static_para_t *conv_para_);

int32_t luna_conv1d_i8i4o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_conv1d_i8i4o16(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int16_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_conv1d_i8i4o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_conv1d_i8i8o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_conv1d_i8i8o16(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int16_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_conv1d_i8i8o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);

int32_t luna_depthwise1d_i8i4o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_depthwise1d_i8i4o16(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int16_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_depthwise1d_i8i4o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_depthwise1d_i8i8o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_depthwise1d_i8i8o16(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int16_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_depthwise1d_i8i8o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);

int32_t luna_deconv1d_i8i4o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_deconv1d_i8i4o16(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int16_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_deconv1d_i8i4o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_deconv1d_i8i8o8(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_deconv1d_i8i8o16(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int16_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_deconv1d_i8i8o32(const int8_t *p_in, int8_t *p_weight, int32_t *p_bias, int32_t *p_out, luna_cnn_static_para_t *conv_para_);

int32_t luna_max_pooling1d_i8o8(const int8_t *p_in, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_mean_pooling1d_i8o8(const int8_t *p_in, int8_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_mean_pooling1d_i8o16(const int8_t *p_in, int16_t *p_out, luna_cnn_static_para_t *conv_para_);
int32_t luna_mean_pooling1d_i8o32(const int8_t *p_in, int32_t *p_out, luna_cnn_static_para_t *conv_para_);

#endif
