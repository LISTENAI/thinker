#ifndef __LUNA_API_CNN_H__
#define __LUNA_API_CNN_H__

#include "luna/luna.h"

typedef enum LunaCnnLayer_e
{
	CNN_LAYER_CONV = 0,
	CNN_LAYER_DEPTHWISE_CONV,
	CNN_LAYER_MAX_POOLING,
	CNN_LAYER_MEAN_POOLING,
	CNN_LAYER_DECONV,
	CNN_LAYER_SPLIT_CONV
}LunaCnnLayer_e;


typedef struct LunaCnnSettings
{
	// offset 0
	uint32_t r8_c0_c2;
	uint32_t r9_c0_c2;
	uint16_t r8_c4;
	uint16_t r9_c4;
	uint16_t r10_saddr_h;
	uint16_t r11_saddr_l;
	uint16_t r12_saddr_h;
	uint16_t r13_saddr_l;
	uint32_t r14_p0_addr;
	uint16_t r18_w_l;
	uint16_t r19_h_l;
	uint16_t r20_c_l;
	uint16_t r21_sw_h;

	// offset 8 * 4 bytes
	uint16_t r8_c0;
	uint16_t r9_c0;
	uint32_t r14_p1_addr;
	uint32_t r18_W;
	uint32_t is_bias;	//r1

	// offset 12 * 4 bytes
	uint32_t r8_st_rw_0;
	uint32_t r9_st_rw_0;
	uint32_t r8_st_rw_1;
	uint32_t r9_st_rw_1;
	uint32_t r8_st_rw_2;
	uint32_t r9_st_rw_2;
	uint32_t r8_st_rw_3;	//for depthwise
	uint32_t r9_st_rw_3;

	// offset 20 * 4 bytes
	// infmap P0 address
	uint32_t r10_c_kh_0;
	uint32_t r11_kw_w_0;
	uint32_t r12_c_kh_0;
	uint32_t r13_kw_w_0;
	uint32_t r10_h_g_0;
	uint32_t r11_gs_0;
	uint32_t r12_h_g_0;
	uint32_t r13_gs_0;

	// offset 28 * 4 bytes
	// infmap P1 address
	uint32_t r10_c_kh_1;
	uint32_t r11_kw_w_1;
	uint32_t r12_c_kh_1;
	uint32_t r13_kw_w_1;
	uint32_t r10_h_g_1;
	uint32_t r11_gs_1;
	uint32_t r12_h_g_1;
	uint32_t r13_gs_1;

	// offset 36 * 4 bytes
	uint32_t r16_h;	//
	uint32_t r1_judge_sw;	//r1
	// uint32_t r10_h_stride_ker;
	uint32_t r17;	//
	uint32_t r14_bias_addr;

	uint32_t r23_shift;
	uint32_t r22_ow;
	uint32_t r27;

	uint32_t r20_o_addr;
	uint32_t r18_ch_len;
	uint32_t r19_ch_num;
	uint32_t r21_interval;

	// offset 47 * 4 bytes
	uint32_t ou_bits;	//r2
	uint32_t weight_bits;	//r3
	LunaCnnLayer_e cnn_layer;
}LunaCnnSettings_t;

typedef struct LunaCnnSplitSettings
{
	// offset 0
	uint32_t weight_s;		// R8
	// offset 4
	uint32_t weight_addr;		// R14
	// offset 8
	uint32_t slave_w;		// R18

	uint32_t i_addr;	//R1
	uint32_t i_addr_inc_1st;	//R2
	uint32_t i_addr_inc;	//R3
	uint32_t o_addr;	//R4
	uint32_t o_addr_inc;	//R5
	uint32_t const_split_num;	//R6
	uint32_t split_num;		//R7

	uint32_t in_w;	//R9
	uint16_t in_h_1st;	//R10_L
	uint16_t in_c;	//R10_H
	uint32_t in_h_last;	//R11_L(tmp)
	uint32_t in_h;		//R12_L(tmp)
	uint16_t h_addr;	//R13_L
	uint16_t c_addr;	//R13_H

	uint32_t in_h_last_2;	//R16_L
	uint32_t in_h_2;	//R17_L
	uint32_t in_w_2;	//R18
	uint32_t in_h_1st_2;	//R19_L
	uint32_t in_c_2;	//R20_L
	uint32_t local_addr;	//R21_H
	uint32_t is_bias;	//R15

	uint16_t r8_s;	//R8_L
	uint16_t r8_t;	//R8_H
	uint32_t r9_w;	//R9
	uint32_t r10_c;	//R10
	uint16_t r8_i;	//R8_L
	uint16_t r8_j;	//R8_H
	uint32_t r9_pad_1st;	//R9
	uint32_t r10_pad_last;	//R10
	uint32_t r11_pad;	//R11
	uint16_t r10_kernel_l;	//R10_L
	uint16_t r10_kernel_h;	//R10_H
	uint32_t r11_taddr;	//R11
	uint32_t r12_waddr;	//R12
	uint32_t r13_caddr;	//R13
	uint32_t r14_base_addr;	//R14
	uint32_t kernel_size;	//R15

	uint32_t r11_jaddr;	//R11
	uint32_t r12_delta;	//R12
	uint16_t outmap_w;	//R13_L
	uint16_t outmap_h;	//R13_H
	uint32_t bias_addr;	//R14

	uint32_t shift;	//R23_H
	uint32_t integr;	//R27
	uint16_t r22_l;	//R22_L
	uint16_t r22_h;	//R22_H
	uint32_t channel_len;	//R18
	uint32_t channel_num;	//R19
	uint32_t interval;	//R21

	uint32_t activation_type;
	uint32_t prelu_shift;
	uint32_t ou_bits;
	LunaCnnLayer_e cnn_layer;
}LunaCnnSplitSettings_t;

extern __luna_cmd_attr__ uint32_t luna_api_cnn[];
extern __luna_cmd_attr__ uint32_t luna_api_depthwise[];
extern __luna_cmd_attr__ uint32_t luna_api_deconv[];
extern __luna_cmd_attr__ uint32_t luna_api_meanpool[];
extern __luna_cmd_attr__ uint32_t luna_api_maxpool[];

extern __luna_cmd_attr__ uint32_t luna_api_split_cnn[];
extern __luna_cmd_attr__ uint32_t luna_api_split_depthwise[];
extern __luna_cmd_attr__ uint32_t luna_api_split_pool[];
extern __luna_cmd_attr__ uint32_t luna_api_split_deconv[];
#endif

