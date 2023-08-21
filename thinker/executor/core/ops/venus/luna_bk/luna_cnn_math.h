/***************************************************************************
 * .h                                                  *
 *                                                                         *
 * Copyright (C) 2020 listenai Co.Ltd                   			       *
 * All rights reserved.                                                    *
 ***************************************************************************/
#ifndef __LUNA_CNN_MATH_H__
#define __LUNA_CNN_MATH_H__

#include "luna_math_types.h"

typedef enum
{
	PoolMethod_MAX = 0,
	PoolMethod_AVE = 1,
	PoolMethod_None = 2
} Pooling_Type;


typedef enum activation_type
{
	RELU	 = 0,
	PRELU    = 1,
	SIGMOID	 = 2,
	TANH	 = 3,

	NO_ACTIVE = 128,
}e_activation_type;

typedef enum ShiftType
{
	ShiftType_FloorX05 = 0,	//(floor(x+0.5))
	ShiftType_Floor,
}E_ShiftType;

typedef struct conv_struct
{
	uint32_t input_c;
	uint32_t input_w;
	uint32_t input_h;
	uint32_t padding_w_left;
	uint32_t padding_w_right;
	uint32_t padding_h_up;
	uint32_t padding_h_down;
	uint32_t input_w_after_padding;
	uint32_t input_h_after_padding;
	uint32_t weight_w;
	uint32_t weight_h;
	uint32_t stride_w;
	uint32_t stride_h;
	uint32_t output_c;
	uint32_t output_w;
	uint32_t output_h;
	uint32_t is_bias;
	uint32_t is_activation;
	uint32_t activation_type;
	uint32_t positive_shift_value;
	uint32_t positive_shift_type;
	uint32_t negative_shift_value;
	uint32_t negative_shift_type;
	uint32_t pooling_type;
	uint32_t batch_num;
}s_conv_struct;

int32_t luna_conv_q7_int8(const int8_t *pInput, int8_t *pWeight, int32_t *pBias, int8_t *pOutput,  struct conv_struct *conv_struct_);
int32_t luna_conv_q7_int16(const int8_t *pInput, int8_t *pWeight, int32_t *pBias, int16_t *pOutput, struct conv_struct *conv_struct_);
int32_t luna_conv_q7_int32(const int8_t *pInput, int8_t *pWeight, int32_t *pBias, int32_t *pOutput, struct conv_struct *conv_struct_);

int32_t luna_depthwise_conv_q7_int8(const int8_t *pInput, int8_t *pWeight, int32_t *pBias, int8_t *pOutput,  struct conv_struct *conv_struct_);
int32_t luna_depthwise_conv_q7_int16(const int8_t *pInput, int8_t *pWeight, int32_t *pBias, int16_t *pOutput, struct conv_struct *conv_struct_);
int32_t luna_depthwise_conv_q7_int32(const int8_t *pInput, int8_t *pWeight, int32_t *pBias, int32_t *pOutput, struct conv_struct *conv_struct_);

int32_t luna_deconv_q7_int8( const int8_t *pInput, int8_t *pWeight, int32_t *pBias, int8_t *pOutput,  struct conv_struct *conv_struct_);
int32_t luna_deconv_q7_int16(const int8_t *pInput, int8_t *pWeight, int32_t *pBias, int16_t *pOutput, struct conv_struct *conv_struct_);
int32_t luna_deconv_q7_int32(const int8_t *pInput, int8_t *pWeight, int32_t *pBias, int32_t *pOutput, struct conv_struct *conv_struct_);

int32_t luna_max_pooling(const int8_t *pInput, int8_t *pOutput, struct conv_struct *conv_struct_);
int32_t luna_mean_pooling_int8(const int8_t *pInput, int8_t *pOutput, struct conv_struct *conv_struct_);
int32_t luna_mean_pooling_int16(const int8_t *pInput, int16_t *pOutput, struct conv_struct *conv_struct);

int32_t luna_conv_intx_int8(const int8_t* pInput, int8_t* pWeight, int32_t* pBias, int8_t* pOutput, struct conv_struct* conv_struct_, uint32_t in_bits);
int32_t luna_conv_intx_int16(const int8_t* pInput, int8_t* pWeight, int32_t* pBias, int16_t* pOutput, struct conv_struct* conv_struct_, uint32_t in_bits);
int32_t luna_conv_intx_int32(const int8_t* pInput, int8_t* pWeight, int32_t* pBias, int32_t* pOutput, struct conv_struct* conv_struct_, uint32_t in_bits);

int32_t luna_depthwise_conv_intx_int8(const int8_t* pInput, int8_t* pWeight, int32_t* pBias, int8_t* pOutput, struct conv_struct* conv_struct_, uint32_t in_bits);
int32_t luna_depthwise_conv_intx_int16(const int8_t* pInput, int8_t* pWeight, int32_t* pBias, int16_t* pOutput, struct conv_struct* conv_struct_, uint32_t in_bits);
int32_t luna_depthwise_conv_intx_int32(const int8_t* pInput, int8_t* pWeight, int32_t* pBias, int32_t* pOutput, struct conv_struct* conv_struct_, uint32_t in_bits);

int32_t luna_deconv_intx_int8(const int8_t* pInput, int8_t* pWeight, int32_t* pBias, int8_t* pOutput, struct conv_struct* conv_struct_, uint32_t in_bits);
int32_t luna_deconv_intx_int16(const int8_t* pInput, int8_t* pWeight, int32_t* pBias, int16_t* pOutput, struct conv_struct* conv_struct_, uint32_t in_bits);
int32_t luna_deconv_intx_int32(const int8_t* pInput, int8_t* pWeight, int32_t* pBias, int32_t* pOutput, struct conv_struct* conv_struct_, uint32_t in_bits);

int32_t luna_conv_split_q7_int8(const int8_t* pInput, int8_t* pWeight, int32_t* pBias, int8_t* pOutput, struct conv_struct* conv_struct_);
int32_t luna_conv_split_q7_int16(const int8_t* pInput, int8_t* pWeight, int32_t* pBias, int16_t* pOutput, struct conv_struct* conv_struct_);
int32_t luna_conv_split_q7_int32(const int8_t* pInput, int8_t* pWeight, int32_t* pBias, int32_t* pOutput, struct conv_struct* conv_struct_);

#endif
