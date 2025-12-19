#ifndef __LUNA_CNN_TOOLS_H__
#define __LUNA_CNN_TOOLS_H__

#include <stdio.h>
#include <stdint.h>
#include "luna_cnn_math.h"

#ifndef MASK_SIM_TYPEDEF_TOOL
#define MASK_SIM_TYPEDEF_TOOL

#define CONV_IN_CONDITION	(16 * 1024)
#define CONV_WEIGHT_CONDITION	(8 * 1024)

typedef enum {
	LUNA_CONV = 0,
	LUNA_DECONV = 1,
	LUNA_DEPTHWISE = 2,
	LUNA_MAX_POOLING = 3,
    LUNA_MEAN_POOLING = 4,
	LUNA_CONV1D,
	LUNA_DECONV1D,
	LUNA_DEPTHWISE1D,
	LUNA_MAX_POOLING1D,
    LUNA_MEAN_POOLING1D,
} LUNA_CNN_OP;
#endif //MASK_SIM_TYPEDEF_TOOL

// reshape weight
int32_t reshape_weight_for_conv(int8_t *input_weight, int8_t *input_weight_T, conv_struct_t *conv_struct_);
int32_t reshape_weight_for_depthwise(int8_t *input_weight, int8_t *input_weight_T, conv_struct_t *conv_struct_);
int32_t reshape_weight_for_conv_4bit(int8_t *input_weight, int8_t *input_weight_T, conv_struct_t *conv_struct_);
int32_t reshape_weight_for_depthwise_4bit(int8_t *input_weight, int8_t *input_weight_T, conv_struct_t *conv_struct_);
int32_t reshape_weight_for_deconv(int8_t *input_weight, int8_t *input_weight_T, conv_struct_t *conv_struct_);
int32_t reshape_weight_for_deconv_4bit(int8_t *input_weight, int8_t *input_weight_T, conv_struct_t *conv_struct_);

int32_t luna_split_conv_para_pack(conv_struct_t *conv_st, luna_cnn_static_para_t *p_cnn_static_para, int32_t cnn_layer_type);

int32_t luna_deconv_torch2convst(conv_struct_t *conv_struct_);

#endif // __LUNA_CNN_TOOLS_H__
