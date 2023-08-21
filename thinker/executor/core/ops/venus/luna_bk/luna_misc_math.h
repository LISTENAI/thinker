/***************************************************************************
 * luna_misc_math.h                                                        *
 *                                                                         *
 * Copyright (C) 2020 listenai Co.Ltd                   			       *
 * All rights reserved.                                                    *
 ***************************************************************************/
#ifndef __LIBS_LUNA_LUNA_MISC_MATH_H__
#define __LIBS_LUNA_LUNA_MISC_MATH_H__

#include "luna_math_types.h"


typedef enum {
	LUNA_ACTIVATION_SIGMOID 	= 0,
	LUNA_ACTIVATION_RELU,
	LUNA_ACTIVATION_PRELU,

	LUNA_ACTIVATION_END
} LUNA_ACTIVATION;

int32_t luna_sigmoid(const q15_t *src, int16_t *dst, uint32_t size);
int32_t luna_sigmoid_int8(const q15_t *src, int8_t *dst, uint32_t size);
int32_t luna_tanh(const q15_t *src, int16_t *dst, uint32_t size);
int32_t luna_tanh_int8(const q15_t *src, int8_t *dst, uint32_t size);

int32_t luna_relu_q7_int8(const q7_t *src, int8_t *dst, uint32_t size, uint32_t post_shift);
int32_t luna_relu_q7_int16(const q7_t *src, int16_t *dst, uint32_t size, uint32_t post_shift);
int32_t luna_relu_q7_int32(const q7_t *src, int32_t *dst, uint32_t size, uint32_t post_shift);
int32_t luna_relu_q15_int8(const q15_t *src, int8_t *dst, uint32_t size, uint32_t post_shift);
int32_t luna_relu_q15_int16(const q15_t *src, int16_t *dst, uint32_t size, uint32_t post_shift);
int32_t luna_relu_q15_int32(const q15_t *src, int32_t *dst, uint32_t size, uint32_t post_shift);
int32_t luna_relu_q31_int8(const q31_t *src, int8_t *dst, uint32_t size, uint32_t post_shift);
int32_t luna_relu_q31_int16(const q31_t *src, int16_t *dst, uint32_t size, uint32_t post_shift);
int32_t luna_relu_q31_int32(const q31_t *src, int32_t *dst, uint32_t size, uint32_t post_shift);
int32_t luna_prelu_q7_int8(const q7_t *src, uint32_t slope, int8_t *dst, uint32_t size, uint32_t post_shift);
int32_t luna_prelu_q7_int16(const q7_t *src, uint32_t slope, int16_t *dst, uint32_t size, uint32_t post_shift);
int32_t luna_prelu_q7_int32(const q7_t *src, uint32_t slope, int32_t *dst, uint32_t size, uint32_t post_shift);
int32_t luna_prelu_q15_int8(const q15_t *src, uint32_t slope, int8_t *dst, uint32_t size, uint32_t post_shift);
int32_t luna_prelu_q15_int16(const q15_t *src, uint32_t slope, int16_t *dst, uint32_t size, uint32_t post_shift);
int32_t luna_prelu_q15_int32(const q15_t *src, uint32_t slope, int32_t *dst, uint32_t size, uint32_t post_shift);
int32_t luna_prelu_q31_int8(const q31_t *src, uint32_t slope, int8_t *dst, uint32_t size, uint32_t post_shift);
int32_t luna_prelu_q31_int16(const q31_t *src, uint32_t slope, int16_t *dst, uint32_t size, uint32_t post_shift);
int32_t luna_prelu_q31_int32(const q31_t *src, uint32_t slope, int32_t *dst, uint32_t size, uint32_t post_shift);
int32_t luna_memcpy(void* dst, void* src, int size);
int32_t luna_memset(void *dst, q7_t value, uint32_t size);
int32_t luna_memset_int16(void *dst, q15_t value, uint32_t size);
int32_t luna_memset_int32(void *dst, q31_t value, uint32_t size);
int32_t luna_memcpy_psram2sharemem(void* dst, void* src, int size);

#endif /* __LIBS_LUNA_LUNA_MISC_MATH_H__ */
