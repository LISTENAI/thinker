/***************************************************************************
 * luna_misc_math.h                                                        *
 *                                                                         *
 * Copyright (C) 2020 listenai Co.Ltd                   			       *
 * All rights reserved.                                                    *
 ***************************************************************************/
#ifndef __LIBS_LUNA_LUNA_MISC_MATH_H__
#define __LIBS_LUNA_LUNA_MISC_MATH_H__

#include "luna_math_types.h"

int32_t luna_memcpy_i8o8(int8_t* dst, int8_t* src, uint32_t size);
int32_t luna_memset_i8o8(int8_t *dst, int8_t value, uint32_t size);
int32_t luna_memset_i16o16(int16_t *dst, int16_t value, uint32_t size);
int32_t luna_memset_i32o32(int32_t *dst, int32_t value, uint32_t size);
int32_t luna_psrammemcpy_i8o8(int8_t* dst, int8_t* src, uint32_t size);

int32_t luna_relu_i8o8(const int8_t *src, int8_t *dst, uint32_t size, uint32_t shift);
int32_t luna_relu_i8o16(const int8_t *src, int16_t *dst, uint32_t size, uint32_t shift);
int32_t luna_relu_i8o32(const int8_t *src, int32_t *dst, uint32_t size, uint32_t shift);
int32_t luna_relu_i16o8(const int16_t *src, int8_t *dst, uint32_t size, uint32_t shift);
int32_t luna_relu_i16o16(const int16_t *src, int16_t *dst, uint32_t size, uint32_t shift);
int32_t luna_relu_i16o32(const int16_t *src, int32_t *dst, uint32_t size, uint32_t shift);
int32_t luna_relu_i32o8(const int32_t *src, int8_t *dst, uint32_t size, uint32_t shift);
int32_t luna_relu_i32o16(const int32_t *src, int16_t *dst, uint32_t size, uint32_t shift);
int32_t luna_relu_i32o32(const int32_t *src, int32_t *dst, uint32_t size, uint32_t shift);

int32_t luna_prelu_i8o8(const int8_t *src, uint32_t slope, int8_t *dst, uint32_t size, uint32_t shift);
int32_t luna_prelu_i8o16(const int8_t *src, uint32_t slope, int16_t *dst, uint32_t size, uint32_t shift);
int32_t luna_prelu_i8o32(const int8_t *src, uint32_t slope, int32_t *dst, uint32_t size, uint32_t shift);
int32_t luna_prelu_i16o8(const int16_t *src, uint32_t slope, int8_t *dst, uint32_t size, uint32_t shift);
int32_t luna_prelu_i16o16(const int16_t *src, uint32_t slope, int16_t *dst, uint32_t size, uint32_t shift);
int32_t luna_prelu_i16o32(const int16_t *src, uint32_t slope, int32_t *dst, uint32_t size, uint32_t shift);
int32_t luna_prelu_i32o8(const int32_t *src, uint32_t slope, int8_t *dst, uint32_t size, uint32_t shift);
int32_t luna_prelu_i32o16(const int32_t *src, uint32_t slope, int16_t *dst, uint32_t size, uint32_t shift);
int32_t luna_prelu_i32o32(const int32_t *src, uint32_t slope, int32_t *dst, uint32_t size, uint32_t shift);

int32_t luna_relux_i8o8(const int8_t *src, const int8_t x, int8_t *dst, uint32_t size, uint32_t shift);
int32_t luna_relux_i16o8(const int16_t *src, const int8_t x, int8_t *dst, uint32_t size, uint32_t shift);
int32_t luna_relux_i32o8(const int32_t *src, const int8_t x, int8_t *dst, uint32_t size, uint32_t shift);

int32_t luna_sigmoid_i32o32(const int32_t *src, int32_t *dst, uint32_t size);
int32_t luna_tanh_i32o32(const int32_t *src, int32_t *dst, uint32_t size);
int32_t luna_swish_i32o32(const int32_t *src, int32_t *dst, uint32_t size);
int32_t luna_gelu_i32o32(const int32_t *src, int32_t *dst, uint32_t size);

int32_t luna_exp_i32o32(const int32_t *src, int32_t *dst, uint32_t size);
int32_t luna_softmax_i32o32(const int32_t *src, int32_t *dst, uint32_t size);
int32_t luna_logsoftmax_i32o32(const int32_t *src, int32_t *dst, uint32_t size);

int32_t luna_sigmoid_i32o16(const int32_t *src, int16_t *dst, uint32_t size);
int32_t luna_tanh_i32o16(const int32_t *src, int16_t *dst, uint32_t size);
int32_t luna_swish_i32o16(const int32_t *src, int16_t *dst, uint32_t size);
int32_t luna_gelu_i32o16(const int32_t *src, int16_t *dst, uint32_t size);

int32_t luna_sigmoid_i32o8(const int32_t *src, int8_t *dst, uint32_t size);
int32_t luna_tanh_i32o8(const int32_t *src, int8_t *dst, uint32_t size);
int32_t luna_swish_i32o8(const int32_t *src, int8_t *dst, uint32_t size);
int32_t luna_gelu_i32o8(const int32_t *src, int8_t *dst, uint32_t size);

int32_t luna_lut_i8o8(const int8_t *data, uint8_t *index, int8_t *dst, uint32_t data_size, uint32_t index_size);
int32_t luna_lut_i16o16(const int16_t *data, uint16_t *index, int16_t *dst, uint32_t data_size, uint32_t index_size);
int32_t luna_lut_i32o32(const int32_t *data, uint32_t *index, int32_t *dst, uint32_t data_size, uint32_t index_size);

int32_t luna_gpdma_start(uint32_t chn, void *dst, void *src, uint32_t size);
int32_t luna_gpdma_wait(uint32_t chn);

int32_t luna_gpdma2d_start(uint32_t chn, void *dst, void *src, uint32_t size, uint32_t sg_num, uint32_t sg_itv, uint32_t ds_num, uint32_t ds_itv);
int32_t luna_gpdma2d_wait(uint32_t chn);

int32_t luna_crc32_i8iu32ou32(int8_t *src, uint32_t init_value, uint32_t *result_value, uint32_t size);

int32_t luna_rgb_sub128_iu8o8(uint8_t *src, int8_t *dst, uint32_t size);

#endif /* __LIBS_LUNA_LUNA_MISC_MATH_H__ */
