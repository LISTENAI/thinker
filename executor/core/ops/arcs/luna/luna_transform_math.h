/***************************************************************************
 * luna_transform_math.h                                                   *
 *                                                                         *
 * Copyright (C) 2020 listenai Co.Ltd                   			       *
 * All rights reserved.                                                    *
 ***************************************************************************/

#ifndef __LUNA_TRANSFORM_MATH_H__
#define __LUNA_TRANSFORM_MATH_H__

#include "luna_math_types.h"

#ifndef MASK_FFT_TYPEDEF
#define MASK_FFT_TYPEDEF

typedef enum fft_points
{
	FFT_64	 = 64,
	FFT_128  = 128,
	FFT_256	 = 256,
	FFT_512	 = 512,
}e_fft_points;

#endif 

int32_t luna_cfft_i32o32(const int32_t *src, int32_t *dst, int32_t *fft_rshift, e_fft_points points);
int32_t luna_cifft_i32o32(const int32_t *src, int32_t *dst, int32_t *fft_rshift, e_fft_points points);
int32_t luna_rfft_i32o32(const int32_t *src, int32_t *dst, int32_t *fft_rshift, e_fft_points points);
int32_t luna_rifft_i32o32(const int32_t *src, int32_t *dst, int32_t *fft_rshift, e_fft_points points);

#endif /* __LUNA_TRANSFORM_MATH_H__ */
