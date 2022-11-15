/***************************************************************************
 * luna_transform_math.h                                                   *
 *                                                                         *
 * Copyright (C) 2020 listenai Co.Ltd                   			       *
 * All rights reserved.                                                    *
 ***************************************************************************/

#ifndef __LUNA_TRANSFORM_MATH_H__
#define __LUNA_TRANSFORM_MATH_H__

#include "luna_math_types.h"

//////////////////////////////////////////////////////////////////////
int32_t luna_cfft64_q31(const q31_t *src,   q31_t *dst, int32_t *shift);
int32_t luna_cfft128_q31(const q31_t *src,   q31_t *dst, int32_t *shift);
int32_t luna_cfft256_q31(const q31_t *src,   q31_t *dst, int32_t *shift);
int32_t luna_cfft512_q31(const q31_t *src,   q31_t *dst, int32_t *shift);
int32_t luna_cfft1024_q31(const q31_t *src,  q31_t *dst, int32_t *shift);

int32_t luna_cifft64_q31(const q31_t *src,  q31_t *dst, int32_t *shift);
int32_t luna_cifft128_q31(const q31_t *src,  q31_t *dst, int32_t *shift);
int32_t luna_cifft256_q31(const q31_t *src,  q31_t *dst, int32_t *shift);
int32_t luna_cifft512_q31(const q31_t *src,  q31_t *dst, int32_t *shift);
int32_t luna_cifft1024_q31(const q31_t *src, q31_t *dst, int32_t *shift);

//////////////////////////////////////////////////////////////////////
//int32_t luna_cfft64_rrii_q31(const q31_t *src,   q31_t *dst, int32_t *shift);
int32_t luna_cfft128_rrii_q31(const q31_t *src,   q31_t *dst, int32_t *shift);
int32_t luna_cfft256_rrii_q31(const q31_t *src,   q31_t *dst, int32_t *shift);
int32_t luna_cfft512_rrii_q31(const q31_t *src,   q31_t *dst, int32_t *shift);
int32_t luna_cfft1024_rrii_q31(const q31_t *src,  q31_t *dst, int32_t *shift);

//int32_t luna_cifft64_rrii_q31(const q31_t *src,  q31_t *dst, int32_t *shift);
int32_t luna_cifft128_rrii_q31(const q31_t *src,  q31_t *dst, int32_t *shift);
int32_t luna_cifft256_rrii_q31(const q31_t *src,  q31_t *dst, int32_t *shift);
int32_t luna_cifft512_rrii_q31(const q31_t *src,  q31_t *dst, int32_t *shift);
int32_t luna_cifft1024_rrii_q31(const q31_t *src, q31_t *dst, int32_t *shift);

//////////////////////////////////////////////////////////////////////
//int32_t luna_rfft64_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rfft128_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rfft256_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rfft512_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rfft1024_q31(const q31_t *src, q31_t *dst, int32_t *shift);

//int32_t luna_rifft64_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft128_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft256_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft512_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft1024_q31(const q31_t *src, q31_t *dst, int32_t *shift);

//////////////////////////////////////////////////////////////////////
//int32_t luna_rfft64_rrii_q31(const q31_t *src,   q31_t *dst, int32_t *shift);
int32_t luna_rfft128_rrii_q31(const q31_t *src,   q31_t *dst, int32_t *shift);
int32_t luna_rfft256_rrii_q31(const q31_t *src,   q31_t *dst, int32_t *shift);
int32_t luna_rfft512_rrii_q31(const q31_t *src,   q31_t *dst, int32_t *shift);
int32_t luna_rfft1024_rrii_q31(const q31_t *src,  q31_t *dst, int32_t *shift);

//int32_t luna_rifft64_rrii_q31(const q31_t *src,  q31_t *dst, int32_t *shift);
int32_t luna_rifft128_rrii_q31(const q31_t *src,  q31_t *dst, int32_t *shift);
int32_t luna_rifft256_rrii_q31(const q31_t *src,  q31_t *dst, int32_t *shift);
int32_t luna_rifft512_rrii_q31(const q31_t *src,  q31_t *dst, int32_t *shift);
int32_t luna_rifft1024_rrii_q31(const q31_t *src, q31_t *dst, int32_t *shift);

//////////////////////////////////////////////////////////////////////

int32_t luna_rfft64_64rrrr_32riri_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rfft128_128rrrr_64riri_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rfft256_256rrrr_128riri_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rfft512_512rrrr_256riri_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rfft512_512rrrr_256riri_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rfft1024_1024rrrr_512riri_q31(const q31_t *src, q31_t *dst, int32_t *shift);

//int32_t luna_rfft64_64rrrr_32rrii2_q31(const q31_t *src, q31_t *dst1, q31_t *dst2, int32_t *shift);
int32_t luna_rfft128_128rrrr_64rrii2_q31(const q31_t *src, q31_t *dst1, q31_t *dst2, int32_t *shift);
int32_t luna_rfft256_256rrrr_128rrii2_q31(const q31_t *src, q31_t *dst1, q31_t *dst2, int32_t *shift);
int32_t luna_rfft512_512rrrr_256rrii2_q31(const q31_t *src, q31_t *dst1, q31_t *dst2, int32_t *shift);
int32_t luna_rfft512_512rrrr_256rrii2_q31(const q31_t *src, q31_t *dst1, q31_t *dst2, int32_t *shift);
int32_t luna_rfft1024_1024rrrr_512rrii2_q31(const q31_t *src, q31_t *dst1, q31_t *dst2, int32_t *shift);

//////////////////////////////////////////////////////////////////////

//int32_t luna_rifft64_32riri_64rrrr_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft128_64riri_128rrrr_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft256_128riri_256rrrr_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft512_256riri_512rrrr_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft1024_512riri_1024rrrr_q31(const q31_t *src, q31_t *dst, int32_t *shift);

//int32_t luna_rifft64_32rrii2_64rrrr_q31(const q31_t *src1, q31_t *src2, q31_t *dst, int32_t *shift);
int32_t luna_rifft128_64rrii2_128rrrr_q31(const q31_t *src1, q31_t *src2, q31_t *dst, int32_t *shift);
int32_t luna_rifft256_128rrii2_256rrrr_q31(const q31_t *src1, q31_t *src2, q31_t *dst, int32_t *shift);
int32_t luna_rifft512_256rrii2_512rrrr_q31(const q31_t *src1, q31_t *src2, q31_t *dst, int32_t *shift);
int32_t luna_rifft1024_512rrii2_1024rrrr_q31(const q31_t *src1, q31_t *src2, q31_t *dst, int32_t *shift);

//int32_t luna_rifft64_33riri_64rrrr_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft128_65riri_128rrrr_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft256_129riri_256rrrr_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft512_257riri_512rrrr_q31(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft1024_513riri_1024rrrr_q31(const q31_t *src, q31_t *dst, int32_t *shift);

//////////////////////////////////////////////////////////////////////

//int32_t luna_rifft64_32riri_64rrrr_shift(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft128_64riri_128rrrr_shift(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft256_128riri_256rrrr_shift(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft512_256riri_512rrrr_shift(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft1024_512riri_1024rrrr_shift(const q31_t *src, q31_t *dst, int32_t *shift);

//int32_t luna_rifft64_32rrii2_64rrrr_shift(const q31_t *src1, q31_t *src2, q31_t *dst, int32_t *shift);
int32_t luna_rifft128_64rrii2_128rrrr_shift(const q31_t *src1, q31_t *src2, q31_t *dst, int32_t *shift);
int32_t luna_rifft256_128rrii2_256rrrr_shift(const q31_t *src1, q31_t *src2, q31_t *dst, int32_t *shift);
int32_t luna_rifft512_256rrii2_512rrrr_shift(const q31_t *src1, q31_t *src2, q31_t *dst, int32_t *shift);
int32_t luna_rifft1024_512rrii2_1024rrrr_shift(const q31_t *src1, q31_t *src2, q31_t *dst, int32_t *shift);

//int32_t luna_rifft64_33riri_64rrrr_shift(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft128_65riri_128rrrr_shift(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft256_129riri_256rrrr_shift(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft512_257riri_512rrrr_shift(const q31_t *src, q31_t *dst, int32_t *shift);
int32_t luna_rifft1024_513riri_1024rrrr_shift(const q31_t *src, q31_t *dst, int32_t *shift);

#endif /* __LUNA_TRANSFORM_MATH_H__ */
