/***************************************************************************
 * .h                                                  *
 *                                                                         *
 * Copyright (C) 2020 listenai Co.Ltd                   			       *
 * All rights reserved.                                                    *
 ***************************************************************************/
#ifndef __LUNA_COMPLEX_MATH_H__
#define __LUNA_COMPLEX_MATH_H__

/**
 * @defgroup complex Complex Functions
 * This set of functions operates on complex data vectors.
 * The data in the input <code>src</code> vector and output <code>dst</code>
 * are arranged in the array as: [real, imag, real, imag, real, imag, ...).
 */
#include "luna_math_types.h"

// Complex Multiplication
/**
 * @brief Multiply two q15 complex vector.
 * @param[in]		*src1 the first input complex vector.
 * @param[in]		*src2 the second input complex vector.
 * @param[out]		*dst  output complex vector.
 * @param[in]		size size of the vectors.
 * @return none.
 *
 * The multiplication outputs are in 1.27 x 1.27 = 2.30 format and
 * finally output is shift into 3.13 format.
 */

int32_t luna_clx_mul_i32i32o8(const int32_t *src1, const int32_t *src2, int8_t *dst, uint32_t size, uint32_t shift);
int32_t luna_clx_mul_i32i32o32(const int32_t *src1, const int32_t *src2, int32_t *dst, uint32_t size, uint32_t shift);
int32_t luna_clx_conj_mul_i32i32o32(const int32_t *src1, const int32_t *src2, int32_t *dst, uint32_t size, uint32_t shift);
int32_t luna_clx_mul_real_i32i32o32(const int32_t *src1, const int32_t *src2, int32_t *dst, uint32_t size, uint32_t shift);
int32_t luna_clx_mul_out_real_i32i32o32(const int32_t *src1, const int32_t *src2, int32_t *dst, uint32_t size, uint32_t shift);
int32_t luna_conjugate_i8o8(const int8_t *src, int8_t *dst, uint32_t size, uint32_t shift);
int32_t luna_conjugate_i32o32(const int32_t *src, int32_t *dst, uint32_t size, uint32_t shift);
int32_t luna_power_spectrum_i32o32(const int32_t *src, int32_t *dst, uint32_t size, uint32_t shift);
int32_t luna_power_spectrum_i32o64(const int32_t *src, int64_t *dst, uint32_t size, uint32_t shift);

#endif // __LUNA_COMPLEX_MATH_H__
