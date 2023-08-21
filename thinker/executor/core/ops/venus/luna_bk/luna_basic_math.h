/***************************************************************************
 * .h                                                  *
 *                                                                         *
 * Copyright (C) 2020 listenai Co.Ltd                   			       *
 * All rights reserved.                                                    *
 ***************************************************************************/
#ifndef __LUNA_BASIC_MATH_H__
#define __LUNA_BASIC_MATH_H__


#include "luna_math_types.h"

typedef enum
{
	LUNA_CMP_GREATER_THAN = 0,
	LUNA_CMP_GREATER_OR_EQUAL,
	LUNA_CMP_LESS_THAN,
	LUNA_CMP_LESS_OR_EQUAL,
	LUNA_CMP_EQUAL,
} VEC_CMP_MODE;

/**
 * @brief Dot Product of q7 vectors.
 * @param[in]       *src1 points to the first input vector.
 * @param[in]       *src2 points to the second input vector.
 * @param[out]      *dst  points to the output scalar of dot product of two input vectors..
 * @param[in]       size  size of the vectors. should aligned to 32-Points(256bits/8bits)
 * @return function execute result.
 *
 * Ouput results will be saturated in Q31 range [0x80000000 0x7FFFFFFF].
 */
/**
 * @brief Addition of q7 vectors.
 * @param[in]       *src1 points to the first input vector.
 * @param[in]       *src2 points to the second input vector.
 * @param[out]      *dst  points to the output vector.
 * @param[in]       size  size of the vectors.
 * @return none.
 *
 * Ouput results will be saturated in Q7 range [0x80 0x7F].
 */
int32_t luna_add_q7_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t size, uint32_t shift);
int32_t luna_add_q7_int16(const q7_t* src1, const q7_t* src2, q15_t* dst, uint32_t size, uint32_t shift);
int32_t luna_add_q7_int32(const q7_t* src1, const q7_t* src2, q31_t* dst, uint32_t size, uint32_t shift);
int32_t luna_add_q15_int8(const q15_t* src1, const q15_t* src2, q7_t* dst, uint32_t size, uint32_t shift);
int32_t luna_add_q15_int16(const q15_t *src1, const q15_t *src2, q15_t *dst, uint32_t size, uint32_t shift);;
int32_t luna_add_q15_int32(const q15_t *src1, const q15_t *src2, q31_t *dst, uint32_t size, uint32_t shift);
int32_t luna_add_q31_int8(const q31_t *src1, const q31_t *src2, q7_t *dst, uint32_t size, uint32_t shift);
int32_t luna_add_q31_int16(const q31_t *src1, const q31_t *src2, q15_t *dst, uint32_t size, uint32_t shift);
int32_t luna_add_q31_int32(const q31_t *src1, const q31_t *src2, q31_t *dst, uint32_t size, uint32_t shift);

int32_t luna_sub_q7_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t size, uint32_t shift);
int32_t luna_sub_q7_int16(const q7_t* src1, const q7_t* src2, q15_t* dst, uint32_t size, uint32_t shift);
int32_t luna_sub_q7_int32(const q7_t* src1, const q7_t* src2, q31_t* dst, uint32_t size, uint32_t shift);
int32_t luna_sub_q15_int8(const q15_t* src1, const q15_t* src2, q7_t* dst, uint32_t size, uint32_t shift);
int32_t luna_sub_q15_int16(const q15_t *src1, const q15_t *src2, q15_t *dst, uint32_t size, uint32_t shift);
int32_t luna_sub_q15_int32(const q15_t *src1, const q15_t *src2, q31_t *dst, uint32_t size, uint32_t shift);
int32_t luna_sub_q31_int8(const q31_t *src1, const q31_t *src2, q7_t *dst, uint32_t size, uint32_t shift);
int32_t luna_sub_q31_int16(const q31_t *src1, const q31_t *src2, q15_t *dst, uint32_t size, uint32_t shift);
int32_t luna_sub_q31_int32(const q31_t *src1, const q31_t *src2, q31_t *dst, uint32_t size, uint32_t shift);

int32_t luna_mul_q7_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t size, uint32_t shift);
int32_t luna_mul_q7_int16(const q7_t *src1, const q7_t *src2, q15_t *dst, uint32_t size, uint32_t shift);
int32_t luna_mul_q7_int32(const q7_t *src1, const q7_t *src2, q31_t *dst, uint32_t size, uint32_t shift);
int32_t luna_mul_q15_int8(const q15_t *src1, const q15_t *src2, q7_t *dst, uint32_t size, uint32_t shift);
int32_t luna_mul_q15_int16(const q15_t *src1, const q15_t *src2, q15_t *dst, uint32_t size, uint32_t shift);
int32_t luna_mul_q15_int32(const q15_t *src1, const q15_t *src2, q31_t *dst, uint32_t size, uint32_t shift);
int32_t luna_mul_q31_int8(const q31_t *src1, const q31_t *src2, q7_t *dst, uint32_t size, uint32_t shift);
int32_t luna_mul_q31_int16(const q31_t *src1, const q31_t *src2, q15_t *dst, uint32_t size, uint32_t shift);
int32_t luna_mul_q31_int32(const q31_t *src1, const q31_t *src2, q31_t *dst, uint32_t size, uint32_t shift);

int32_t luna_scale_q7_int8(const q7_t *src1, const q7_t scalar, q7_t *dst, uint32_t size, uint32_t shift);
int32_t luna_scale_q7_int16(const q7_t *src1, const q7_t scalar, q15_t *dst, uint32_t size, uint32_t shift);
int32_t luna_scale_q7_int32(const q7_t *src1, const q7_t scalar, q31_t *dst, uint32_t size, uint32_t shift);
int32_t luna_scale_q15_int8(const q15_t *src1, const q15_t scalar, q7_t *dst, uint32_t size, uint32_t shift);
int32_t luna_scale_q15_int16(const q15_t *src1, const q15_t scalar, q15_t *dst, uint32_t size, uint32_t shift);
int32_t luna_scale_q15_int32(const q15_t *src1, const q15_t scalar, q31_t *dst, uint32_t size, uint32_t shift);
int32_t luna_scale_q31_int8(const q31_t *src1, const q31_t scalar, q7_t *dst, uint32_t size, uint32_t shift);
int32_t luna_scale_q31_int16(const q31_t *src1, const q31_t scalar, q15_t *dst, uint32_t size, uint32_t shift);
int32_t luna_scale_q31_int32(const q31_t *src1, const q31_t scalar, q31_t *dst, uint32_t size, uint32_t shift);

int32_t luna_dot_prod_q7_int8(const q7_t *src1, const q7_t *src2, int8_t *dst, uint32_t size, uint32_t shift);
int32_t luna_dot_prod_q7_int16(const q7_t *src1, const q7_t *src2, int16_t *dst, uint32_t size, uint32_t shift);
int32_t luna_dot_prod_q7_int32(const q7_t *src1, const q7_t *src2, int32_t *dst, uint32_t size, uint32_t shift);
int32_t luna_dot_prod_q15_int8(const q15_t *src1, const q15_t *src2, int8_t *dst, uint32_t size, uint32_t shift);
int32_t luna_dot_prod_q15_int16(const q15_t *src1, const q15_t *src2, int16_t *dst, uint32_t size, uint32_t shift);
int32_t luna_dot_prod_q15_int32(const q15_t *src1, const q15_t *src2, int32_t *dst, uint32_t size, uint32_t shift);
int32_t luna_dot_prod_q31_int8(const q31_t *src1, const q31_t *src2, int8_t *dst, uint32_t size, uint32_t shift);
int32_t luna_dot_prod_q31_int16(const q31_t *src1, const q31_t *src2, int16_t *dst, uint32_t size, uint32_t shift);
int32_t luna_dot_prod_q31_int32(const q31_t *src1, const q31_t *src2, int32_t *dst, uint32_t size, uint32_t shift);
int32_t luna_dot_prod_q31_int64(const q31_t *src1, const q31_t *src2, int64_t *dst, uint32_t size, uint32_t shift);

int32_t luna_offset_q7_int8(const  q7_t *src, const q7_t offset, q7_t *dst, uint32_t size, uint32_t shift);
int32_t luna_offset_q7_int16(const  q7_t *src, const q7_t offset, q15_t *dst, uint32_t size, uint32_t shift);
int32_t luna_offset_q7_int32(const  q7_t *src, const q7_t offset, q31_t *dst, uint32_t size, uint32_t shift);
int32_t luna_offset_q15_int8(const  q15_t *src, const q15_t offset, q7_t *dst, uint32_t size, uint32_t shift);
int32_t luna_offset_q15_int16(const  q15_t *src, const q15_t offset, q15_t *dst, uint32_t size, uint32_t shift);
int32_t luna_offset_q15_int32(const  q15_t *src, const q15_t offset, q31_t *dst, uint32_t size, uint32_t shift);
int32_t luna_offset_q31_int8(const  q31_t *src, const q31_t offset, q7_t *dst, uint32_t size, uint32_t shift);
int32_t luna_offset_q31_int16(const  q31_t *src, const q31_t offset, q15_t *dst, uint32_t size, uint32_t shift);
int32_t luna_offset_q31_int32(const  q31_t *src, const q31_t offset, q31_t *dst, uint32_t size, uint32_t shift);

int32_t luna_cmp_vv_q7_int8(const q7_t *src1, const q7_t *src2, int8_t *dst, uint32_t size, uint32_t cmp_mode);
int32_t luna_cmp_vv_q15_int16(const q15_t *src1, const q15_t *src2, int16_t *dst, uint32_t size, uint32_t cmp_mode);
int32_t luna_cmp_vv_q31_int32(const q31_t *src1, const q31_t *src2, int32_t *dst, uint32_t size, uint32_t cmp_mode);
int32_t luna_cmp_vs_q7_int8(const q7_t *src1, const q7_t scalar, int8_t *dst, uint32_t size, uint32_t cmp_mode);
int32_t luna_cmp_vs_q15_int16(const q15_t *src1, const q15_t scalar, int16_t *dst, uint32_t size, uint32_t cmp_mode);
int32_t luna_cmp_vs_q31_int32(const q31_t *src1, const q31_t scalar, int32_t *dst, uint32_t size, uint32_t cmp_mode);

int32_t luna_max_q7(const q7_t *src, q31_t *dst, uint32_t size);
int32_t luna_min_q7(const q7_t *src, q31_t *dst, uint32_t size);
int32_t luna_max_q15(const q15_t *src, q31_t *dst, uint32_t size);
int32_t luna_min_q15(const q15_t *src, q31_t *dst, uint32_t size);
int32_t luna_max_q31(const q31_t *src, q31_t* dst, uint32_t size);
int32_t luna_min_q31(const q31_t *src, q31_t *dst, uint32_t size);

int32_t luna_mul_q7q3_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t length, uint32_t shift);
int32_t luna_mul_q7q3_int16(const q7_t *src1, const q7_t *src2, q15_t *dst, uint32_t length, uint32_t shift);
int32_t luna_mul_q7q3_int32(const q7_t *src1, const q7_t *src2, q31_t *dst, uint32_t length, uint32_t shift);
int32_t luna_mul_q15q7_int32(const q15_t *src1, const q7_t *src2, q31_t *dst, uint32_t length, uint32_t shift);
int32_t luna_mul_q31q7_int32(const q31_t *src1, const q7_t *src2, q31_t *dst, uint32_t length, uint32_t shift);
int32_t luna_mul_q31q15_int32(const q31_t *src1, const q15_t *src2, q31_t *dst, uint32_t length, uint32_t shift);

int32_t luna_div_q31_int32(const q31_t *src1, uint32_t q_src1, const q31_t *src2, uint32_t q_src2, int32_t *dst, uint32_t q_out, uint32_t size);

//Z = a*X + b*Y
int32_t luna_scale_add_q7_int8(const q7_t *src1, const q7_t scale_a, const q7_t *src2, const q7_t scale_b, q7_t *dst, uint32_t size, uint32_t shift);
int32_t luna_scale_add_q15_int16(const q15_t *src1, const q15_t scale_a, const q15_t *src2, const q15_t scale_b, q15_t *dst, uint32_t size, uint32_t shift);
int32_t luna_scale_add_q31_int32(const q31_t *src1, const q31_t scale_a, const q31_t *src2, const q31_t scale_b, q31_t *dst, uint32_t size, uint32_t shift);

int32_t luna_vector_sum_q7_int32(const q7_t *src, int32_t *dst, uint32_t size, uint32_t shift);
int32_t luna_vector_sum_q15_int32(const q15_t *src, int32_t *dst, uint32_t size, uint32_t shift);
int32_t luna_vector_sum_q31_int32(const q31_t *src, int32_t *dst, uint32_t size, uint32_t shift);
int32_t luna_vector_sum_q31_int64(const q31_t *src, int64_t *dst, uint32_t size, uint32_t shift);

/**
 * (M, N)*(1, N)=(M, N)
 */
int32_t luna_multi_vec_mul_q7_int8(const q7_t *src1, const q7_t *src2, int8_t *dst, uint32_t src2_size, uint32_t multi_times, uint32_t shift);
int32_t luna_multi_vec_mul_q7_int16(const q7_t *src1, const q7_t *src2, int16_t *dst, uint32_t src2_size, uint32_t multi_times, uint32_t shift);
int32_t luna_multi_vec_mul_q7_int32(const q7_t *src1, const q7_t *src2, int32_t *dst, uint32_t src2_size, uint32_t multi_times, uint32_t shift);
int32_t luna_multi_vec_mul_q15_int16(const q15_t *src1, const q15_t *src2, int16_t *dst, uint32_t src2_size, uint32_t multi_times, uint32_t shift);
int32_t luna_multi_vec_mul_q15_int32(const q15_t *src1, const q15_t *src2, int32_t *dst, uint32_t src2_size, uint32_t multi_times, uint32_t shift);
int32_t luna_multi_vec_mul_q31_int32(const q31_t *src1, const q31_t *src2, int32_t *dst, uint32_t src2_size, uint32_t multi_times, uint32_t shift);

/**
 * N1 + N2 + ... + Nm
 */
int32_t luna_multi_vec_add_q31_int32(const q31_t *src, int32_t *dst, uint32_t dst_size, uint32_t multi_times, uint32_t shift);

#endif // __LUNA_BASIC_MATH_H__
