#ifndef __NNBLAS_BIAS_MATH_H__
#define __NNBLAS_BIAS_MATH_H__

#include "luna_math_types.h"

int32_t nnblas_add_q7_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_add_q7_int16(const q7_t *src1, const q7_t *src2, q15_t * dst, uint32_t size, uint32_t shift);

int32_t nnblas_add_q7_int32(const q7_t *src1, const q7_t *src2, q31_t * dst, uint32_t size, uint32_t shift);

int32_t nnblas_add_q15_int8(const q15_t *src1, const q15_t *src2, q7_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_add_q15_int16(const q15_t *src1, const q15_t *src2, q15_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_add_q15_int32(const q15_t *src1, const q15_t *src2, q31_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_add_q31_int8(const q31_t * src1, const q31_t * src2, q7_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_add_q31_int16(const q31_t *src1, const q31_t *src2, q15_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_add_q31_int32(const q31_t *src1, const q31_t *src2, q31_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_sub_q7_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_sub_q7_int16(const q7_t *src1, const q7_t *src2, q15_t * dst, uint32_t size, uint32_t shift);

int32_t nnblas_sub_q7_int32(const q7_t *src1, const q7_t *src2, q31_t * dst, uint32_t size, uint32_t shift);

int32_t nnblas_sub_q15_int8(const q15_t *src1, const q15_t *src2, q7_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_sub_q15_int16(const q15_t *src1, const q15_t *src2, q15_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_sub_q15_int32(const q15_t *src1, const q15_t *src2, q31_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_sub_q31_int8(const q31_t *src1, const q31_t *src2, q7_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_sub_q31_int16(const q31_t *src1, const q31_t *src2, q15_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_sub_q31_int32(const q31_t *src1, const q31_t *src2, q31_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_mul_q7_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_mul_q7_int16(const q7_t *src1, const q7_t *src2, q15_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_mul_q7_int32(const q7_t *src1, const q7_t *src2, q31_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_mul_q15_int8(const q15_t *src1, const q15_t *src2, q7_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_mul_q15_int16(const q15_t *src1, const q15_t *src2, q15_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_mul_q15_int32(const q15_t *src1, const q15_t *src2, q31_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_mul_q31_int8(const q31_t *src1, const q31_t *src2, q7_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_mul_q31_int16(const q31_t *src1, const q31_t *src2, q15_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_mul_q31_int32(const q31_t *src1, const q31_t *src2, q31_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_scale_q7_int8(const q7_t *src, const int8_t scalar, q7_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_scale_q7_int16(const q7_t *src, const int8_t scalar, q15_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_scale_q7_int32(const q7_t *src, const int8_t scalar, q31_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_scale_q15_int8(const q15_t *src, const int16_t scalar, q7_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_scale_q15_int16(const q15_t *src, const int16_t scalar, q15_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_scale_q15_int32(const q15_t *src, const int16_t scalar, q31_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_scale_q31_int8(const q31_t *src, const int32_t scalar, q7_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_scale_q31_int16(const q31_t *src, const int32_t scalar, q15_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_scale_q31_int32(const q31_t *src, const int32_t scalar, q31_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_dot_prod_q7_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_dot_prod_q7_int16(const q7_t *src1, const q7_t *src2, q15_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_dot_prod_q7_int32(const q7_t *src1, const q7_t *src2, q31_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_dot_prod_q15_int8(const q15_t *src1, const q15_t *src2, q7_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_dot_prod_q15_int16(const q15_t *src1, const q15_t *src2, q15_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_dot_prod_q15_int32(const q15_t *src1, const q15_t *src2, q31_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_dot_prod_q31_int8(const q31_t *src1, const q31_t *src2, q7_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_dot_prod_q31_int16(const q31_t *src1, const q31_t *src2, q15_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_dot_prod_q31_int32(const q31_t *src1, const q31_t *src2, q31_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_offset_q7_int8(const  q7_t *src, const int8_t offset, q7_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_offset_q7_int16(const  q7_t *src, const int8_t offset, q15_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_offset_q7_int32(const  q7_t *src, const int8_t offset, q31_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_offset_q15_int8(const  q15_t *src, const int16_t offset, q7_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_offset_q15_int16(const  q15_t *src, const int16_t offset, q15_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_offset_q15_int32(const  q15_t *src, const int16_t offset, q31_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_offset_q31_int8(const  q31_t *src, const int32_t offset, q7_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_offset_q31_int16(const  q31_t *src, const int32_t offset, q15_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_offset_q31_int32(const  q31_t *src, const int32_t offset, q31_t *dst, uint32_t size, uint32_t shift);

int32_t nnblas_max_q7(const q7_t *src, q31_t *dst, uint32_t size);

int32_t nnblas_max_q15(const q15_t *src, q31_t *dst, uint32_t size);

int32_t nnblas_max_q31(const q31_t *src, q31_t * dst, uint32_t size);

int32_t nnblas_min_q7(const q7_t *src, q31_t *dst, uint32_t size);

int32_t nnblas_min_q15(const q15_t *src, q31_t *dst, uint32_t size);

int32_t nnblas_min_q31(const q31_t *src, q31_t * dst, uint32_t size);

#endif

