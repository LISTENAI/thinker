#ifndef __NNBLAS_MISC_MATH_H__
#define __NNBLAS_MISC_MATH_H__

#include "luna_math_types.h"

int32_t nnblas_sigmoid(const q15_t *src, q15_t *dst, uint32_t size);

int32_t nnblas_sigmoid_int8(const q15_t *src, q7_t *dst, uint32_t size);

int32_t nnblas_tanh(const q15_t *src, q15_t *dst, uint32_t size);

int32_t nnblas_tanh_int8(const q15_t *src, q7_t *dst, uint32_t size);

int32_t nnblas_relu_q7_int8(const q7_t *src, q7_t *dst, uint32_t size, uint32_t post_shift);

int32_t nnblas_relu_q7_int16(const q7_t *src, q15_t *dst, uint32_t size, uint32_t post_shift);

int32_t nnblas_relu_q7_int32(const q7_t *src, q31_t *dst, uint32_t size, uint32_t post_shift);

int32_t nnblas_relu_q15_int8(const q15_t *src, q7_t *dst, uint32_t size, uint32_t post_shift);

int32_t nnblas_relu_q15_int16(const q15_t *src, q15_t *dst, uint32_t size, uint32_t post_shift);

int32_t nnblas_relu_q15_int32(const q15_t *src, q31_t *dst, uint32_t size, uint32_t post_shift);

int32_t nnblas_relu_q31_int8(const q31_t *src, q7_t *dst, uint32_t size, uint32_t post_shift);

int32_t nnblas_relu_q31_int16(const q31_t *src, q15_t *dst, uint32_t size, uint32_t post_shift);

int32_t nnblas_relu_q31_int32(const q31_t *src, q31_t *dst, uint32_t size, uint32_t post_shift);

int32_t nnblas_prelu_q7_int8(const q7_t *src, uint32_t slope, q7_t *dst, uint32_t size, uint32_t post_shift);

int32_t nnblas_prelu_q7_int16(const q7_t *src, uint32_t slope, q15_t *dst, uint32_t size, uint32_t post_shift);

int32_t nnblas_prelu_q7_int32(const q7_t *src, uint32_t slope, q31_t *dst, uint32_t size, uint32_t post_shift);

int32_t nnblas_prelu_q15_int8(const q15_t *src, uint32_t slope, q7_t *dst, uint32_t size, uint32_t post_shift);

int32_t nnblas_prelu_q15_int16(const q15_t *src, uint32_t slope, q15_t *dst, uint32_t size, uint32_t post_shift);

int32_t nnblas_prelu_q15_int32(const q15_t *src, uint32_t slope, q31_t *dst, uint32_t size, uint32_t post_shift);

int32_t nnblas_prelu_q31_int8(const q31_t *src, uint32_t slope, q7_t *dst, uint32_t size, uint32_t post_shift);

int32_t nnblas_prelu_q31_int16(const q31_t *src, uint32_t slope, q15_t *dst, uint32_t size, uint32_t post_shift);

int32_t nnblas_prelu_q31_int32(const q31_t *src, uint32_t slope, q31_t *dst, uint32_t size, uint32_t post_shift);

int32_t nnblas_memcpy(void* dst, void* src, int size);

int32_t nnblas_memset(void *dst, q7_t value, uint32_t size);

int32_t nnblas_memset_int16(void *dst, int16_t value, uint32_t size);

int32_t nnblas_memset_int32(void *dst, int32_t value, uint32_t size);

int32_t nnblas_memcpy_psram2sharemem(void* dst, void* src, int size);

#endif
