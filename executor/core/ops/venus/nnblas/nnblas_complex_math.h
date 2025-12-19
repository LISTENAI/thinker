#ifndef __NNBLAS_COMPLEX_MATH_H__
#define __NNBLAS_COMPLEX_MATH_H__

#include "luna_math_types.h"

int32_t nnblas_power_spectrum_q31_int32(const q31_t *src, q31_t *dst, uint32_t size, uint32_t shift);

#endif
