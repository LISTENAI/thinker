
#ifndef __NNBLAS_DIV_H__
#define __NNBLAS_DIV_H__

#include "luna_math_types.h"

int32_t nnblas_div_q31_int32(const q31_t *src1, uint32_t q_src1, const q31_t *src2, uint32_t q_src2, q31_t *dst, uint32_t q_out, uint32_t size);

#endif
