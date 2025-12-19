/***************************************************************************
 * luna_filter_math.h                                                        *
 *                                                                         *
 * Copyright (C) 2020 listenai Co.Ltd                   			       *
 * All rights reserved.                                                    *
 ***************************************************************************/
#ifndef __LUNA_FILTER_MATH_H__
#define __LUNA_FILTER_MATH_H__


#include "luna_math_types.h"

int32_t luna_filter_iir(const q31_t *src, const q31_t *coeffs, q31_t *dst, int size, uint32_t shift);

#endif /* __LUNA_FILTER_MATH_H__ */
