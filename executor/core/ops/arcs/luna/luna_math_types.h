/*
 * luna_math_types.h
 *
 *  Created on: 2020Äê9ÔÂ4ÈÕ
 *      Author: dwwang3
 */

#ifndef __LUNA_MATH_TYPES_H__
#define __LUNA_MATH_TYPES_H__

#include <stdint.h>


typedef int8_t 		q7_t;
typedef int16_t 	q15_t;
typedef int32_t 	q31_t;
typedef int64_t 	q63_t;
typedef float 		float32_t;
typedef double		float64_t;

typedef int8_t 		int4_t;

#ifndef bool
#define bool int8_t
#endif

#endif // __LUNA_MATH_TYPES_H__

