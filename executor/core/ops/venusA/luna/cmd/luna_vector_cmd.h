#ifndef __LUNA_VECTOR_H__
#define __LUNA_VECTOR_H__

#include "luna/luna.h"

typedef struct LunaVectorParams
{
	uint32_t *src1;		// r0
	uint32_t scalar_a; 	// r1
	uint32_t *src2; 	// r2
	uint32_t scalar_b; 	// r3
	uint32_t *dst;		// r4
	struct {
		uint32_t shift : 8;  	// r5[0:7]  mode for vector_cmp
		uint32_t size  : 24;	// r5[8:31]
	};
	struct {
		uint8_t in_dtype;	// r6[0:7]
		uint8_t api_type;   // r6[8:15]
		uint8_t ou_dtype;	// r6[16:23]
		uint8_t psram_or_share;   // r6[24:31],0:share,1:psram
	};
}LunaVectorParams_t;

extern __luna_cmd_attr__ uint32_t luna_api_vector_add_new[];
extern __luna_cmd_attr__ uint32_t luna_api_vector_sum[];
extern __luna_cmd_attr__ uint32_t luna_api_vector_cmd[];
extern __luna_cmd_attr__ uint32_t luna_api_vector_cmp[];
extern __luna_cmd_attr__ uint32_t luna_api_vector_maxmin[];
extern __luna_cmd_attr__ uint32_t luna_api_vector_conj[];
extern __luna_cmd_attr__ uint32_t luna_api_vec_cplx_mul[];
extern __luna_cmd_attr__ uint32_t luna_api_vec_cplx_mul_real[];
extern __luna_cmd_attr__ uint32_t luna_api_vec_cplx_mul_ou_real[];
extern __luna_cmd_attr__ uint32_t luna_api_vec_cplx_modulus[];
extern __luna_cmd_attr__ uint32_t luna_api_vector_div[];
extern __luna_cmd_attr__ uint32_t luna_api_vector_sqrt[];
#endif

