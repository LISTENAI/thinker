#ifndef _NNBLAS_OP_H_
#define _NNBLAS_OP_H_

#define NNBLAS_SIM(name) name

#include "nnblas_basic_math.h"
#include "nnblas_complex_math.h"
#include "nnblas_matrix_math.h"
#include "nnblas_misc_math.h"
#include "nnblas_cnn_math.h"
#include "nnblas_div.h"

void nnblas_init();
uint32_t nnblas_version();
void start_counter();
uint32_t get_counter();

#if USE_SHAREMEM_CMD
//#define __luna_cmd_attr__		const _FAST_FUNC_RO
#define __luna_cmd_attr__		const __attribute__ ((section (".sharedmem.text."_STR(__LINE__))))
#define __luna_param_attr__
#else
#define __luna_cmd_attr__		const
#define __luna_param_attr__
#endif

#endif
