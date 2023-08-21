/***************************************************************************
 * luna_controller_math.h                                                  *
 *                                                                         *
 * Copyright (C) 2020 listenai Co.Ltd                   			       *
 * All rights reserved.                                                    *
 ***************************************************************************/

#ifndef __LUNA_CONTROLLER_MATH_H__
#define __LUNA_CONTROLLER_MATH_H__
/**
 * @defgroup controller Controller Functions
 */
#include "luna_math_types.h"

typedef void* bigop_handle_t;
typedef void* (*bigop_alloc_func_t)(int32_t type, uint32_t size); //type: 0-code, 1-data
typedef int32_t (*bigop_execute_op_func_t)(const void* api, uint32_t api_size, const void* param, uint32_t param_size);
typedef int32_t (*bigop_execute_code_func_t)(const void* api, uint32_t api_size, const void* param, uint32_t param_size);
extern bigop_alloc_func_t bigop_alloc;
extern bigop_execute_op_func_t bigop_execute_op;
extern bigop_execute_code_func_t bigop_execute_code;

bigop_handle_t bigop_init(void* objmem, uint32_t objmem_size, uint32_t code_size, uint32_t data_size);
int32_t bigop_begin(bigop_handle_t handle);
int32_t bigop_end(bigop_handle_t handle);
int32_t bigop_reset(bigop_handle_t handle);
int32_t bigop_run(bigop_handle_t handle);
int32_t bigop_run_async(bigop_handle_t handle);
int32_t bigop_wait_complete(bigop_handle_t handle);

//void* bigop_alloc_op(uint32_t param_size);
//int32_t bigop_execute_op(void* api, void* param, uint32_t param_size);
int32_t bigop_set_mode(int32_t mode); //mode:0-normal 1-bigop 2-stat
int32_t bigop_stat(uint32_t* objmem_size, uint32_t* code_size, uint32_t* data_size);

int32_t luna_shift_mov_inline(const q31_t *src1, q31_t *dst);

void luna_bigop_test();

#endif // __LUNA_CONTROLLER_MATH_H__
