/*
 * luna.h
 *
 *  Created on: Aug 18, 2017
 *      Author: dwwang
 */

#ifndef __LUNA_LUNA_H__
#define __LUNA_LUNA_H__

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "luna_error.h"
#include "luna_bits.h"
#include "luna_math_types.h"
#include "luna_controller_math.h"

#define LUNA_VER_MAJOR 3
#define LUNA_VER_MINOR 0
#define LUNA_VER_PATCH 1
#define LUNA_VER_BUILD 0
#define LUNA_VERSION   ( (LUNA_VER_MAJOR << 24) + (LUNA_VER_MINOR << 16) +  (LUNA_VER_PATCH << 8)  +  (LUNA_VER_BUILD << 0) )

#define LUNA_SHARE_MEM_AHB_BASE			(0x48000000)
#define LUNA_SHARE_MEM_AHB_BASE_MASK	(0x480FFFFF)
#define LUNA_SHARE_MEM_BASE				(0x5FE00000)
#define LUNA_PSRAM_MEM_BASE             (0x60000000)

#define _FAST_FUNC_RO           __attribute__ ((section (".sharedmem.text"))) 	//code in shared memory
#define _FAST_DATA_VI           __attribute__ ((section (".sharedmem.data"))) 	//initialized data in shared memory
#define _FAST_DATA_ZI           __attribute__ ((section (".sharedmem.bss"))) 	//zero initialized data in shared memory

#define USE_BIGOP 			1
#define USE_SHAREMEM_CMD 	1

#define LUNA_LOG printf
#define _STR(x) _VAL(x)
#define _VAL(x) #x

/*******************  luna function  ********************/
unsigned int reg_read(unsigned int addr);
void reg_write(unsigned int addr, unsigned int data);
void luna_print_regs();

void luna_init();
int32_t luna_execute(const uint32_t *api, void* param);

uint32_t luna_version();
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


#if USE_BIGOP
#define luna_execute_cmd(api, param, param_size) bigop_execute_op(api, 0, param, param_size)
#else
#define luna_execute_cmd(api, param, param_size) luna_execute(api, param)
#endif

#endif /* __LUNA_LUNA_H__ */
