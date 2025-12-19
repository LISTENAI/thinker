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
#include "luna_math_types.h"
#if !(defined(WIN32) || defined(linux))
#include "log_print.h"
#endif

#define LUNA_VER_MAJOR 3
#define LUNA_VER_MINOR 0
#define LUNA_VER_PATCH 1
#define LUNA_VER_BUILD 0
#define LUNA_VERSION   ( (LUNA_VER_MAJOR << 24) + (LUNA_VER_MINOR << 16) +  (LUNA_VER_PATCH << 8)  +  (LUNA_VER_BUILD << 0) )
  
#define LUNA_SHARE_MEM_AHB_BASE			(0x20020000)
#define LUNA_SHARE_MEM_BASE				(0x20020000)
#define LUNA_PSRAM_MEM_BASE             (0x38000000)
#define LUNA_FLASH_MEM_BASE				(0x30000000)
#define LUNA_SHARE_MEM_SIZE				(384*1024)

#if !(defined(WIN32) || defined(linux))
#define LUNA_LOG CLOGD          //TODO:
#endif

#define _FAST_FUNC_RO       __attribute__ ((section (".sharedmem.text")))
#define _FAST_DATA_VI       __attribute__ ((section (".sharedmem.data")))       
#define _FAST_DATA_ZI       __attribute__ ((section (".sharedmem.bss")))

#define LUNA_SHARE_ADDR_OFFSET(addr)	((uint32_t)((uint32_t)(addr)-LUNA_SHARE_MEM_BASE))
#define LUNA_PSRAM_ADDR_OFFSET(addr)	((uint32_t)((uint32_t)(addr)-LUNA_PSRAM_MEM_BASE))
#define LUNA_FLASH_ADDR_OFFSET(addr)    ((uint32_t)((uint32_t)(addr)-LUNA_FLASH_MEM_BASE))

#define __luna_cmd_attr__	_FAST_FUNC_RO

void luna_init();
int32_t luna_execute_cmd(const uint32_t *api, void* param, uint32_t param_size);
int32_t luna_execute(const uint32_t *api, void* param);

uint32_t luna_version();
void start_counter();
uint32_t get_counter();

// luna execute cmd hook function type.
typedef int32_t (*luna_hook_func_t)(const uint32_t *api, void* param, uint32_t param_size, void* user_data);
// register execute cmd hook function.
void luna_register_hook(luna_hook_func_t func, void *userdata);

#endif /* __LUNA_LUNA_H__ */
