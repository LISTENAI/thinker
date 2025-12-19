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
// #include "log_print.h"


#define LUNA_VER_MAJOR 3
#define LUNA_VER_MINOR 0
#define LUNA_VER_PATCH 1
#define LUNA_VER_BUILD 0
#define LUNA_VERSION   ( (LUNA_VER_MAJOR << 24) + (LUNA_VER_MINOR << 16) +  (LUNA_VER_PATCH << 8)  +  (LUNA_VER_BUILD << 0) )
  
#define LUNA_SHARE_MEM_AHB_BASE			(0x20050000)
#define LUNA_SHARE_MEM_BASE				(0x20050000)
#define LUNA_PSRAM_MEM_BASE             (0x28000000)
#define LUNA_FLASH_MEM_BASE				(0x30000000)

#define LUNA_LOG CLOGD          //TODO:

void luna_init();
int32_t luna_execute(const uint32_t *api, void* param);
int32_t luna_execute_cmd(const uint32_t *api, void* param, uint32_t param_size);
int32_t luna_check_flash_addr(uint32_t addr);

uint32_t luna_version();
void start_counter();
uint32_t get_counter();

#endif /* __LUNA_LUNA_H__ */
