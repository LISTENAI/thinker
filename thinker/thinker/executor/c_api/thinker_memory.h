/**
 * @file	thinker_memory.h
 * @brief	memmory mallc and free of x Engine @listenai
 *
 * @author	LISTENAI
 * @version	1.0
 * @date	2020/5/11
 *
 * @Version Record:
 *    -- v1.0: create 2020/5/11
 * Copyright (C) 2022 listenai Co.Ltd
 * All rights reserved.
 */
#include <stdlib.h>

#include "thinker_define.h"
#include "thinker_type.h"

tStatus tMemoryMalloc(tMemory *memory);
void tMemoryFree(tMemory *memory);
