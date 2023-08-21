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

/*******************  luna function  ********************/
void luna_init();
uint32_t luna_version();
void start_counter();
uint32_t get_counter();

#endif /* __LUNA_LUNA_H__ */
