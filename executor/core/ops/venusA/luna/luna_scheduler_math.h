#ifndef __LUNA_SCHEDULER_H_
#define __LUNA_SCHEDULER_H_

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include "luna/luna.h"
#include "luna/luna_mtq_math.h"

#define SCHEDULER_CHANNEL_NUM       (4)
#define SCHEDULER_ITEM_NUM          (8)
#define SCHEDULER_PARAM_SIZE        (256)
#define SCHEDULER_TIMEOUT_THRESHOLD (0xFFFFFFFF)

int luna_scheduler_init();
void luna_scheduler_deinit(void);

int luna_scheduler_run_dynamic(uint32_t ch, luna_mtq_sq_elem_t * user_sq_addr, luna_mtq_cq_elem_t * user_cq_addr, uint32_t user_sq_length, uint32_t user_cq_length, 
    uint32_t priority_level, uint32_t priority_weight);
int luna_scheduler_run_static(uint32_t ch, luna_mtq_sq_elem_t * user_sq_addr, luna_mtq_cq_elem_t * user_cq_addr, uint32_t user_sq_length, uint32_t user_cq_length, 
    uint32_t priority_level, uint32_t priority_weight);
int luna_scheduler_run_cmd(uint32_t ch, const void* api, void *param);

int luna_scheduler_wait(uint32_t ch);

#endif 
