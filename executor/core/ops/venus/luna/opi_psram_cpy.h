#ifndef __OPI_PSRAM_CPY_H__
#define __OPI_PSRAM_CPY_H__

#include "luna_math.h"

typedef void (*dma_sync_call_func)(void* param);

void dma_init();
void dma_wait_complete(int chn);
void dma_cpy_async(int chn, void *dst, void *src, int32_t size);

void opi_psram_cpy_out(void *dst, void *src, int32_t size);
void opi_psram_cpy_in(void *dst, void *src, int32_t size);

void opi_psram_cpy_out_pro(void *dst, void *src, int32_t size, dma_sync_call_func func, void* param);
void opi_psram_cpy_in_pro(void *dst,  void *src, int32_t size, dma_sync_call_func func, void* param);

#endif //__OPI_PSRAM_CPY_H__
