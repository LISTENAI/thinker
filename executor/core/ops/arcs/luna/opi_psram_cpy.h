#ifndef __DMA_CPY_H__
#define __DMA_CPY_H__

#include <stdint.h>
#include <stdbool.h>

// By default DMA channel 0 is dedicated to copy algorithm data.
// If other DMA channel is used, opi_psram_cpy_out & opi_psram_cpy_in CANNOT work!
#define ALG_DMA_CH      0

//void dma_init();
bool dma_init(int chn); //[CHANGED]: specify DMA channel
void dma_wait_complete(int chn);
void dma_cpy_async(int chn, void *dst, void *src, int32_t size);

void opi_psram_cpy_out(void *dst, void *src, int32_t size); // use ALG_DMA_CH
void opi_psram_cpy_in(void *dst, void *src, int32_t size); // use ALG_DMA_CH

void dma_uninit(int chn); //[NEW]: to release used DMA channel

//typedef void (*dma_sync_call_func)(void* param);
//void opi_psram_cpy_out_pro(void *dst, void *src, int32_t size, dma_sync_call_func func, void* param);
//void opi_psram_cpy_in_pro(void *dst,  void *src, int32_t size, dma_sync_call_func func, void* param);

#endif //__DMA_CPY_H__