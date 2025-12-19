#ifndef _THINKER_UTILS_H_
#define _THINKER_UTILS_H_

#include <fenv.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "c_api/thinker_define.h"
#include "thinker_define.h"
#include "thinker_type.h"

#if THINKER_USE_MTQ
#include "luna/luna_mtq_math.h"
#endif

#if !(defined(WIN32) || defined(linux))
#include "core_feature_cache.h"
#endif

#define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC 0

// Macro definitions for common operations
#define MAX(a, b) ((a) > (b) ? (a) : (b))           // Maximum of two values
#define MIN(a, b) (a) < (b) ? (a) : (b)             // Minimum of two values
#define SUM(a, b) ((a) + (b))                       // Sum of two values

// Bit manipulation macros for saturation
#define MAX_BITS(bits) ((1LL << (bits - 1)) - 1)    // Maximum signed value for given bits
#define MIN_BITS(bits) (-(1LL << (bits - 1)))       // Minimum signed value for given bits
#define SATURATE(x, bits) ((x) > MAX_BITS(bits) ? MAX_BITS(bits) \
                         : ((x) < MIN_BITS(bits) ? MIN_BITS(bits) : (x)))  // Saturate value to bit range
#define SATURATE_8BITS(x) ((x) > 127 ? 127 : ((x) < -128 ? -128 : (x)))  // 8-bit saturation
#define SATURATE_16BITS(x) ((x) > 32768 ? 32768 : ((x) < -32768 ? -32768 : (x)))  // 16-bit saturation
#define SATURATE_U8BITS(x) ((x) > 255 ? 255 : ((x) < 0 ? 0 : (x)))  // Unsigned 8-bit saturation
#define SATURATE_32BITS(x) ((x) > 2147483647 ? 2147483647 \
                         : ((x) < -2147483648 ? -2147483648 : (x)))  // 32-bit saturation

#define _FE_ROUND FE_TONEAREST                     // Floating point rounding mode

// Function declarations for shape operations
bool equalShape(tShape *src, tShape *dst);         // Compare two shapes for equality
size_t getShapeSize(tShape *shape);                // Calculate total elements in a shape
size_t getTensorSize(const tTensor *tensor);       // Calculate total elements in a tensor considering layout

tShape calcStride(const tShape *shape);            // Calculate stride for a given shape

// Quantization functions
void quant(float *src, int8_t *dst, int32_t size, int8_t scale);  // Quantize floats to int8
void dequant8bit(int8_t *src, float *dst, int32_t size, int8_t scale);  // Dequantize int8 to floats
void dequantU8bit(uint8_t *src, float *dst, int32_t size, int8_t scale);  // Dequantize uint8 to floats
void dequant32bit(int32_t *src, float *dst, int32_t size, int8_t scale);  // Dequantize int32 to floats

// 4-bit conversion functions
void convert_4bitto8bit(int8_t *dst, int8_t *src, int32_t size);  // Convert 4-bit to 8-bit with sign extension
void convert_4bitto32bit(int32_t *dst, int8_t *src, int32_t size);  // Convert 4-bit to 32-bit with sign extension

// Venus-specific functions
#ifdef THINKER_USE_VENUS
void lunaDmaInit(void);                            // Initialize Luna DMA
void getWeightData(tDMA_List *dma_list, int32_t channel);  // Get weight data via DMA
#endif

// Arcs-specific functions
#ifdef THINKER_USE_ARCS
#include "ops/arcs/luna/opi_psram_cpy.h"
void getWeightData(tDMA_List *dma_list, int32_t channel);  // Get weight data via DMA
void cpu_memcpy(void *dst, const void *src, size_t size);  // CPU memory copy function
#endif

// VenusA-specific functions
#ifdef THINKER_USE_VENUSA
#include "../ops/venusA/luna/include/cache.h"
#include "../ops/venusA/luna/luna_misc_math.h"
void dma_wait_complete(int chn);                   // Wait for DMA completion
void dma_cpy_async(int chn, void *dst, void *src, int32_t size);  // Asynchronous DMA copy
void opi_psram_cpy_out(void *dst, void *src, int32_t size);  // PSRAM copy out
void getWeightData(tDMA_List *dma_list, int32_t channel);  // Get weight data via DMA
void cpu_memcpy(void *dst, const void *src, size_t size);  // CPU memory copy function

#if THINKER_USE_MTQ
// MTQ execution hooks
int32_t luna_execute_cmd_hook_for_get_list_length(const uint32_t *api, void* param, uint32_t param_size, void* userdata);  // Hook for getting list length
int32_t luna_execute_cmd_hook_for_build_list(const uint32_t *api, void* param, uint32_t param_size, void* userdata);  // Hook for building list
#endif
#endif

#endif  // _THINKER_EXEC_CORE_CPU_ARM_UTILS_H_

// Timer functions for profiling
#if THINKER_PROFILE
#if defined(WIN32) || defined(linux)
double tick_count(void);                           // High-resolution timer for Windows/Linux
#else
uint64_t tick_count(void);                         // High-resolution timer for embedded systems
#endif
#endif

// MTQ external variables
#if THINKER_USE_MTQ
extern luna_mtq_sq_elem_t *g_sq_addr_user_ch;      // SQ element address for MTQ
extern luna_mtq_cq_elem_t *g_cq_addr_user_ch;      // CQ element address for MTQ
extern uint32_t g_submit_pos;                      // Current submission position
extern uint32_t g_param_size;                      // Parameter size counter
extern int8_t* g_param_addr;                       // Parameter address
#endif