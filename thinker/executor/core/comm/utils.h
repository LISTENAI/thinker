#ifndef _THINKER_EXEC_CORE_CPU_ARM_UTILS_H_
#define _THINKER_EXEC_CORE_CPU_ARM_UTILS_H_

#include <fenv.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "c_api/thinker_define.h"
#include "thinker_define.h"
#include "thinker_memory.h"
#include "thinker_type.h"

#define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC 0

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) (a) < (b) ? (a) : (b)
#define SUM(a, b) ((a) + (b))

#define MAX_BITS(bits) ((1LL << (bits - 1)) - 1)
#define MIN_BITS(bits) (-(1LL << (bits - 1)))
#define SATURATE(x, bits)                \
  ((x) > MAX_BITS(bits) ? MAX_BITS(bits) \
                        : ((x) < MIN_BITS(bits) ? MIN_BITS(bits) : (x)))
#define SATURATE_8BITS(x) ((x) > 127 ? 127 : ((x) < -128 ? -128 : (x)))
#define SATURATE_16BITS(x) ((x) > 32768 ? 32768 : ((x) < -32768 ? -32768 : (x)))
#define SATURATE_U8BITS(x) ((x) > 255 ? 255 : ((x) < 0 ? 0 : (x)))
#define SATURATE_32BITS(x) \
  ((x) > 2147483647 ? 2147483647 : ((x) < -2147483648 ? -2147483648 : (x)))

#define _FE_ROUND FE_TONEAREST

bool equalShape(tShape *src, tShape *dst);

size_t getShapeSize(tShape *shape);

size_t getTensorSize(const tTensor *tensor);

tShape calcStride(const tShape *shape);

void quant(float *src, int8_t *dst, int32_t size, int8_t scale);

void dequant8bit(int8_t *src, float *dst, int32_t size, int8_t scale);

void dequantU8bit(uint8_t *src, float *dst, int32_t size, int8_t scale);

void dequant32bit(int32_t *src, float *dst, int32_t size, int8_t scale);

#ifdef THINKER_USE_VENUS
void lunaDmaInit(void);

void getWeightData(tDMA_List *dma_list, int32_t channel);
#endif  // THINKER_USE_VENUS

#endif  // _THINKER_EXEC_CORE_CPU_ARM_UTILS_H_
