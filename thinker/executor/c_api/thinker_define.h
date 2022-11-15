/**
 * @file	thinker_define.h
 * @brief	C API of thinker Engine Marco Define @listenai
 *
 * @author	LISTENAI
 * @version	1.0
 * @date	2020/5/11
 *
 * @Version Record:
 *    -- v1.0: create 2020/5/12
 * Copyright (C) 2022 listenai Co.Ltd
 * All rights reserved.
 */
#ifndef __THINKER_ENGINE_DEFINE_HPP__
#define __THINKER_ENGINE_DEFINE_HPP__
#include <stdint.h>

#include "thinker_status.h"
#include "thinker_type.h"

#define STR_IMP(x) #x
#define STR(x) STR_IMP(x)
#define THINKER_VERSION_MAJOR 1
#define THINKER_VERSION_MINOR 0
#define THINKER_VERSION_PATCH 0
#define THINKER_VERSION      \
  STR(THINKER_VERSION_MAJOR) \
  "." STR(THINKER_VERSION_MINOR) "." STR(THINKER_VERSION_PATCH)

#if THINKER_USE_VENUS
#include "core/ops/venus/luna/luna.h"
#define VENUS_VERSION  \
    STR(LUNA_VER_MAJOR)"."STR(LUNA_VER_MINOR)"."STR(LUNA_VER_PATCH)"."STR(LUNA_VER_BUILD)
#else
#define VENUS_VERSION   "0.0.0.0"
#endif

typedef struct _thinker_IO_Header_ {
  uint16_t num_input_;
  uint16_t num_output_;
  uint16_t num_state_;
  uint16_t name_length_;
  uint32_t tensor_offset_;
  uint32_t name_offset_;
} tIOHeader;

typedef struct _thinker_DMA_ {
  uint32_t size_;
  tTensor *src_tensors_;
  tTensor *dst_tensors_;
} thinkerDMA;

typedef struct _thinker_DMA_list_ {
  uint32_t cout_;
  uint32_t total_;
  thinkerDMA dma_[256];
} tDMA_List;

// this struct is add for X(init), if need transeform some param, you can add
// the param in this struct
typedef struct _thinker_Hypeparam_ {
  int32_t op_index;
  int8_t *cache_dir;
  int8_t *token_id;

  int8_t *model_data;
  int32_t model_size;
  int8_t *share_model;
} tHypeparam;
typedef enum {
  NCHW = 0,
  NHWC = 1,
  NC4HW4 = 2,
  NC8HW8 = 3,
  NHWC4 = 4,
  NC16HW16 = 5,
  NC4HW4_T = 6
} tLayoutType;

#endif  // __THINKER_ENGINE_RESOURCE_HPP__
