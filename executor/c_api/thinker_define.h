/**
 * @file    thinker_define.h
 * @brief   C API of thinker Engine Marco Define @listenai
 *
 * @author  LISTENAI
 * @version 1.0
 * @date    2020/5/11
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

// Stringify macro
#define STR_IMP(x) #x
#define STR(x) STR_IMP(x)

// Version macros
#define THINKER_VERSION_MAJOR 3
#define THINKER_VERSION_MINOR 0
#define THINKER_VERSION_PATCH 2
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

// IO Header structure
typedef struct _thinker_IO_Header_ {
    uint16_t num_input_;    // Number of inputs
    uint16_t num_output_;   // Number of outputs
    uint16_t num_state_;    // Number of states
    uint16_t name_length_;  // Length of names
    uint32_t tensor_offset_; // Offset of tensors
    uint32_t name_offset_;  // Offset of names
} tIOHeader;

// Scalar graph structure
typedef struct _thinker_ScalarGraph_
{
    int32_t  num_input_;     // Number of inputs
    int32_t  num_output_;    // Number of outputs
    int32_t  num_node_;      // Number of nodes
    int32_t  num_scalars_;   // Number of scalars
    int32_t  *inputs_;       // Input indices
    int32_t  *outputs_;      // Output indices
    char     *input_names_;  // Input names
    uint8_t  name_max_len;   // Maximum length of name
    int32_t  reserved;       // Reserved field
    double   *scalars_;      // Scalar values
    int32_t  *node_metas_;   // Node metadata
    int32_t  *nodes_;        // Nodes data
} tScalarGraph;

// Shape inference structure
typedef struct _thinker_ShapeInfer_
{
    tScalarGraph  *graph_;         // Pointer to scalar graph
    tTenDimPair   *tid_pairs_;     // Tensor ID pairs
    tDyAxisInfo   *dynamic_axis_;  // Dynamic axis info
    uint32_t      num_id_pair_;    // Number of ID pairs
    uint32_t      num_dy_axis_;    // Number of dynamic axes
    tMemory       inst_memory_;    // Instance memory
} tShapeInfer;

// DMA structure
typedef struct _thinker_DMA_ {
    uint32_t size_;          // Size of DMA transfer
    tTensor *src_tensors_;   // Source tensors
    tTensor *dst_tensors_;   // Destination tensors
} thinkerDMA;

// DMA list structure
typedef struct _thinker_DMA_list_ {
    uint32_t cout_;         // Current count
    uint32_t total_;        // Total count
    thinkerDMA dma_[320];   // Array of DMA operations
} tDMA_List;

// Hyperparameter structure
typedef struct _thinker_Hypeparam_ {
    int32_t op_index;       // Operation index
    int8_t *cache_dir;      // Cache directory
    int8_t *token_id;       // Token ID
    int8_t *model_data;     // Model data pointer
    int32_t model_size;     // Model size
    int8_t *share_model;    // Shared model pointer
} tHypeparam;

// Layout type enumeration
typedef enum {
    NCHW = 0,     // Channel first format
    NHWC = 1,     // Channel last format
    NC4HW4 = 2,   // Channel packed format
    NC8HW8 = 3,   // Channel packed format
    NHWC4 = 4,    // Channel last with packing
    NC16HW16 = 5, // Channel packed format
    NC4HW4_T = 6  // Transposed channel packed format
} tLayoutType;

#endif  // __THINKER_ENGINE_DEFINE_HPP__