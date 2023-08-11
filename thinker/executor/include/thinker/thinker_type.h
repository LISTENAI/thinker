/** @file */

// thinker_type.h - THINKER API interface and data structures visible to user
// code.

// Copyright (c) 2021-2022 LISTENAI, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#ifndef __THINKER_TYPE_HPP__
#define __THINKER_TYPE_HPP__ 1
#include <stdint.h>

typedef uint64_t addr_type;

typedef addr_type tModelHandle;
typedef addr_type tExecHandle;

typedef enum __tDType__ {
  DTypeUndefined = 0,
  Float16 = 0x6602,  //'f', 2
  Float32 = 0x6604,  //'f', 4
  Float64 = 0x6608,  //'f', 8
  Int8 = 0x6901,     //'i', 1
  Int16 = 0x6902,    //'i', 2
  Int32 = 0x6904,    //'i', 4
  Int64 = 0x6908,    //'i', 8
  Uint8 = 0x7501,    //'u', 1
  Uint16 = 0x7502,   //'u', 2
  Uint32 = 0x7504,   //'u', 4
  Uint64 = 0x7508,   //'u', 8
  Bool = 0x6201,
  Int4 = 0x6801,
} tDType;
typedef enum __MEM_TYPE__ {
  FLASH = 0,
  PSRAM = 1,
  SHARE_MEM = 2,
  UNCERTAIN = 3,
} MemType;

typedef struct _t_Shape_ {
  uint32_t ndim_;
  uint32_t dims_[7];
} tShape;

typedef struct _t_Device_ {
  uint8_t type_;
  uint8_t id_;
} tDevice;

typedef struct _t_Mem_ {
  uint8_t type_;  // dev_type
  uint8_t mem_type_;
} tMem;

typedef struct _thinker_Data_ {
  void *dptr_;
  uint16_t dev_type_;
  uint16_t dtype_;
  uint16_t zero_;
  float scale_;
  tShape shape_;
} tData;

typedef struct _thinker_Memory_ {
  uint32_t size_;     // size of memory
  uint8_t dev_type_;  // device of memory default {0,0}
  uint8_t mem_type_;  // type of memory: (0, model_inst), (1, exec_inst), (2,
                  // share_memory), (3, runtime), (4, shape), (5, xflow inst)

  addr_type dptr_;  // address of memory
} tMemory;          // 16bytes

typedef struct _t_Memory_List_ {
  uint16_t shared_count_;
  uint16_t total_count_;
  uint32_t elem_size_;
  uint32_t header_size_;
  uint32_t offset_;
} tMemoryList;  // 16bytes

typedef struct _t_IO_ {
  uint32_t tensor_id;
  char name[60];
} tIO;

typedef struct _t_State_ {
  uint32_t input_tensor_id;
  uint32_t output_tensor_id;
} tState;

typedef struct _t_Tensor_ {
  tMem mem_;
  union {
    uint16_t dtype_;
    uint8_t byte_;
  };

  int32_t mem_id_;
  float scale_;
  int32_t zero_;

  tShape shape_;
  union {
    addr_type dptr_;
    addr_type offset_;
  };
  uint8_t layout_;
  uint32_t reserved;
} tTensor;  // 56bytes

typedef struct _t_TensorList_ {
  uint32_t count_;
  uint32_t elem_size_;
  uint32_t header_size_;
  uint32_t offset_;
} tTensorList;

typedef struct _t_Operator_ {
  uint16_t op_id_;
  uint16_t attr_offset_;
  uint16_t tensor_offset_;
  uint16_t scalar_offset_;
  uint16_t total_size_;
  uint16_t num_input_;
  uint16_t num_output_;
  uint8_t num_temp_;
  uint8_t num_scalar_;
} tOperator;  // 16bytes

typedef struct _t_OperatorList_ {
  uint16_t op_count_;
  uint16_t type_count_;
  uint8_t type_length_;
  uint8_t header_size_;
  uint16_t type_offset_;
  uint32_t op_offset_;
  uint32_t op_size_;
} tOperatorList;  // 16bytes

typedef struct _t_Parameter_ {
  tMem memory_;
  uint16_t mem_id_;
  uint32_t offset_;
  uint64_t size_;
} tParameter;  // 16bytes

typedef struct _t_ParameterList_ {
  uint32_t count_;
  uint32_t elem_size_;
  uint32_t header_size_;
  uint32_t offset_;
} tParameterList;

typedef struct _t_Model_Header_ {
  uint8_t label_[16];
  uint32_t crc32_;
  uint32_t memory_offset_;  // the first offset of device memory header
  uint32_t tensor_offset_;  // the first offset of tensor header
  uint32_t scalar_offset_;  // the first offset of tensor header
  uint32_t op_offset_;      // the first offset of op name list
  uint32_t io_offset_;      // input & output tensor id offset
  uint32_t state_offset_;   // state offset
  uint32_t debug_offset_;
  uint32_t shape_infer_offset_;  // shape inference offset
  uint32_t param_offset_;        // param offset
  uint32_t dma_offset_;          // dma offset
  uint32_t reserved;
  uint64_t total_size_;  // param size

} tModelHeader;  // 64bytes

typedef struct _t_DMA_ {
  tDevice src_device_;
  tDevice dst_device_;
  uint16_t src_tensor_id_;
  uint16_t dst_tensor_id_;
  uint64_t size_;
} tDMA;  // 16bytes

typedef struct _t_DMAList_ {
  uint32_t count_;
  uint32_t elem_size_;
  uint32_t header_size_;
  uint32_t offset_;
} tDMAList;

typedef struct _t_Tensor_Name_ {
  char name_[64];
} tTensorName;

typedef struct _t_DebugList_ {
  uint32_t tensor_name_count_;
  tTensorName *tensor_name_list_;
  uint32_t offset_;
} tDebugList;

#endif  // __THINKER_TYPE_HPP__
