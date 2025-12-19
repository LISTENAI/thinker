#include "thinker_api.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_init.h"
#include "core/operator_register.h"
#include "thinker_crc.h"
#include "thinker_debug.h"
#include "thinker_define.h"
#include "thinker_type.h"
#include "core/shape_infer/shape_infer.h"

#if THINKER_USE_VENUS
#include "ops/venus/luna/opi_psram_cpy.h"
#endif

#if THINKER_USE_MTQ
#include "luna/luna_mtq_math.h"
#endif

#ifndef NULL
#define NULL 0
#endif

#define THINKER_INST_FLAG 0x20201201

// Model structure definition
typedef struct _t_Model_ {
  uint32_t flag_;

  tMemory inst_memory_;

  uint32_t num_tensor_;
  tTensor *tensor_;

  uint16_t num_shared_memory_;
  uint16_t num_memory_;
  tMemory *memory_;

  uint16_t num_operator_;
  tOperatorAPI **op_api_;
  uint8_t *op_buffer_;

  uint16_t num_input_;
  uint16_t num_output_;
  int32_t *io_tensors_;
  char *io_names_;
  int32_t io_name_len_;
  tShape *input_shape_;

  uint16_t num_state_;
  tState *states_;
  tShapeInfer   *shape_infer;
  tDebugList *debug_info;
  tDMAList *dma_info_;
  tDMA *dma_;
} tModel;

// Execution instance structure definition
typedef struct _t_Instance_ {
  uint32_t flag_;

  tMemory inst_memory_;
  tTensor *tensor_;
  tMemory *memory_;
  tModel *model_;
  double *shape_scalars_;
  tDMA_List *dma_list_;
  uint32_t force_stop_flag;
  int32_t reserved_args[8];
} tExecInst;

/**
 * Get version string based on index
 * @param index Version index (1 for Venus, otherwise Thinker)
 * @return Version string pointer
 */
const char *tGetVersion(const int8_t index){
   return (1 == index) ? VENUS_VERSION : THINKER_VERSION;
}

/**
 * Initialize the system
 * @return Status code
 */
tStatus tInitialize() {
  init_ops_list();
  return T_SUCCESS;
}

/**
 * Uninitialize the system
 * @return Status code
 */
tStatus tUninitialize() {
	return T_SUCCESS;
}

/**
 * Calculate memory requirements for model execution
 * @param memory_list Output array of memory requirements
 * @param num_memory Pointer to store number of memory entries
 * @param res Pointer to model resource data
 * @param size Size of model resource data
 * @return Status code
 */
tStatus tGetMemoryPlan(tMemory *memory_list, int32_t *num_memory,
                       const int8_t *res, const uint64_t size) {
  tModelHeader *res_hdr = (tModelHeader *)res;

  if (res_hdr->total_size_ > size) {
    return T_ERR_RES_INCOMPLETE;
  }

  // check CRC
#ifdef THINKER_CHECK_CRC
 if (res_hdr->crc32_ != 0) {
   uint8_t *res_model_ptr = (uint8_t *)res + ALIGN16(sizeof(tModelHeader));
   int32_t res_model_size = size - ALIGN16(sizeof(tModelHeader));
   int32_t crc_check = crc24_calc(0, res_model_ptr, res_model_size);
   if (res_hdr->crc32_ != crc_check) {
     printf("%d,%d", res_hdr->crc32_, crc_check);
     return T_ERR_RES_CRC_CHECK;
   }
 }
#endif

#if THINKER_CHECK_PLATFORM
#if THINKER_USE_VENUS
 if (res_hdr->reserved != 0) {
  printf("Incompatible resource for venus platform!\n");
  printf("Please choose the correct platform!\n");
  return T_ERR_INVALID_PLATFROM;
 }
#elif THINKER_USE_MARS
 if (res_hdr->reserved != 1) {
  printf("Incompatible resource for mars platform!\n");
  printf("Please choose the correct platform!\n");
  return T_ERR_INVALID_PLATFROM;
 }
#elif THINKER_USE_ARCS
 if (res_hdr->reserved != 2) {
  printf("Incompatible resource for arcs platform!\n");
  printf("Please choose the correct platform!\n");
  return T_ERR_INVALID_PLATFROM;
 }
#elif THINKER_USE_VENUSA
 if (res_hdr->reserved != 3) {
  printf("Incompatible resource for venusA platform!\n");
  printf("Please choose the correct platform!\n");
  return T_ERR_INVALID_PLATFROM;
 }
#endif
#endif

  // Calculate model instance size
  int32_t model_inst_size = 0;
  model_inst_size += ALIGN16(sizeof(tModel));

  tMemoryList mem_hdr = *(tMemoryList *)(res + res_hdr->memory_offset_);
  tTensorList tensor_hdr = *(tTensorList *)(res + res_hdr->tensor_offset_);
  tIOHeader io_hdr = *(tIOHeader *)(res + res_hdr->io_offset_);
  tOperatorList op_hdr = *(tOperatorList *)(res + res_hdr->op_offset_);
  tParameterList param_hdr = *(tParameterList *)(res + res_hdr->param_offset_);
  tDebugList debug_hdr = *(tDebugList *)(res + res_hdr->debug_offset_);
  tDMAList dma_hdr = *(tDMAList *)(res + res_hdr->dma_offset_);

  model_inst_size += ALIGN16(mem_hdr.total_count_ * sizeof(tMemory));
  model_inst_size += ALIGN16(tensor_hdr.count_ * sizeof(tTensor));
  model_inst_size += ALIGN16((io_hdr.num_input_ + io_hdr.num_output_) * sizeof(int32_t));
  model_inst_size += ALIGN16((io_hdr.num_input_ + io_hdr.num_output_) * io_hdr.name_length_);
  model_inst_size += ALIGN16(io_hdr.num_input_ * sizeof(tShape));
  model_inst_size += ALIGN16(op_hdr.type_count_ * sizeof(tOperatorAPI));
  model_inst_size += ALIGN16(op_hdr.op_size_);
  model_inst_size += ALIGN16(sizeof(tShapeInfer));
  model_inst_size += ALIGN16(sizeof(tDebugList));
  model_inst_size += ALIGN16(sizeof(tDMAList));
  model_inst_size += ALIGN16(dma_hdr.count_ * sizeof(tDMA));

  // Add model instance memory
  int32_t num = 0;
  tMemory model_inst_memory;
  model_inst_memory.size_ = model_inst_size;
  model_inst_memory.dptr_ = 0;
  model_inst_memory.dev_type_ = 1;
  model_inst_memory.mem_type_ = 0;
  memory_list[num] = model_inst_memory;
  num += 1;

  // Add parameter memory
  tMemory *shared_memory = (tMemory *)(res + res_hdr->memory_offset_ + mem_hdr.offset_);

  for (int32_t i = 0; i < mem_hdr.shared_count_; ++i) {
    tParameter *params = (tParameter *)((uint8_t *)res + res_hdr->param_offset_ + param_hdr.header_size_);

    memory_list[num] = shared_memory[i];
    memory_list[num].dptr_ = (addr_type)((int8_t *)params + params->offset_);
    memory_list[num].mem_type_ = 2;
    num += 1;
  }

  // Add runtime memory
  for (int32_t i = mem_hdr.shared_count_; i < mem_hdr.total_count_; ++i) {
    memory_list[num] = shared_memory[i];
    memory_list[num].mem_type_ = 3;
    num += 1;
  }

  // Calculate execution instance size
  tShapeInferHdr *shape_hdr = (tShapeInferHdr *)((char *)res + res_hdr->shape_infer_offset_);
  tScalarGraph *scalar_graph = (tScalarGraph *)((char*)shape_hdr + shape_hdr->graph_offset_);

  int32_t inst_size = 0;
  inst_size += ALIGN16(sizeof(tExecInst));
  inst_size += ALIGN16(mem_hdr.total_count_ * sizeof(tMemory));
  inst_size += ALIGN16(tensor_hdr.count_ * sizeof(tTensor));

  inst_size += ALIGN16(scalar_graph->num_scalars_*sizeof(double)); 

  inst_size += ALIGN16(sizeof(tDMA_List));
  tMemory inst_memory;
  inst_memory.size_ = inst_size;
  inst_memory.dptr_ = 0;
  inst_memory.dev_type_ = 1;
  inst_memory.mem_type_ = 1;
  memory_list[num] = inst_memory;
  num += 1;

  // Calculate shape inference workspace size
  uint64_t total_size    = ALIGN16(sizeof(tScalarGraph));
  total_size             += ALIGN16(shape_hdr->graph_size_);
  total_size             += ALIGN16(shape_hdr->num_id_pair_ * sizeof(tTenDimPair));
  total_size             += ALIGN16(shape_hdr->num_dy_axis_ * sizeof(tDyAxisInfo));
  tMemory tShapeInfer_memory = { total_size, 1, 4, 0};
  memory_list[num] = tShapeInfer_memory;
  num += 1;

  num_memory[0] = num;
  return T_SUCCESS;
}

/**
 * Initialize model from resource data
 * @param hdl Output handle to model
 * @param res Pointer to model resource data
 * @param size Size of model resource data
 * @param memory_list Array of memory allocations
 * @param num_memory Number of memory entries
 * @return Status code
 */
#if !(defined(WIN32) || defined(linux))
#pragma clang optimize off
#endif
tStatus tModelInit(tModelHandle *hdl, const int8_t *res, const uint64_t size,
                   const tMemory *memory_list, const int32_t num_memory) {
  tModelHeader *res_hdr = (tModelHeader *)res;
  int8_t *cpu_memory = NULL;
  int32_t i, j;
  if (res == NULL || size == 0) {
    return T_ERR_RES_MISSING;
  }

  if (res_hdr->total_size_ > size) {
    return T_ERR_RES_INCOMPLETE;
  }
  // CRC check
#ifdef THINKER_CHECK_CRC
  if (res_hdr->crc32_ != 0) {
    uint8_t *res_model_ptr = (uint8_t *)res + ALIGN16(sizeof(tModelHeader));
    int32_t res_model_size = size - ALIGN16(sizeof(tModelHeader));
    int32_t crc_check = crc24_calc(0, res_model_ptr, res_model_size);
    if (res_hdr->crc32_ != crc_check) {
      return T_ERR_RES_CRC_CHECK;
    }
  }
#endif

  int32_t inst_size = 0;
  inst_size += ALIGN16(sizeof(tModel));
  tMemoryList mem_hdr = *(tMemoryList *)(res + res_hdr->memory_offset_);
  tTensorList tensor_hdr = *(tTensorList *)(res + res_hdr->tensor_offset_);
  tIOHeader io_hdr = *(tIOHeader *)(res + res_hdr->io_offset_);
  tOperatorList op_hdr = *(tOperatorList *)(res + res_hdr->op_offset_);
  tParameterList param_hdr = *(tParameterList *)(res + res_hdr->param_offset_);
  tDebugList debug_hdr = *(tDebugList *)(res + res_hdr->debug_offset_);
  tDMAList dma_hdr = *(tDMAList *)(res + res_hdr->dma_offset_);

  inst_size += ALIGN16(mem_hdr.total_count_ * sizeof(tMemory));
  inst_size += ALIGN16(tensor_hdr.count_ * sizeof(tTensor));
  inst_size +=
      ALIGN16((io_hdr.num_input_ + io_hdr.num_output_) * sizeof(int32_t));
  inst_size +=
      ALIGN16((io_hdr.num_input_ + io_hdr.num_output_) * io_hdr.name_length_);
  inst_size += ALIGN16(io_hdr.num_input_ * sizeof(tShape));
  inst_size += ALIGN16(op_hdr.type_count_ * sizeof(tOperatorAPI));
  inst_size += ALIGN16(op_hdr.op_size_);
  inst_size += ALIGN16(sizeof(tShapeInfer));
  inst_size += ALIGN16(sizeof(tDebugList));
  inst_size += ALIGN16(sizeof(tDMAList));
  inst_size += ALIGN16(dma_hdr.count_ * sizeof(tDMA));

  tMemory inst_memory;
  inst_memory.size_ = inst_size;
  inst_memory.dptr_ = 0;
  for (i = 0; i < num_memory; i++) {
    if (memory_list[i].mem_type_ == 0) {
      assert(inst_memory.size_ <= memory_list[i].size_);
      inst_memory = memory_list[i];
      break;
    }
  }

  int8_t *ptr = (int8_t *)(inst_memory.dptr_);
  tModel *inst = (tModel *)ptr;
  inst->flag_ = THINKER_INST_FLAG;
  inst->inst_memory_ = inst_memory;
  ptr += ALIGN16(sizeof(tModel));
  inst->num_memory_ = mem_hdr.total_count_;
  inst->num_shared_memory_ = mem_hdr.shared_count_;
  assert(mem_hdr.header_size_ == sizeof(tMemory));
  inst->memory_ = (tMemory *)ptr;
  memcpy(inst->memory_, res + res_hdr->memory_offset_ + mem_hdr.offset_,
         inst->num_memory_ * sizeof(tMemory));

  j = 0;
  for (i = 0; i < inst->num_shared_memory_; ++i) {
    while (j < num_memory) {
      if (memory_list[j].mem_type_ == 2) {
        inst->memory_[i] = memory_list[j];
        break;
      }
      j++;
    }
  }

  ptr += ALIGN16(inst->num_memory_ * sizeof(tMemory));
  inst->tensor_ = (tTensor *)ptr;
  inst->num_tensor_ = tensor_hdr.count_;
  assert(tensor_hdr.elem_size_ == sizeof(tTensor));
  memcpy(inst->tensor_, res + res_hdr->tensor_offset_ + tensor_hdr.offset_,
         inst->num_tensor_ * sizeof(tTensor));
  ptr += ALIGN16(inst->num_tensor_ * sizeof(tTensor));

  inst->num_input_ = io_hdr.num_input_;
  inst->num_output_ = io_hdr.num_output_;
  inst->io_tensors_ = (int32_t *)ptr;
  memcpy(inst->io_tensors_, res + res_hdr->io_offset_ + io_hdr.tensor_offset_,
         (inst->num_input_ + inst->num_output_) * sizeof(int32_t));
  ptr += ALIGN16((inst->num_input_ + inst->num_output_) * sizeof(int32_t));

  inst->io_names_ = (char *)ptr;
  inst->io_name_len_ = io_hdr.name_length_;
  memcpy(inst->io_names_, res + res_hdr->io_offset_ + io_hdr.name_offset_,
         (inst->num_input_ + inst->num_output_) * io_hdr.name_length_);
  ptr += ALIGN16((inst->num_input_ + inst->num_output_) * io_hdr.name_length_);

  inst->input_shape_ = (tShape *)ptr;
  for (i = 0; i < inst->num_input_; ++i) {
    uint32_t tensor_id = inst->io_tensors_[i];
    inst->input_shape_[i] = inst->tensor_[tensor_id].shape_;
  }
  ptr += ALIGN16(inst->num_input_ * sizeof(tShape));

  inst->num_operator_ = op_hdr.op_count_;
  inst->op_api_ = (tOperatorAPI **)ptr;
  const char *type_ptr =
      (const char *)res + res_hdr->op_offset_ + op_hdr.type_offset_;
  for (i = 0; i < op_hdr.type_count_; ++i) {
    tOperatorAPI *op_api = GetOperatorAPI(type_ptr);
    if (op_api == NULL) {
      printf("init model not support op : %s, register op count:%d\n", type_ptr,
             GetOperatorCount());
      return T_ERR_NO_SUPPORT_OP;
    }
    inst->op_api_[i] = op_api;
    type_ptr += op_hdr.type_length_;
  }
  ptr += ALIGN16(op_hdr.type_count_ * sizeof(tOperatorAPI));

  inst->op_buffer_ = (uint8_t *)ptr;
  memcpy(inst->op_buffer_, res + res_hdr->op_offset_ + op_hdr.op_offset_,
         op_hdr.op_size_);
  ptr += ALIGN16(op_hdr.op_size_);

  for (i = 0; i < inst->num_tensor_; ++i) {
    tTensor *tensor = inst->tensor_ + i;
    tMemory *memory = &inst->memory_[tensor->mem_id_];
    if (tensor->mem_id_ >= inst->num_shared_memory_) {
      continue;
    }
    uint64_t offset = tensor->offset_;
    tensor->dptr_ = memory->dptr_ + offset;
  }


  inst->shape_infer = (tShapeInfer *)ptr;

  for(i = 0; i < num_memory; i++){
    if(memory_list[i].mem_type_ == 4) {
      inst->shape_infer->inst_memory_ = memory_list[i];
    }
  }
  int32_t ret = tShapeInferInit((char *)res + res_hdr->shape_infer_offset_, inst->shape_infer);
  ptr += ALIGN16(sizeof(tShapeInfer));


  inst->debug_info = (tDebugList *)ptr;
  inst->debug_info->tensor_name_count_ = debug_hdr.tensor_name_count_;
  inst->debug_info->tensor_name_list_ =
      (void *)(res + res_hdr->debug_offset_ + sizeof(tDebugList));
  ptr += ALIGN16(sizeof(tDebugList));

  inst->dma_info_ = (tDMAList *)ptr;
  inst->dma_info_->count_ = dma_hdr.count_;
  inst->dma_info_->elem_size_ = dma_hdr.elem_size_;
  inst->dma_info_->header_size_ = dma_hdr.header_size_;
  inst->dma_info_->offset_ = dma_hdr.offset_;
  ptr += ALIGN16(sizeof(tDMAList));

  inst->dma_ = (tDMA *)ptr;
  memcpy(inst->dma_, res + res_hdr->dma_offset_ + dma_hdr.offset_,
         dma_hdr.count_ * sizeof(tDMA));
  *hdl = ~((tModelHandle)inst);
  return T_SUCCESS;
}
#if !(defined(WIN32) || defined(linux))
#pragma clang optimize on
#endif

/**
 * Finalize model and free resources
 * @param hdl Model handle to finalize
 * @return Status code
 */
tStatus tModelFini(tModelHandle hdl) {
#if !THINKER_USE_ACL
  tModel *model = (tModel *)~hdl;
  if (model == NULL || model->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }

  return T_SUCCESS;
#endif
}

/**
 * Get number of input tensors
 * @param hdl Model handle
 * @return Number of input tensors
 */
int32_t tGetInputCount(const tModelHandle hdl) {
  tModel *model = (tModel *)~hdl;
  return model->num_input_;
}

/**
 * Get input information by index
 * @param hdl Execution handle
 * @param idx Input index
 * @param info Output input information
 * @return Status code
 */
tStatus tGetInputInfo(const tExecHandle hdl, const int32_t idx,
                  tData *input) {
  tExecInst *inst = (tExecInst *)~hdl;
  tModel *model = inst->model_;
  if (inst == NULL || inst->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }

  if (idx < 0 || idx >= model->num_input_) {
    return T_ERR_INDEX_OF_BOUND;
  }

  if (input == NULL) {
    return T_ERR_INVALID_DATA;
  }

  {
    tTensor *tensor = inst->tensor_ + model->io_tensors_[idx];
    input->dev_type_ = tensor->mem_.type_;
    input->dtype_ = tensor->dtype_;
    input->scale_ = tensor->scale_;
    input->shape_ = tensor->shape_;
    input->zero_ = tensor->zero_;
    input->dptr_ = (void *)tensor->dptr_;
  }
  return T_SUCCESS;
}

/**
 * Get input name by index
 * @param hdl Model handle
 * @param idx Input index
 * @return Input name string
 */
const char *tGetInputName(const tModelHandle hdl, const int32_t idx) {
  tModel *model = (tModel *)~hdl;
  return model->io_names_ + idx * model->io_name_len_;
}

/**
 * Get number of output tensors
 * @param hdl Model handle
 * @return Number of output tensors
 */
int32_t tGetOutputCount(const tModelHandle hdl) {
  tModel *model = (tModel *)~hdl;
  if (model == NULL || model->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }

  return model->num_output_;
}

/**
 * Get output name by index
 * @param hdl Model handle
 * @param idx Output index
 * @return Output name string
 */
const char *tGetOutputName(const tModelHandle hdl, const int32_t idx) {
  tModel *model = (tModel *)~hdl;
  int32_t id = idx + model->num_input_;
  return model->io_names_ + id * model->io_name_len_;
}

/**
 * Get input data type by index
 * @param hdl Model handle
 * @param idx Input index
 * @return Data type
 */
tDType tGetInputDataType(const tModelHandle hdl, const int32_t idx) {
  tModel *model = (tModel *)~hdl;
  tTensor *tensor = model->tensor_ + model->io_tensors_[idx];
  return tensor->dtype_;
}

/**
 * Get output data type by index
 * @param hdl Model handle
 * @param idx Output index
 * @return Data type
 */
tDType tGetOutputDataType(const tModelHandle hdl, const int32_t idx) {
  tModel *model = (tModel *)~hdl;
  tTensor *tensor =
      model->tensor_ + model->io_tensors_[idx + model->num_input_];
  return tensor->dtype_;
}

/**
 * Get input shape by index
 * @param hdl Model handle
 * @param idx Input index
 * @return Input shape
 */
tShape tGetInputShape(const tModelHandle hdl, const int32_t idx) {
  tModel *model = (tModel *)~hdl;
  tTensor *tensor = model->tensor_ + model->io_tensors_[idx];
  return tensor->shape_;
}

/**
 * Get output shape by index
 * @param hdl Model handle
 * @param idx Output index
 * @return Output shape
 */
tShape tGetOutputShape(const tModelHandle hdl, const int32_t idx) {
  tModel *model = (tModel *)~hdl;
  tTensor *tensor =
      model->tensor_ + model->io_tensors_[idx + model->num_input_];
  return tensor->shape_;
}

/**
 * Create executor for model execution
 * @param model_hdl Model handle
 * @param hdl Output execution handle
 * @param memory_list Memory allocation list
 * @param num_memory Number of memory entries
 * @return Status code
 */
#if !(defined(WIN32) || defined(linux))
#pragma clang optimize off
#endif
tStatus tCreateExecutor(const tModelHandle model_hdl, tExecHandle *hdl,
                        const tMemory *memory_list, const int32_t num_memory) {
  tModel *model = (tModel *)~model_hdl;
  if (hdl == NULL) {
    return T_ERR_INVALID_PARA;
  }

  uint16_t i = 0;
  uint16_t j = 0;
  int32_t inst_size = 0;
  inst_size += ALIGN16(sizeof(tExecInst));
  inst_size += ALIGN16(model->num_memory_ * sizeof(tMemory));
  inst_size += ALIGN16(model->num_tensor_ * sizeof(tTensor));
  inst_size += ALIGN16(sizeof(tDMA_List));
  tMemory inst_memory;

  for (i = 0; i < num_memory; i++) {
    if (memory_list[i].mem_type_ == 1) {
      assert(inst_size <= memory_list[i].size_);
      inst_memory = memory_list[i];
    }
  }

  int8_t *ptr = (int8_t *)(inst_memory.dptr_);
  tExecInst *inst = (tExecInst *)ptr;
  inst->flag_ = THINKER_INST_FLAG;
  inst->model_ = model;
  inst->inst_memory_ = inst_memory;
  ptr += ALIGN16(sizeof(tExecInst));

  inst->memory_ = (tMemory *)ptr;
  memcpy(inst->memory_, model->memory_, model->num_memory_ * sizeof(tMemory));

  for (i = model->num_shared_memory_; i < model->num_memory_; i++) {
    for (j = 0; j < num_memory; j++) {
      if ((memory_list[j].mem_type_ == 3) &&
          (memory_list[j].dev_type_ == inst->memory_[i].dev_type_)) {
        assert(inst->memory_[i].size_ <= memory_list[j].size_);
        inst->memory_[i] = memory_list[j];
        break;
      }
    }
  }
  ptr += ALIGN16((model->num_memory_) * sizeof(tMemory));

  inst->tensor_ = (tTensor *)ptr;
  memcpy(inst->tensor_, model->tensor_, model->num_tensor_ * sizeof(tTensor));
  ptr += ALIGN16(model->num_tensor_ * sizeof(tTensor));

  tMemory *memory = NULL;
  for (i = 0; i < model->num_tensor_; ++i) {
    tTensor *tensor = inst->tensor_ + i;
    for (j = model->num_shared_memory_; j < model->num_memory_; j++) {
      if (tensor->mem_.type_ == inst->memory_[j].dev_type_) {
        memory = &inst->memory_[j];
        break;
      }
    }
    if (tensor->mem_id_ < model->num_shared_memory_) {
      continue;
    }
    uint64_t offset = tensor->offset_;
    tensor->dptr_ = memory->dptr_ + offset;
  }

  uint8_t *p_op = model->op_buffer_;
  tTensor *local_tensor[512];
  for (i = 0; i < model->num_operator_; i++) {
    tOperator *op = (tOperator *)p_op;
    uint32_t *tensor_ids = (uint32_t *)(p_op + op->tensor_offset_);
    uint32_t num_tensor = op->num_input_ + op->num_output_ + op->num_temp_;
    tOperatorAPI *op_api = model->op_api_[op->op_id_];
    for (j = 0; j < num_tensor; j++) {
      local_tensor[j] = inst->tensor_ + tensor_ids[j];
    }
    tStatus ret;
    tHypeparam parm = {-1, NULL, NULL};
    ret = op_api->init(op, local_tensor, num_tensor, &parm);
    if (ret != T_SUCCESS) {
      return ret;
    }
    p_op += op->total_size_;
  }


  // // copy scalars of scalar graph to executor
  inst->shape_scalars_ = (double *) ptr;
  memcpy(inst->shape_scalars_, model->shape_infer->graph_->scalars_, 
         sizeof(double) * model->shape_infer->graph_->num_scalars_);
  ptr += ALIGN16(model->shape_infer->graph_->num_scalars_*sizeof(double));        

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
  inst->dma_list_ = (tDMA_List *)ptr;
  inst->dma_list_->total_ = model->dma_info_->count_;
  inst->dma_list_->cout_ = 0;
  tDMA *dma_temp = model->dma_;
  for (i = 0; i < inst->dma_list_->total_; i++) {
    uint32_t src_id = dma_temp->src_tensor_id_;
    uint32_t dst_id = dma_temp->dst_tensor_id_;
    inst->dma_list_->dma_[i].src_tensors_ = inst->tensor_ + src_id;
    inst->dma_list_->dma_[i].dst_tensors_ = inst->tensor_ + dst_id;
    inst->dma_list_->dma_[i].size_ = dma_temp->size_;
    dma_temp++;
  }
#else
    return T_ERR_INVALID_PLATFROM;
#endif

  *hdl = ~((tModelHandle)inst);
  return T_SUCCESS;
}
#if !(defined(WIN32) || defined(linux))
#pragma clang optimize on
#endif

/**
 * Release executor resources
 * @param hdl Execution handle to release
 * @return Status code
 */
tStatus tReleaseExecutor(tExecHandle hdl) {
  tExecInst *inst = (tExecInst *)~hdl;
  tModel *model = inst->model_;
  if (inst == NULL || inst->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }
  uint8_t *p_op = model->op_buffer_;
  tTensor *local_tensor[512];
  for (int32_t i = 0; i < model->num_operator_; ++i) {
    tOperator *op = (tOperator *)p_op;
    uint32_t *tensor_ids = (uint32_t *)(p_op + op->tensor_offset_);
    uint32_t num_tensor = op->num_input_ + op->num_output_ + op->num_temp_;
    tOperatorAPI *op_api = model->op_api_[op->op_id_];
    for (int32_t ii = 0; ii < num_tensor; ++ii) {
      local_tensor[ii] = inst->tensor_ + tensor_ids[ii];
    }

    tStatus ret = op_api->fini(op, local_tensor, num_tensor);
    if (ret != T_SUCCESS) {
      return ret;
    }
    p_op += op->total_size_;
  }

  return T_SUCCESS;
}

/**
 * Set input tensor by index
 * @param hdl Execution handle
 * @param idx Input index
 * @param input Input data
 * @return Status code
 */
tStatus tSetInput(const tExecHandle hdl, const int32_t idx,
                  const tData *input) {
  tExecInst *inst = (tExecInst *)~hdl;
  tModel *model = inst->model_;
  if (inst == NULL || inst->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }

  if (idx < 0 || idx >= model->num_input_) {
    return T_ERR_INDEX_OF_BOUND;
  }

  if (input == NULL) {
    return T_ERR_INVALID_DATA;
  }

  {
    tTensor *tensor = inst->tensor_ + model->io_tensors_[idx];
    tDType dtype = input->dtype_;
    if ((dtype & 0xFF) != (tensor->dtype_ & 0xff)) {
      printf("input dtype :%d, model need type:%d\n", dtype, tensor->dtype_);
      return T_ERR_INVALID_DATATYPE;
    }

    // TODO: check shape size
    tensor->dtype_ = dtype;
    tShape *max_shape = &model->input_shape_[idx];
    if (max_shape->ndim_ != tensor->shape_.ndim_) {
      printf("input dims :%d, model need dims:%d\n", max_shape->ndim_,
             tensor->shape_.ndim_);
      return T_ERR_INVALID_DATA;
    }
    for (int32_t i = 0; i < input->shape_.ndim_; i++) {
      if (input->shape_.dims_[i] > max_shape->dims_[i]) {
        printf("%d, input shape :%d, model max shape:%d\n", i,
               input->shape_.dims_[i], max_shape->dims_[i]);
        return T_ERR_INVALID_DATA;
      }
    }
    tensor->shape_ = input->shape_;
    tensor->scale_ = input->scale_;
    uint64_t bytes = getShapeSize(&tensor->shape_) * (dtype & 0xFF);
    if ((uint64_t)tensor->dptr_ != (uint64_t)input->dptr_ && bytes != 0)
    {
		memcpy((void *)tensor->dptr_, input->dptr_, bytes);
#if !(defined(WIN32) || defined(linux))
#if THINKER_USE_ARCS
		if (((uint32_t)(tensor->dptr_) & 0x28000000) == 0x28000000)
		{
			HAL_FlushDCache_by_Addr((uint32_t *)(tensor->dptr_), bytes);
		}
#elif THINKER_USE_VENUSA
		if (((uint32_t)(tensor->dptr_) & 0x38000000) == 0x38000000)
		{
			HAL_FlushDCache_by_Addr((uint32_t *)(tensor->dptr_), bytes);
		}
#endif
#endif
    }
  }
  return T_SUCCESS;
}

/**
 * Set input tensor by name
 * @param hdl Execution handle
 * @param name Input name
 * @param input Input data
 * @return Status code
 */
tStatus tSetInputByName(const tExecHandle hdl, const char *name,
                        const tData *input) {
  tExecInst *inst = (tExecInst *)~hdl;
  tModel *model = inst->model_;
  if (inst == NULL || inst->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }

  if (input == NULL) {
    return T_ERR_INVALID_PARA;
  }

  char *input_name = model->io_names_;
  for (int32_t j = 0; j < model->num_input_; j++) {
    if (strcmp(input_name, name) == 0) {
      return tSetInput(hdl, j, input);
    }
    input_name += model->io_name_len_;
  }
  return T_SUCCESS;
}

/**
 * Get output tensor by index
 * @param hdl Execution handle
 * @param idx Output index
 * @param output Output data structure
 * @return Status code
 */
tStatus tGetOutput(const tExecHandle hdl, const int32_t idx, tData *output) {
  tExecInst *inst = (tExecInst *)~hdl;
  tModel *model = inst->model_;
  if (inst == NULL || inst->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }

  if (output == NULL) {
    return T_ERR_INVALID_PARA;
  }

  if (idx < 0 || idx >= model->num_output_) {
    return T_ERR_INDEX_OF_BOUND;
  }

  {
    int32_t id = idx + model->num_input_;
    tTensor *tensor = inst->tensor_ + model->io_tensors_[id];
    output->dptr_ = (void *)tensor->dptr_;
    output->dtype_ = tensor->dtype_;
    output->shape_ = tensor->shape_;
    output->scale_ = tensor->scale_;
  }
  return T_SUCCESS;
}

/**
 * Get output tensor by name
 * @param hdl Execution handle
 * @param name Output name
 * @param output Output data structure
 * @return Status code
 */
tStatus tGetOutputByName(const tExecHandle hdl, const char *name,
                         tData *output) {
  tExecInst *inst = (tExecInst *)~hdl;
  tModel *model = inst->model_;
  if (inst == NULL || inst->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }
  if (output == NULL) {
    return T_ERR_INVALID_PARA;
  }

  char *input_name = model->io_names_ + model->num_input_ * model->io_name_len_;
  for (int32_t j = 0; j < model->num_output_; j++) {
    if (strcmp(input_name, name) == 0) {
      return tGetOutput(hdl, j, output);
    }
    input_name += model->io_name_len_;
  }
  return T_ERR_INVALID_PARA;
}

/**
 * Execute forward pass
 * @param hdl Execution handle
 * @return Status code
 */
tStatus tForward(const tExecHandle hdl) {
  tStatus ret = T_SUCCESS;
  tExecInst *inst = (tExecInst *)~hdl;
  tModel *model = inst->model_;
  tTensor *local_tensor[512];
  uint8_t *p_op = NULL;
  int32_t i;
  if (inst == NULL || inst->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }
  p_op = model->op_buffer_;

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
  tDMA_List *dma_list = inst->dma_list_;
  if (dma_list->total_ > 0) {
	  dma_list->cout_ = 0;
	  int32_t index = dma_list->cout_;
	  tTensor *src = dma_list->dma_[index].src_tensors_;
	  tTensor *dst = dma_list->dma_[index].dst_tensors_;
	  int32_t size = dma_list->dma_[index].size_;
#if !defined(THINKER_USE_VENUSA)
	  if (size > 0) {
      dma_cpy_async(0, (void *)dst->dptr_, (void *)src->dptr_, size);
      dma_list->cout_++;
	  }
  }
#else 
  #include "core/ops/venusA/luna/luna_misc_math.h"
  if (size > 0) {
    dma_cpy_async(5, (int8_t *)dst->dptr_, (int8_t *)src->dptr_, size);
		dma_list->cout_++;
	  }
  }
#endif // !defined(THINKER_USE_VENUSA)
#endif // defined(THINKER_USE_VENUS) || defined(THINKER_USE_ARCS) || defined(THINKER_USE_VENUSA)

  for (i = 0; i < model->num_operator_; ++i) {
    int32_t ii = 0;
    tOperator *op = (tOperator *)p_op;
    uint32_t *tensor_ids = (uint32_t *)(p_op + op->tensor_offset_);
    uint32_t num_tensor = op->num_input_ + op->num_output_ + op->num_temp_;
    tOperatorAPI *op_api = model->op_api_[op->op_id_];

    if (T_FORCE_STOP_VALUE == inst->force_stop_flag)  //user force to stop
    {
#if THINKER_USE_VENUS || THINKER_USE_ARCS
		if (dma_list->cout_ && (dma_list->cout_ < dma_list->total_))
		{
			dma_wait_complete(0);
		}
#elif THINKER_USE_VENUSA
    if (dma_list->cout_ && (dma_list->cout_ < dma_list->total_))
		{
      luna_gpdma_wait(5);
		}
#endif // defined(THINKER_USE_VENUSA)
        return T_FORCE_STOP_VALUE;
    }

    for (ii = 0; ii < num_tensor; ++ii) {
      local_tensor[ii] = inst->tensor_ + tensor_ids[ii];
    }
    PROFILE_BEGIN
    ret = op_api->forward(op, local_tensor, num_tensor, inst->dma_list_);
  //  printf("[%d]op_name:%s\n", i, op_api->name());
    if (ret != T_SUCCESS) {
      printf("forward error code :%d, op index :%d, op name: %s\n", ret, i,
             op_api->name());
      return ret;
    }
    tTensorName *name_list =
        (tTensorName *)inst->model_->debug_info->tensor_name_list_;

#if THINKER_DUMP 
#if (defined(WIN32) || defined(linux))
    for (size_t j = 0; j < op->num_output_; j++) {
      tTensor *out_tensor = local_tensor[op->num_input_ + j];
      int32_t tensor_id = tensor_ids[op->num_input_ + j];
      write_file(name_list[tensor_id].name_, out_tensor);
    }
#endif
#endif

#ifdef THINKER_RESULT_CRC_PRINT
    for (size_t j = 0; j < op->num_output_; j++) {
      tTensor *out_tensor = local_tensor[op->num_input_ + j];
      int32_t tensor_id = tensor_ids[op->num_input_ + j];
      uint8_t *data = (uint8_t *)out_tensor->dptr_;
      uint32_t data_size = getTensorSize(out_tensor)*out_tensor->byte_;
#if !(defined(WIN32) || defined(linux))
#if THINKER_USE_ARCS
		if (((uint32_t)(out_tensor->dptr_) & 0x28000000) == 0x28000000)
		{
			HAL_FlushDCache_by_Addr((uint32_t *)(out_tensor->dptr_), data_size);
		}
#elif THINKER_USE_VENUSA
		if (((uint32_t)(out_tensor->dptr_) & 0x38000000) == 0x38000000)
		{
			HAL_FlushInvalidateDCache_by_Addr((uint32_t *)(out_tensor->dptr_), data_size);
		}
#endif
#endif
      int32_t result_crc = crc32_calc(data, data_size);
      printf("crc32_calc = 0x%08x, data = [0x%08x-0x%08x-0x%08x], name = %s\n", result_crc,
            ((uint32_t *)(data))[0], ((uint32_t *)(data + data_size / 2))[0],
            ((uint32_t *)(data + data_size - 4))[0], name_list[tensor_id].name_);
    }
#endif
    PROFILE_END

    p_op += op->total_size_;
  }
  return T_SUCCESS;
}

/**
 * Update dynamic shapes during execution
 * @param hdl Execution handle
 * @param axis_names Array of axis names
 * @param axis_sizes Array of axis sizes
 * @param num Number of axes to update
 * @return Status code
 */
tStatus tUpdateShape(tExecHandle hdl, const char **axis_names, const uint32_t *axis_sizes, int32_t num)
{
    tExecInst *inst = (tExecInst *)~hdl;
    tModel *model   = inst->model_;
    tStatus ret     = T_SUCCESS;
    if (num != 0)
        ret = tSetShapeInferInputByNames(model->shape_infer, inst->shape_scalars_, axis_names, axis_sizes, num);
    else
        ret = tSetShapeInferInputByTensors(model->shape_infer, inst->shape_scalars_, inst->tensor_);
    if (ret != T_SUCCESS) return ret;
    ret = tShapeInferForward(model->shape_infer, inst->shape_scalars_, inst->tensor_);
    return ret;
}

/**
 * Start execution
 * @param hdl Execution handle
 * @return Status code
 */
tStatus tExecutorStart(tExecHandle hdl)
{
    tExecInst *inst = (tExecInst *)~hdl;
    if (inst == NULL || inst->flag_ != THINKER_INST_FLAG)
    {
        return T_ERR_INVALID_INST;
    }
    inst->force_stop_flag = 0;
    return T_SUCCESS;
}

/**
 * Stop execution
 * @param hdl Execution handle
 * @return Status code
 */
tStatus tExecutorStop(tExecHandle hdl)
{
    tExecInst *inst = (tExecInst *)~hdl;
    if (inst == NULL || inst->flag_ != THINKER_INST_FLAG)
    {
        return T_ERR_INVALID_INST;
    }
    inst->force_stop_flag = T_FORCE_STOP_VALUE;
    return T_SUCCESS;
}

#if THINKER_USE_MTQ
/**
 * Get Luna list size information
 * @param hdl Execution handle
 * @param list_size Output list size
 * @param list_length Output list length
 * @param total_param Output total parameter size
 * @return Status code
 */
tStatus tGetLunaListSize(const tExecHandle hdl, uint32_t *list_size, uint32_t *list_length, uint32_t *total_param) {
  tStatus ret = T_SUCCESS;
  tExecInst *inst = (tExecInst *)~hdl;
  tModel *model = inst->model_;
  tTensor *local_tensor[512];
  uint8_t *p_op = NULL;
  g_submit_pos = 0;
  g_param_size = 0;
  int32_t i;
  if (inst == NULL || inst->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }
  p_op = model->op_buffer_;

  luna_register_hook(luna_execute_cmd_hook_for_get_list_length, 0);

#if THINKER_USE_VENUSA
  #include "core/ops/venusA/luna/luna_misc_math.h"
  tDMA_List *dma_list = inst->dma_list_;
  if (dma_list->total_ > 0) {
	  dma_list->cout_ = 0;
	  int32_t index = dma_list->cout_;
	  tTensor *src = dma_list->dma_[index].src_tensors_;
	  tTensor *dst = dma_list->dma_[index].dst_tensors_;
	  int32_t size = dma_list->dma_[index].size_;
    if (size > 0) {
      dma_cpy_async(5, (int8_t *)dst->dptr_, (int8_t *)src->dptr_, size);
      dma_list->cout_++;
    }
  }
#endif // defined(THINKER_USE_VENUSA)

  for (i = 0; i < model->num_operator_; ++i) {
    int32_t ii = 0;
    tOperator *op = (tOperator *)p_op;
    uint32_t *tensor_ids = (uint32_t *)(p_op + op->tensor_offset_);
    uint32_t num_tensor = op->num_input_ + op->num_output_ + op->num_temp_;
    tOperatorAPI *op_api = model->op_api_[op->op_id_];

    if (T_FORCE_STOP_VALUE == inst->force_stop_flag)  //user force to stop
    {
#if THINKER_USE_VENUSA
    if (dma_list->cout_ && (dma_list->cout_ < dma_list->total_))
		{
      dma_wait_complete(5);
		}
#endif // defined(THINKER_USE_VENUSA)
        return T_FORCE_STOP_VALUE;
    }

    for (ii = 0; ii < num_tensor; ++ii) {
      local_tensor[ii] = inst->tensor_ + tensor_ids[ii];
    }

    ret = op_api->forward(op, local_tensor, num_tensor, inst->dma_list_);
  //  printf("[%d]op_name:%s\n", i, op_api->name());
    if (ret != T_SUCCESS) {
      printf("forward error code :%d, op index :%d, op name: %s\n", ret, i,
             op_api->name());
      return ret;
    }

    p_op += op->total_size_;
  }

  luna_register_hook(0, 0);

	*list_size = g_submit_pos*sizeof(luna_mtq_sq_elem_t) + g_submit_pos*sizeof(luna_mtq_cq_elem_t) + g_param_size;
	*list_length = g_submit_pos;
	*total_param = g_param_size;
	g_submit_pos = 0;
	g_param_size = 0;
  return ret;
}

/**
 * Build Luna list for execution
 * @param hdl Execution handle
 * @param base_addr Base address for list building
 * @param sq_len Sequence length
 * @return Status code
 */
tStatus tBuildLunaList(const tExecHandle hdl, int8_t *base_addr, uint32_t sq_len) {

	g_sq_addr_user_ch = (int8_t *)base_addr;
	g_cq_addr_user_ch = (int8_t *)base_addr + sq_len*sizeof(luna_mtq_sq_elem_t);
	g_param_addr = (int8_t *)base_addr + sq_len*sizeof(luna_mtq_sq_elem_t) + sq_len*sizeof(luna_mtq_cq_elem_t);
	g_submit_pos = 0;
	g_param_size = 0;

  tStatus ret = T_SUCCESS;
  tExecInst *inst = (tExecInst *)~hdl;
  tModel *model = inst->model_;
  tTensor *local_tensor[512];
  uint8_t *p_op = NULL;
  int32_t i;
  if (inst == NULL || inst->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }
  p_op = model->op_buffer_;

  luna_register_hook(luna_execute_cmd_hook_for_build_list, 0);

#if THINKER_USE_VENUSA
  #include "core/ops/venusA/luna/luna_misc_math.h"
  tDMA_List *dma_list = inst->dma_list_;
  if (dma_list->total_ > 0) {
	  dma_list->cout_ = 0;
	  int32_t index = dma_list->cout_;
	  tTensor *src = dma_list->dma_[index].src_tensors_;
	  tTensor *dst = dma_list->dma_[index].dst_tensors_;
	  int32_t size = dma_list->dma_[index].size_;
    if (size > 0) {
      dma_cpy_async(5, (int8_t *)dst->dptr_, (int8_t *)src->dptr_, size);
      dma_list->cout_++;
    }
  }
#endif // defined(THINKER_USE_VENUSA)

  for (i = 0; i < model->num_operator_; ++i) {
    int32_t ii = 0;
    tOperator *op = (tOperator *)p_op;
    uint32_t *tensor_ids = (uint32_t *)(p_op + op->tensor_offset_);
    uint32_t num_tensor = op->num_input_ + op->num_output_ + op->num_temp_;
    tOperatorAPI *op_api = model->op_api_[op->op_id_];

    if (T_FORCE_STOP_VALUE == inst->force_stop_flag)  //user force to stop
    {
#if THINKER_USE_VENUSA
    if (dma_list->cout_ && (dma_list->cout_ < dma_list->total_))
		{
      dma_wait_complete(5);
		}
#endif // defined(THINKER_USE_VENUSA)
        return T_FORCE_STOP_VALUE;
    }

    for (ii = 0; ii < num_tensor; ++ii) {
      local_tensor[ii] = inst->tensor_ + tensor_ids[ii];
    }

    ret = op_api->forward(op, local_tensor, num_tensor, inst->dma_list_);
  //  printf("[%d]op_name:%s\n", i, op_api->name());
    if (ret != T_SUCCESS) {
      printf("forward error code :%d, op index :%d, op name: %s\n", ret, i,
             op_api->name());
      return ret;
    }
    tTensorName *name_list =
        (tTensorName *)inst->model_->debug_info->tensor_name_list_;

    p_op += op->total_size_;
  }

  luna_register_hook(0, 0);
  return T_SUCCESS;
}

/**
 * Subdivide Luna list
 * @param sq_addr Sequence address
 * @param sq_len Sequence length
 * @param total_param_size Total parameter size
 * @return Status code
 */
tStatus tSubLunaList(int8_t *sq_addr, uint32_t sq_len, uint32_t total_param_size) {
	luna_mtq_sq_elem_t *last_op = (luna_mtq_sq_elem_t *)sq_addr + (sq_len-1);
	last_op->op_interrupt_enable = 1;
	HAL_FlushInvalidateDCache_by_Addr(sq_addr, sq_len*sizeof(luna_mtq_sq_elem_t) + sq_len*sizeof(luna_mtq_cq_elem_t) + total_param_size);
	luna_mtq_sq_elem_t *cq_addr = (int8_t *)sq_addr + sq_len*sizeof(luna_mtq_sq_elem_t);
	return(luna_scheduler_run_static(0, sq_addr, cq_addr, sq_len, sq_len, 0, 1));
}

/**
 * Get result from Luna list
 * @param hdl Execution handle
 * @return Status code
 */
tStatus tGetListResult(const tExecHandle hdl) {
  return luna_scheduler_wait(0);
}
#endif

/**
 * Get API interface pointer
 * @return Pointer to thinker API structure
 */
static thinkerApi g_api;
const thinkerApi *thinkerGetApi() {
        g_api.tGetVersion = tGetVersion;

        g_api.tInitialize = tInitialize;
        g_api.tUninitialize = tUninitialize;

        g_api.tGetMemoryPlan = tGetMemoryPlan;

        g_api.tModelInit = tModelInit;
        g_api.tModelFini = tModelFini;

        g_api.tGetInputCount = tGetInputCount;
        g_api.tGetInputInfo = tGetInputInfo;
        g_api.tGetInputName = tGetInputName;
        g_api.tGetOutputCount = tGetOutputCount;
        g_api.tGetOutputName = tGetOutputName;
        g_api.tGetInputDataType = tGetInputDataType;
        g_api.tGetOutputDataType = tGetOutputDataType;
        g_api.tGetInputShape = tGetInputShape;
        g_api.tGetOutputShape = tGetOutputShape;

        g_api.tCreateExecutor = tCreateExecutor;
        g_api.tReleaseExecutor = tReleaseExecutor;

        g_api.tSetInput = tSetInput;
        g_api.tSetInputByName = tSetInputByName;
        g_api.tUpdateShape = tUpdateShape;
        g_api.tGetOutput = tGetOutput;
        g_api.tGetOutputByName = tGetOutputByName;
        g_api.tForward = tForward;

        g_api.tExecutorStart = tExecutorStart;
        g_api.tExecutorStop = tExecutorStop;

#if THINKER_USE_MTQ
        g_api.tGetLunaListSize = tGetLunaListSize;
        g_api.tBuildLunaList = tBuildLunaList;
        g_api.tSubLunaList = tSubLunaList;
        g_api.tGetListResult = tGetListResult;
#endif

  return &g_api;
}
