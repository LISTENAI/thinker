#include "thinker_api.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_init.h"
#include "core/operator_register.h"
#include "thinker_crc24.h"
#include "thinker_debug.h"
#include "thinker_define.h"
#include "thinker_memory.h"

#if THINKER_USE_VENUS
#include "ops/venus/luna/opi_psram_cpy.h"
#endif

#ifndef NULL
#define NULL 0
#endif

#define THINKER_INST_FLAG 0x20201201
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
  tDebugList *debug_info;
  tDMAList *dma_info_;
  tDMA *dma_;
} tModel;

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

const char *tGetVersion(const int8_t index){
   return (1 == index) ? VENUS_VERSION : THINKER_VERSION;
}

tStatus tInitialize() {
  init_ops_list();

  return T_SUCCESS;
}

tStatus tUninitialize() { return T_SUCCESS; }

tStatus tGetMemoryPlan(tMemory *memory_list, int32_t *num_memory,
                       const int8_t *res, const uint64_t size) {
  tModelHeader *res_hdr = (tModelHeader *)res;

  if (res_hdr->total_size_ > size) {
    return T_ERR_RES_INCOMPLETE;
  }
  // check CRC
 if (res_hdr->crc32_ != 0) {
   uint8_t *res_model_ptr = (uint8_t *)res + ALIGN16(sizeof(tModelHeader));
   int32_t res_model_size = size - ALIGN16(sizeof(tModelHeader));
   int32_t crc_check = crc24(0, res_model_ptr, res_model_size);
   if (res_hdr->crc32_ != crc_check) {
     printf("%d,%d", res_hdr->crc32_, crc_check);
     return T_ERR_RES_CRC_CHECK;
   }
 }
  //  model_inst_size
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
  model_inst_size +=
      ALIGN16((io_hdr.num_input_ + io_hdr.num_output_) * sizeof(int32_t));
  model_inst_size +=
      ALIGN16((io_hdr.num_input_ + io_hdr.num_output_) * io_hdr.name_length_);
  model_inst_size += ALIGN16(io_hdr.num_input_ * sizeof(tShape));
  model_inst_size += ALIGN16(op_hdr.type_count_ * sizeof(tOperatorAPI));
  model_inst_size += ALIGN16(op_hdr.op_size_);
  model_inst_size += ALIGN16(sizeof(tDebugList));
  model_inst_size += ALIGN16(sizeof(tDMAList));
  model_inst_size += ALIGN16(dma_hdr.count_ * sizeof(tDMA));

  int32_t num = 0;
  tMemory model_inst_memory;
  model_inst_memory.size_ = model_inst_size;
  model_inst_memory.dptr_ = 0;
  model_inst_memory.dev_type_ = 1;
  model_inst_memory.mem_type_ = 0;
  memory_list[num] = model_inst_memory;
  num += 1;

  // params data size
  tMemory *shared_memory =
      (tMemory *)(res + res_hdr->memory_offset_ + mem_hdr.offset_);

  for (int32_t i = 0; i < mem_hdr.shared_count_; ++i) {
    tParameter *params =
        (tParameter *)((uint8_t *)res + res_hdr->param_offset_ +
                       param_hdr.elem_size_);

    memory_list[num] = shared_memory[i];
    memory_list[num].dptr_ = (addr_type)((int8_t *)params + params->offset_);
    memory_list[num].mem_type_ = 2;
    num += 1;
  }
  // runtime size
  for (int32_t i = mem_hdr.shared_count_; i < mem_hdr.total_count_; ++i) {
    memory_list[num] = shared_memory[i];
    memory_list[num].mem_type_ = 3;
    num += 1;
  }

  int32_t inst_size = 0;
  inst_size += ALIGN16(sizeof(tExecInst));
  inst_size += ALIGN16(mem_hdr.total_count_ * sizeof(tMemory));
  inst_size += ALIGN16(tensor_hdr.count_ * sizeof(tTensor));
  inst_size += ALIGN16(sizeof(tDMA_List));
  tMemory inst_memory;
  inst_memory.size_ = inst_size;
  inst_memory.dptr_ = 0;
  inst_memory.dev_type_ = 1;
  inst_memory.mem_type_ = 1;
  memory_list[num] = inst_memory;
  num += 1;

  num_memory[0] = num;
  return T_SUCCESS;
}

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
  // check CRC
 if (res_hdr->crc32_ != 0) {
   uint8_t *res_model_ptr = (uint8_t *)res + ALIGN16(sizeof(tModelHeader));
   int32_t res_model_size = size - ALIGN16(sizeof(tModelHeader));
   int32_t crc_check = crc24(0, res_model_ptr, res_model_size);
   if (res_hdr->crc32_ != crc_check) {
     return T_ERR_RES_CRC_CHECK;
   }
 }

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

  uint8_t *param_ptr =
      (uint8_t *)res + res_hdr->param_offset_ + param_hdr.elem_size_;
  for (i = 0; i < param_hdr.count_; ++i) {
    tParameter *params = (tParameter *)param_ptr;
    tMemory *memory = &inst->memory_[params->mem_id_];
    param_ptr += params->offset_;
    memcpy((void *)memory->dptr_, param_ptr, params->size_);
    param_ptr += params->size_;
  }

  for (i = 0; i < inst->num_tensor_; ++i) {
    tTensor *tensor = inst->tensor_ + i;
    tMemory *memory = &inst->memory_[tensor->mem_id_];
    if (tensor->mem_id_ >= inst->num_shared_memory_) {
      continue;
    }
    uint64_t offset = tensor->offset_;
    tensor->dptr_ = memory->dptr_ + offset;
  }

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

tStatus tModelFini(tModelHandle hdl) {
#if !THINKER_USE_ACL
  tModel *model = (tModel *)~hdl;
  if (model == NULL || model->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }

  tMemoryFree(&model->inst_memory_);
  return T_SUCCESS;
#endif
}

int32_t tGetInputCount(const tModelHandle hdl) {
  tModel *model = (tModel *)~hdl;
  return model->num_input_;
}

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

const char *tGetInputName(const tModelHandle hdl, const int32_t idx) {
  tModel *model = (tModel *)~hdl;
  return model->io_names_ + idx * model->io_name_len_;
}

int32_t tGetOutputCount(const tModelHandle hdl) {
  tModel *model = (tModel *)~hdl;
  if (model == NULL || model->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }

  return model->num_output_;
}

const char *tGetOutputName(const tModelHandle hdl, const int32_t idx) {
  tModel *model = (tModel *)~hdl;
  int32_t id = idx + model->num_input_;
  return model->io_names_ + id * model->io_name_len_;
}

tDType tGetInputDataType(const tModelHandle hdl, const int32_t idx) {
  tModel *model = (tModel *)~hdl;
  tTensor *tensor = model->tensor_ + model->io_tensors_[idx];
  return tensor->dtype_;
}

tDType tGetOutputDataType(const tModelHandle hdl, const int32_t idx) {
  tModel *model = (tModel *)~hdl;
  tTensor *tensor =
      model->tensor_ + model->io_tensors_[idx + model->num_input_];
  return tensor->dtype_;
}

tShape tGetInputShape(const tModelHandle hdl, const int32_t idx) {
  tModel *model = (tModel *)~hdl;
  tTensor *tensor = model->tensor_ + model->io_tensors_[idx];
  return tensor->shape_;
}

tShape tGetOutputShape(const tModelHandle hdl, const int32_t idx) {
  tModel *model = (tModel *)~hdl;
  tTensor *tensor =
      model->tensor_ + model->io_tensors_[idx + model->num_input_];
  return tensor->shape_;
}

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

#ifdef THINKER_USE_VENUS
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

  tMemoryFree(&inst->inst_memory_);
  return T_SUCCESS;
}

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
    if ((dtype & 0xFF) > (tensor->dtype_ & 0xff)) {
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
    if ((uint64_t)tensor->dptr_ != (uint64_t)input->dptr_)
      memcpy((void *)tensor->dptr_, input->dptr_, bytes);
  }
  return T_SUCCESS;
}

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

#ifdef THINKER_USE_VENUS
  tDMA_List *dma_list = inst->dma_list_;
  dma_list->cout_ = 0;
  int32_t index = dma_list->cout_;
  tTensor *src = dma_list->dma_[index].src_tensors_;
  tTensor *dst = dma_list->dma_[index].dst_tensors_;
  int32_t size = dma_list->dma_[index].size_;
  if (size > 0) {
    dma_cpy_async(0, (void *)dst->dptr_, (void *)src->dptr_, size);
    dma_list->cout_++;
  }
#endif

  for (i = 0; i < model->num_operator_; ++i) {
    int32_t ii = 0;
    tOperator *op = (tOperator *)p_op;
    uint32_t *tensor_ids = (uint32_t *)(p_op + op->tensor_offset_);
    uint32_t num_tensor = op->num_input_ + op->num_output_ + op->num_temp_;
    tOperatorAPI *op_api = model->op_api_[op->op_id_];

    if (T_FORCE_STOP_VALUE == inst->force_stop_flag)  //user force to stop
    {
        return T_FORCE_STOP_VALUE;
    }

    for (ii = 0; ii < num_tensor; ++ii) {
      local_tensor[ii] = inst->tensor_ + tensor_ids[ii];
    }
    PROFILE_BEGIN
    ret = op_api->forward(op, local_tensor, num_tensor, inst->dma_list_);
    if (ret != T_SUCCESS) {
      printf("forward error code :%d, op index :%d, op name: %s\n", ret, i,
             op_api->name());
      return ret;
    }
#if THINKER_DUMP
    tTensorName *name_list =
        (tTensorName *)inst->model_->debug_info->tensor_name_list_;
    for (size_t j = 0; j < op->num_output_; j++) {
      tTensor *out_tensor = local_tensor[op->num_input_ + j];
      int32_t tensor_id = tensor_ids[op->num_input_ + j];
      write_file(name_list[tensor_id].name_, out_tensor);
    }
#endif

    PROFILE_END

    p_op += op->total_size_;
  }
  return T_SUCCESS;
}

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
  g_api.tGetOutput = tGetOutput;
  g_api.tGetOutputByName = tGetOutputByName;
  g_api.tForward = tForward;

  g_api.tExecutorStart = tExecutorStart;
  g_api.tExecutorStop = tExecutorStop;

  return &g_api;
}
