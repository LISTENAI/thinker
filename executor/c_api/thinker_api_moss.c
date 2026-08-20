#if THINKER_USE_MOSS
#include "thinker_api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "moss_model.h"
#include "moss_runtime.h"
#include "thinker_define.h"
#include "thinker_type.h"

#ifndef NULL
#define NULL 0
#endif

#define THINKER_INST_FLAG 0x20201201
#define MOSS_IO_MEMORY_TYPE 5
#define THINKER_MOSS_MAX_RANK 7

typedef struct _t_MossModel_ {
  uint32_t flag_;
  tMemory inst_memory_;
  const MossModelDesc *desc_;
  MossModelDesc *desc_storage_;
  MossTensorInfo *input_infos_;
  MossTensorInfo *output_infos_;
  addr_type weight_ptr_;
  uint16_t num_input_;
  uint16_t num_output_;
  tData *input_info_;
  tData *output_info_;
  tShape *input_shape_;
  tShape *output_shape_;
  char *io_names_;
  int32_t io_name_len_;
} tMossModel;

typedef struct _t_MossExecInst_ {
  uint32_t flag_;
  tMemory inst_memory_;
  tMossModel *model_;
  void *legacy_model_;
  MossTensor *inputs_;
  MossTensor *outputs_;
  uint32_t force_stop_flag;
} tMossExecInst;

/* Layout follows executor/include/moss/runtime.h, used by libmossruntime.so. */
typedef struct _t_MossLegacyModel_ {
  uint32_t space_size_;
  uint32_t workspace_arg_idx_;
  uint32_t weight_arg_idx_;
  int32_t num_input_;
  int32_t num_output_;
  void *inout_list_;
  int32_t num_arg_;
  void **args_;
  int32_t (*entry_func_)(uint32_t, void **);
} tMossLegacyModel;

#define THINKER_MOSS_RESOURCE_LABEL "thinker_moss10"
#define THINKER_MOSS_RESOURCE_LABEL_SIZE 16u
#define THINKER_MOSS_RESOURCE_HEADER_SIZE \
  (THINKER_MOSS_RESOURCE_LABEL_SIZE + 13u * 4u)
#define THINKER_MOSS_RESOURCE_MAGIC 0x53534F4Du
#define THINKER_MOSS_RESOURCE_ABI_VERSION 1
#define THINKER_MOSS_MAX_RESOURCES 8
#define THINKER_MOSS_MAX_IO_TENSORS 16

#if defined(__GNUC__)
#define THINKER_MOSS_WEAK __attribute__((weak))
#else
#define THINKER_MOSS_WEAK
#endif

typedef struct _t_MossPackedResourceHeader_ {
  char label_[16];
  uint32_t magic_;
  uint32_t abi_version_;
  uint32_t header_size_;
  uint32_t total_size_;
  uint32_t flags_;
  uint32_t weight_offset_;
  uint32_t weight_size_;
  uint32_t getter_offset_;
  uint32_t getter_size_;
  uint32_t model_name_offset_;
  uint32_t model_name_size_;
  uint32_t target_arch_offset_;
  uint32_t target_arch_size_;
} tMossPackedResourceHeader;

typedef struct _t_MossResolvedResource_ {
  const int8_t *res_;
  uint64_t size_;
  MossModelDesc desc_;
  MossTensorInfo input_infos_[THINKER_MOSS_MAX_IO_TENSORS];
  MossTensorInfo output_infos_[THINKER_MOSS_MAX_IO_TENSORS];
  void *legacy_model_;
  addr_type weight_ptr_;
  uint8_t used_;
} tMossResolvedResource;

static tMossResolvedResource g_moss_resources[THINKER_MOSS_MAX_RESOURCES];

THINKER_MOSS_WEAK void *thinker_moss_get_registered_model(
    const char *getter_name, const char *model_name) {
  (void)getter_name;
  (void)model_name;
  return NULL;
}

static uint32_t moss_read_le32(const int8_t *ptr) {
  const uint8_t *bytes = (const uint8_t *)ptr;
  return ((uint32_t)bytes[0]) | ((uint32_t)bytes[1] << 8) |
         ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static int32_t moss_parse_packed_header(const int8_t *res, uint64_t size,
                                        tMossPackedResourceHeader *hdr) {
  const int8_t *cursor = NULL;

  if (res == NULL || hdr == NULL ||
      size < THINKER_MOSS_RESOURCE_HEADER_SIZE) {
    return 0;
  }

  memset(hdr, 0, sizeof(*hdr));
  memcpy(hdr->label_, res, THINKER_MOSS_RESOURCE_LABEL_SIZE);
  cursor = res + THINKER_MOSS_RESOURCE_LABEL_SIZE;
  hdr->magic_ = moss_read_le32(cursor);
  cursor += 4;
  hdr->abi_version_ = moss_read_le32(cursor);
  cursor += 4;
  hdr->header_size_ = moss_read_le32(cursor);
  cursor += 4;
  hdr->total_size_ = moss_read_le32(cursor);
  cursor += 4;
  hdr->flags_ = moss_read_le32(cursor);
  cursor += 4;
  hdr->weight_offset_ = moss_read_le32(cursor);
  cursor += 4;
  hdr->weight_size_ = moss_read_le32(cursor);
  cursor += 4;
  hdr->getter_offset_ = moss_read_le32(cursor);
  cursor += 4;
  hdr->getter_size_ = moss_read_le32(cursor);
  cursor += 4;
  hdr->model_name_offset_ = moss_read_le32(cursor);
  cursor += 4;
  hdr->model_name_size_ = moss_read_le32(cursor);
  cursor += 4;
  hdr->target_arch_offset_ = moss_read_le32(cursor);
  cursor += 4;
  hdr->target_arch_size_ = moss_read_le32(cursor);
  return 1;
}

static int32_t moss_is_packed_resource(const int8_t *res, uint64_t size) {
  tMossPackedResourceHeader hdr;

  if (!moss_parse_packed_header(res, size, &hdr)) {
    return 0;
  }
  return memcmp(hdr.label_, THINKER_MOSS_RESOURCE_LABEL,
                strlen(THINKER_MOSS_RESOURCE_LABEL)) == 0 &&
         hdr.magic_ == THINKER_MOSS_RESOURCE_MAGIC;
}

static int32_t moss_range_valid(uint64_t size, uint32_t offset,
                                uint32_t bytes) {
  return offset <= size && bytes <= size - offset;
}

static const char *moss_resource_string(const int8_t *res, uint64_t size,
                                        uint32_t offset, uint32_t bytes) {
  if (bytes == 0 || !moss_range_valid(size, offset, bytes)) {
    return NULL;
  }
  if (res[offset + bytes - 1u] != '\0') {
    return NULL;
  }
  return (const char *)(res + offset);
}

static void thinker_moss_copy_name(char dst[MOSS_MAX_NAME], const char *src) {
  memset(dst, 0, MOSS_MAX_NAME);
  if (src != NULL) {
    strncpy(dst, src, MOSS_MAX_NAME - 1);
  }
}

static int32_t moss_convert_legacy_tensor_info(void *legacy_model,
                                               int32_t is_input,
                                               int32_t index,
                                               MossTensorInfo *info) {
  MossTensor tensor;
  int32_t ret = 0;

  if (legacy_model == NULL || info == NULL) {
    return -1;
  }

  memset(&tensor, 0, sizeof(tensor));
  ret = is_input ? mGetModuleInputInfo(legacy_model, index, &tensor)
                 : mGetModuleOutputInfo(legacy_model, index, &tensor);
  if (ret != 0) {
    return -1;
  }

  memset(info, 0, sizeof(*info));
  thinker_moss_copy_name(info->name, tensor.name);
  info->dtype = (MossDtype)((uint8_t)tensor.memorytype_datatype & 0x0F);
  info->mem_space =
      (MossMemSpace)(((uint8_t)tensor.memorytype_datatype >> 4) & 0x0F);
  info->rank = tensor.dim;
  if (info->rank < 0 || info->rank > THINKER_MOSS_MAX_RANK) {
    return -1;
  }
  for (int32_t i = 0; i < info->rank; ++i) {
    info->shape[i] = tensor.shape[i];
  }
  info->scale = tensor.scale;
  info->zero_point = 0;
  return 0;
}

static tMossResolvedResource *moss_alloc_resource_slot(const int8_t *res,
                                                       uint64_t size) {
  for (int32_t i = 0; i < THINKER_MOSS_MAX_RESOURCES; ++i) {
    if (g_moss_resources[i].used_ && g_moss_resources[i].res_ == res &&
        g_moss_resources[i].size_ == size) {
      return &g_moss_resources[i];
    }
  }
  for (int32_t i = 0; i < THINKER_MOSS_MAX_RESOURCES; ++i) {
    if (!g_moss_resources[i].used_) {
      memset(&g_moss_resources[i], 0, sizeof(g_moss_resources[i]));
      g_moss_resources[i].res_ = res;
      g_moss_resources[i].size_ = size;
      g_moss_resources[i].used_ = 1;
      return &g_moss_resources[i];
    }
  }
  return NULL;
}

static tStatus moss_resolve_packed_resource(const int8_t *res, uint64_t size,
                                            const MossModelDesc **desc,
                                            addr_type *weight_ptr) {
  tMossPackedResourceHeader hdr;
  tMossResolvedResource *slot = NULL;
  const char *getter_name = NULL;
  const char *model_name = NULL;
  const char *target_arch = NULL;
  tMossLegacyModel *legacy = NULL;

  if (!moss_parse_packed_header(res, size, &hdr)) {
    return T_ERR_RES_INCOMPLETE;
  }

  slot = moss_alloc_resource_slot(res, size);
  if (slot == NULL) {
    return T_ERR_NO_IMPLEMENTED;
  }
  if (slot->desc_.abi_version == MOSS_MODEL_ABI_VERSION) {
    if (desc != NULL) {
      *desc = &slot->desc_;
    }
    if (weight_ptr != NULL) {
      *weight_ptr = slot->weight_ptr_;
    }
    return T_SUCCESS;
  }

  if (hdr.abi_version_ != THINKER_MOSS_RESOURCE_ABI_VERSION ||
      hdr.total_size_ > size ||
      hdr.header_size_ < THINKER_MOSS_RESOURCE_HEADER_SIZE ||
      !moss_range_valid(size, hdr.weight_offset_, hdr.weight_size_) ||
      !moss_range_valid(size, hdr.getter_offset_, hdr.getter_size_) ||
      !moss_range_valid(size, hdr.model_name_offset_,
                        hdr.model_name_size_) ||
      !moss_range_valid(size, hdr.target_arch_offset_,
                        hdr.target_arch_size_)) {
    return T_ERR_RES_INCOMPLETE;
  }

  getter_name = moss_resource_string(res, size, hdr.getter_offset_,
                                     hdr.getter_size_);
  model_name = moss_resource_string(res, size, hdr.model_name_offset_,
                                    hdr.model_name_size_);
  target_arch = moss_resource_string(res, size, hdr.target_arch_offset_,
                                     hdr.target_arch_size_);
  if (getter_name == NULL || model_name == NULL || target_arch == NULL) {
    return T_ERR_RES_INCOMPLETE;
  }

  slot->legacy_model_ = thinker_moss_get_registered_model(getter_name,
                                                          model_name);
  legacy = (tMossLegacyModel *)slot->legacy_model_;
  if (legacy == NULL || legacy->entry_func_ == NULL || legacy->num_input_ < 0 ||
      legacy->num_output_ < 0 ||
      legacy->num_arg_ < legacy->num_input_ + legacy->num_output_) {
    return T_ERR_RES_INCOMPLETE;
  }

  if (legacy->num_input_ > THINKER_MOSS_MAX_IO_TENSORS ||
      legacy->num_output_ > THINKER_MOSS_MAX_IO_TENSORS) {
    return T_ERR_NO_IMPLEMENTED;
  }
  memset(slot->input_infos_, 0, sizeof(slot->input_infos_));
  memset(slot->output_infos_, 0, sizeof(slot->output_infos_));
  for (int32_t i = 0; i < legacy->num_input_; ++i) {
    if (moss_convert_legacy_tensor_info(slot->legacy_model_, 1, i,
                                        &slot->input_infos_[i]) != 0) {
      return T_ERR_RES_INCOMPLETE;
    }
  }
  for (int32_t i = 0; i < legacy->num_output_; ++i) {
    if (moss_convert_legacy_tensor_info(slot->legacy_model_, 0, i,
                                        &slot->output_infos_[i]) != 0) {
      return T_ERR_RES_INCOMPLETE;
    }
  }

  slot->desc_.abi_version = MOSS_MODEL_ABI_VERSION;
  slot->desc_.target_arch = target_arch;
  slot->desc_.model_name = model_name;
  slot->desc_.workspace_size = legacy->space_size_;
  slot->desc_.num_inputs = legacy->num_input_;
  slot->desc_.num_outputs = legacy->num_output_;
  slot->desc_.input_infos = slot->input_infos_;
  slot->desc_.output_infos = slot->output_infos_;
  slot->desc_.num_args = legacy->num_arg_;
  slot->desc_.workspace_arg_idx = (int32_t)legacy->workspace_arg_idx_;
  slot->desc_.weight_arg_idx = (int32_t)legacy->weight_arg_idx_;
  slot->desc_.entry_func = (MossKernelFunc)legacy->entry_func_;
  slot->desc_.global_workspace_size = 0;
  slot->desc_.global_workspace_arg_idx = -1;
  slot->weight_ptr_ = (addr_type)(res + hdr.weight_offset_);

  if (desc != NULL) {
    *desc = &slot->desc_;
  }
  if (weight_ptr != NULL) {
    *weight_ptr = slot->weight_ptr_;
  }
  return T_SUCCESS;
}

static uint32_t moss_align16_u32(uint64_t value) {
  return (uint32_t)ALIGN16(value);
}

static tDType moss_to_thinker_dtype(MossDtype dtype) {
  switch (dtype) {
    case MOSS_DTYPE_I4:
      return Int4;
    case MOSS_DTYPE_I8:
      return Int8;
    case MOSS_DTYPE_I16:
      return Int16;
    case MOSS_DTYPE_I32:
      return Int32;
    case MOSS_DTYPE_I64:
      return Int64;
    case MOSS_DTYPE_F16:
      return Float16;
    case MOSS_DTYPE_F32:
      return Float32;
    default:
      return DTypeUndefined;
  }
}

static uint32_t moss_dtype_size(MossDtype dtype) {
  switch (dtype) {
    case MOSS_DTYPE_I4:
      return 0;
    case MOSS_DTYPE_I8:
      return 1;
    case MOSS_DTYPE_I16:
    case MOSS_DTYPE_F16:
    case MOSS_DTYPE_BF16:
      return 2;
    case MOSS_DTYPE_I32:
    case MOSS_DTYPE_F32:
      return 4;
    case MOSS_DTYPE_I64:
      return 8;
    default:
      return 0;
  }
}

static uint64_t moss_tensor_elem_count(const MossTensorInfo *info) {
  uint64_t elem_count = 1;
  if (info == NULL || info->rank <= 0) {
    return 0;
  }
  for (int32_t i = 0; i < info->rank && i < MOSS_MAX_RANK; ++i) {
    elem_count *= (uint64_t)info->shape[i];
  }
  return elem_count;
}

static uint32_t moss_tensor_byte_size(const MossTensorInfo *info) {
  if (info == NULL) {
    return 0;
  }
  uint64_t elem_count = moss_tensor_elem_count(info);
  if (info->dtype == MOSS_DTYPE_I4) {
    return moss_align16_u32((elem_count + 1u) / 2u);
  }
  return moss_align16_u32(elem_count * moss_dtype_size(info->dtype));
}

static tShape moss_to_thinker_shape(const MossTensorInfo *info) {
  tShape shape = {0};
  if (info == NULL || info->rank < 0) {
    return shape;
  }
  shape.ndim_ = (uint32_t)((info->rank > THINKER_MOSS_MAX_RANK)
                               ? THINKER_MOSS_MAX_RANK
                               : info->rank);
  for (uint32_t i = 0; i < shape.ndim_; ++i) {
    shape.dims_[i] = (uint32_t)info->shape[i];
  }
  return shape;
}

static void moss_fill_legacy_tensor(const MossTensorInfo *info,
                                    MossTensor *tensor) {
  if (info == NULL || tensor == NULL) {
    return;
  }
  memset(tensor, 0, sizeof(*tensor));
  tensor->name = (char *)info->name;
  tensor->memorytype_datatype =
      (int8_t)(((int32_t)info->mem_space << 4) | (int32_t)info->dtype);
  int32_t rank = info->rank;
  if (rank < 0) {
    rank = 0;
  } else if (rank > THINKER_MOSS_MAX_RANK) {
    rank = THINKER_MOSS_MAX_RANK;
  }
  tensor->dim = (int8_t)rank;
  for (int32_t i = 0; i < rank; ++i) {
    tensor->shape[i] = (int16_t)info->shape[i];
  }
  tensor->scale = info->scale;
}

static int32_t moss_valid_arg_idx(const MossModelDesc *desc, int32_t idx) {
  return desc != NULL && idx >= 0 && idx < desc->num_args;
}

static void *moss_create_legacy_model(const MossModelDesc *desc,
                                      tMossLegacyModel *model, void **args) {
  if (desc == NULL || model == NULL || desc->num_args < 0 ||
      desc->num_args < desc->num_inputs + desc->num_outputs ||
      desc->global_workspace_size > 0 ||
      (desc->num_args > 0 && args == NULL) ||
      (desc->weight_arg_idx >= 0 &&
       !moss_valid_arg_idx(desc, desc->weight_arg_idx)) ||
      (desc->workspace_size > 0 &&
       !moss_valid_arg_idx(desc, desc->workspace_arg_idx))) {
    return NULL;
  }

  memset(model, 0, sizeof(*model));
  if (desc->num_args > 0) {
    memset(args, 0, (size_t)desc->num_args * sizeof(void *));
  }
  model->space_size_ = desc->workspace_size;
  model->workspace_arg_idx_ = (uint32_t)desc->workspace_arg_idx;
  model->weight_arg_idx_ = (uint32_t)desc->weight_arg_idx;
  model->num_input_ = desc->num_inputs;
  model->num_output_ = desc->num_outputs;
  model->num_arg_ = desc->num_args;
  model->args_ = args;
  model->entry_func_ = (int32_t(*)(uint32_t, void **))desc->entry_func;
  return model;
}

static void moss_destroy_legacy_model(void *model) {
  if (model == NULL) {
    return;
  }
  tMossLegacyModel *legacy_model = (tMossLegacyModel *)model;
  if (legacy_model->args_ != NULL && legacy_model->num_arg_ > 0) {
    memset(legacy_model->args_, 0,
           (size_t)legacy_model->num_arg_ * sizeof(void *));
  }
  memset(legacy_model, 0, sizeof(*legacy_model));
}

static void moss_fill_tdata(const MossTensorInfo *info, void *dptr,
                            tData *data) {
  if (data == NULL || info == NULL) {
    return;
  }
  memset(data, 0, sizeof(*data));
  data->dptr_ = dptr;
  data->dev_type_ = (uint16_t)info->mem_space;
  data->dtype_ = (uint16_t)moss_to_thinker_dtype(info->dtype);
  data->zero_ = (uint16_t)info->zero_point;
  data->scale_ = info->scale;
  data->shape_ = moss_to_thinker_shape(info);
}

static int32_t moss_model_inst_size(const MossModelDesc *desc) {
  int32_t size = 0;
  int32_t io_count = desc->num_inputs + desc->num_outputs;
  size += ALIGN16(sizeof(tMossModel));
  size += ALIGN16(sizeof(MossModelDesc));
  size += ALIGN16(desc->num_inputs * sizeof(MossTensorInfo));
  size += ALIGN16(desc->num_outputs * sizeof(MossTensorInfo));
  size += ALIGN16(desc->num_inputs * sizeof(tData));
  size += ALIGN16(desc->num_outputs * sizeof(tData));
  size += ALIGN16(desc->num_inputs * sizeof(tShape));
  size += ALIGN16(desc->num_outputs * sizeof(tShape));
  size += ALIGN16(io_count * MOSS_MAX_NAME);
  return size;
}

static int32_t moss_exec_inst_size(const MossModelDesc *desc) {
  int32_t size = 0;
  size += ALIGN16(sizeof(tMossExecInst));
  size += ALIGN16(desc->num_inputs * sizeof(MossTensor));
  size += ALIGN16(desc->num_outputs * sizeof(MossTensor));
  size += ALIGN16(sizeof(tMossLegacyModel));
  size += ALIGN16(desc->num_args * sizeof(void *));
  return size;
}

static tStatus moss_validate_desc(const int8_t *res, uint64_t size,
                                  const MossModelDesc **desc,
                                  addr_type *embedded_weight_ptr) {
  const MossModelDesc *model_desc = NULL;
  if (embedded_weight_ptr != NULL) {
    *embedded_weight_ptr = 0;
  }
  if (res == NULL || size == 0) {
    return T_ERR_RES_MISSING;
  }

  if (moss_is_packed_resource(res, size)) {
    return moss_resolve_packed_resource(res, size, desc, embedded_weight_ptr);
  }

  if (size < sizeof(MossModelDesc)) {
    return T_ERR_RES_MISSING;
  }
  model_desc = (const MossModelDesc *)res;
  if (model_desc->abi_version != MOSS_MODEL_ABI_VERSION) {
    return T_ERR_RES_INCOMPLETE;
  }
  if (model_desc->num_inputs < 0 || model_desc->num_outputs < 0 ||
      model_desc->num_args < model_desc->num_inputs + model_desc->num_outputs ||
      model_desc->input_infos == NULL || model_desc->output_infos == NULL ||
      model_desc->entry_func == NULL || model_desc->global_workspace_size > 0) {
    return T_ERR_RES_INCOMPLETE;
  }
  if (desc != NULL) {
    *desc = model_desc;
  }
  return T_SUCCESS;
}

static const MossTensorInfo *moss_get_input_info(const tMossModel *model,
                                                 int32_t idx) {
  return &model->desc_->input_infos[idx];
}

static const MossTensorInfo *moss_get_output_info(const tMossModel *model,
                                                  int32_t idx) {
  return &model->desc_->output_infos[idx];
}

static addr_type moss_find_memory(const tMemory *memory_list,
                                  int32_t num_memory, uint8_t mem_type,
                                  uint8_t dev_type, uint32_t min_size) {
  for (int32_t i = 0; i < num_memory; ++i) {
    if (memory_list[i].mem_type_ == mem_type &&
        memory_list[i].dev_type_ == dev_type &&
        memory_list[i].size_ >= min_size) {
      return memory_list[i].dptr_;
    }
  }
  return 0;
}

const char *tGetVersion(const int8_t index) {
  return (1 == index) ? VENUS_VERSION : THINKER_VERSION;
}

tStatus tInitialize() { return T_SUCCESS; }

tStatus tUninitialize() { return T_SUCCESS; }

tStatus tGetMemoryPlan(tMemory *memory_list, int32_t *num_memory,
                       const int8_t *res, const uint64_t size) {
  const MossModelDesc *desc = NULL;
  addr_type embedded_weight_ptr = 0;
  tStatus ret = moss_validate_desc(res, size, &desc, &embedded_weight_ptr);
  if (ret != T_SUCCESS) {
    return ret;
  }
  if (memory_list == NULL || num_memory == NULL) {
    return T_ERR_INVALID_PARA;
  }

  int32_t num = 0;
  memory_list[num++] = (tMemory){(uint32_t)moss_model_inst_size(desc), 1, 0, 0};
  memory_list[num++] = (tMemory){(uint32_t)moss_exec_inst_size(desc), 1, 1, 0};

  if (desc->weight_arg_idx >= 0) {
    memory_list[num++] = (tMemory){0, MOSS_MEM_FLASH, 2, embedded_weight_ptr};
  }
  if (desc->workspace_size > 0) {
    memory_list[num++] =
        (tMemory){moss_align16_u32(desc->workspace_size), MOSS_MEM_SHARED, 3, 0};
  }

  uint64_t io_sizes[MOSS_MEM_HOST + 1] = {0};
  for (int32_t i = 0; i < desc->num_inputs; ++i) {
    const MossTensorInfo *info = &desc->input_infos[i];
    if (info->mem_space <= MOSS_MEM_HOST) {
      io_sizes[info->mem_space] += moss_tensor_byte_size(info);
    }
  }
  for (int32_t i = 0; i < desc->num_outputs; ++i) {
    const MossTensorInfo *info = &desc->output_infos[i];
    if (info->mem_space <= MOSS_MEM_HOST) {
      io_sizes[info->mem_space] += moss_tensor_byte_size(info);
    }
  }
  for (int32_t i = 0; i <= MOSS_MEM_HOST; ++i) {
    if (io_sizes[i] > 0) {
      memory_list[num++] =
          (tMemory){moss_align16_u32(io_sizes[i]), (uint8_t)i,
                    MOSS_IO_MEMORY_TYPE, 0};
    }
  }

  *num_memory = num;
  return T_SUCCESS;
}

tStatus tModelInit(tModelHandle *hdl, const int8_t *res, const uint64_t size,
                   const tMemory *memory_list, const int32_t num_memory) {
  const MossModelDesc *desc = NULL;
  addr_type embedded_weight_ptr = 0;
  tStatus ret = moss_validate_desc(res, size, &desc, &embedded_weight_ptr);
  if (ret != T_SUCCESS) {
    return ret;
  }
  if (hdl == NULL || memory_list == NULL || num_memory <= 0) {
    return T_ERR_INVALID_PARA;
  }

  tMemory inst_memory = {0};
  for (int32_t i = 0; i < num_memory; ++i) {
    if (memory_list[i].mem_type_ == 0) {
      if (memory_list[i].size_ < (uint32_t)moss_model_inst_size(desc)) {
        return T_ERR_INVALID_PARA;
      }
      inst_memory = memory_list[i];
      break;
    }
  }
  if (inst_memory.dptr_ == 0) {
    return T_ERR_INVALID_PARA;
  }

  addr_type weight_ptr = embedded_weight_ptr;
  for (int32_t i = 0; i < num_memory; ++i) {
    if (memory_list[i].mem_type_ == 2) {
      weight_ptr = memory_list[i].dptr_;
      break;
    }
  }
  if (desc->weight_arg_idx >= 0 && weight_ptr == 0) {
    return T_ERR_INVALID_PARA;
  }

  int8_t *ptr = (int8_t *)inst_memory.dptr_;
  tMossModel *model = (tMossModel *)ptr;
  memset(model, 0, sizeof(*model));
  model->flag_ = THINKER_INST_FLAG;
  model->inst_memory_ = inst_memory;
  model->weight_ptr_ = weight_ptr;
  model->num_input_ = (uint16_t)desc->num_inputs;
  model->num_output_ = (uint16_t)desc->num_outputs;
  model->io_name_len_ = MOSS_MAX_NAME;
  ptr += ALIGN16(sizeof(tMossModel));

  model->desc_storage_ = (MossModelDesc *)ptr;
  ptr += ALIGN16(sizeof(MossModelDesc));
  model->input_infos_ = (MossTensorInfo *)ptr;
  ptr += ALIGN16(desc->num_inputs * sizeof(MossTensorInfo));
  model->output_infos_ = (MossTensorInfo *)ptr;
  ptr += ALIGN16(desc->num_outputs * sizeof(MossTensorInfo));
  *model->desc_storage_ = *desc;
  if (desc->num_inputs > 0) {
    memcpy(model->input_infos_, desc->input_infos,
           (size_t)desc->num_inputs * sizeof(MossTensorInfo));
  }
  if (desc->num_outputs > 0) {
    memcpy(model->output_infos_, desc->output_infos,
           (size_t)desc->num_outputs * sizeof(MossTensorInfo));
  }
  model->desc_storage_->input_infos = model->input_infos_;
  model->desc_storage_->output_infos = model->output_infos_;
  model->desc_ = model->desc_storage_;

  model->input_info_ = (tData *)ptr;
  ptr += ALIGN16(desc->num_inputs * sizeof(tData));
  model->output_info_ = (tData *)ptr;
  ptr += ALIGN16(desc->num_outputs * sizeof(tData));
  model->input_shape_ = (tShape *)ptr;
  ptr += ALIGN16(desc->num_inputs * sizeof(tShape));
  model->output_shape_ = (tShape *)ptr;
  ptr += ALIGN16(desc->num_outputs * sizeof(tShape));
  model->io_names_ = (char *)ptr;

  for (int32_t i = 0; i < desc->num_inputs; ++i) {
    const MossTensorInfo *info = &desc->input_infos[i];
    moss_fill_tdata(info, NULL, &model->input_info_[i]);
    model->input_shape_[i] = model->input_info_[i].shape_;
    char *name = model->io_names_ + i * model->io_name_len_;
    memset(name, 0, model->io_name_len_);
    strncpy(name, info->name, model->io_name_len_ - 1);
  }
  for (int32_t i = 0; i < desc->num_outputs; ++i) {
    const MossTensorInfo *info = &desc->output_infos[i];
    moss_fill_tdata(info, NULL, &model->output_info_[i]);
    model->output_shape_[i] = model->output_info_[i].shape_;
    char *name =
        model->io_names_ + (desc->num_inputs + i) * model->io_name_len_;
    memset(name, 0, model->io_name_len_);
    strncpy(name, info->name, model->io_name_len_ - 1);
  }

  *hdl = ~((tModelHandle)model);
  return T_SUCCESS;
}

tStatus tModelFini(tModelHandle hdl) {
  if (hdl == 0) {
    return T_ERR_INVALID_INST;
  }
  tMossModel *model = (tMossModel *)~hdl;
  if (model == NULL || model->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }
  return T_SUCCESS;
}

int32_t tGetInputCount(const tModelHandle hdl) {
  if (hdl == 0) {
    return T_ERR_INVALID_INST;
  }
  tMossModel *model = (tMossModel *)~hdl;
  if (model == NULL || model->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }
  return model->num_input_;
}

tStatus tGetInputInfo(const tExecHandle hdl, const int32_t idx,
                      tData *input) {
  if (hdl == 0) {
    return T_ERR_INVALID_INST;
  }
  tMossExecInst *inst = (tMossExecInst *)~hdl;
  if (inst == NULL || inst->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }
  if (input == NULL) {
    return T_ERR_INVALID_DATA;
  }
  if (idx < 0 || idx >= inst->model_->num_input_) {
    return T_ERR_INDEX_OF_BOUND;
  }
  *input = inst->model_->input_info_[idx];
  input->dptr_ = inst->inputs_[idx].dataptr;
  return T_SUCCESS;
}

const char *tGetInputName(const tModelHandle hdl, const int32_t idx) {
  if (hdl == 0) {
    return NULL;
  }
  tMossModel *model = (tMossModel *)~hdl;
  if (model == NULL || model->flag_ != THINKER_INST_FLAG || idx < 0 ||
      idx >= model->num_input_) {
    return NULL;
  }
  return model->io_names_ + idx * model->io_name_len_;
}

int32_t tGetOutputCount(const tModelHandle hdl) {
  if (hdl == 0) {
    return T_ERR_INVALID_INST;
  }
  tMossModel *model = (tMossModel *)~hdl;
  if (model == NULL || model->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }
  return model->num_output_;
}

const char *tGetOutputName(const tModelHandle hdl, const int32_t idx) {
  if (hdl == 0) {
    return NULL;
  }
  tMossModel *model = (tMossModel *)~hdl;
  if (model == NULL || model->flag_ != THINKER_INST_FLAG || idx < 0 ||
      idx >= model->num_output_) {
    return NULL;
  }
  return model->io_names_ + (model->num_input_ + idx) * model->io_name_len_;
}

tDType tGetInputDataType(const tModelHandle hdl, const int32_t idx) {
  if (hdl == 0) {
    return DTypeUndefined;
  }
  tMossModel *model = (tMossModel *)~hdl;
  if (model == NULL || model->flag_ != THINKER_INST_FLAG || idx < 0 ||
      idx >= model->num_input_) {
    return DTypeUndefined;
  }
  return (tDType)model->input_info_[idx].dtype_;
}

tDType tGetOutputDataType(const tModelHandle hdl, const int32_t idx) {
  if (hdl == 0) {
    return DTypeUndefined;
  }
  tMossModel *model = (tMossModel *)~hdl;
  if (model == NULL || model->flag_ != THINKER_INST_FLAG || idx < 0 ||
      idx >= model->num_output_) {
    return DTypeUndefined;
  }
  return (tDType)model->output_info_[idx].dtype_;
}

tShape tGetInputShape(const tModelHandle hdl, const int32_t idx) {
  tShape shape = {0};
  if (hdl == 0) {
    return shape;
  }
  tMossModel *model = (tMossModel *)~hdl;
  if (model == NULL || model->flag_ != THINKER_INST_FLAG || idx < 0 ||
      idx >= model->num_input_) {
    return shape;
  }
  return model->input_shape_[idx];
}

tShape tGetOutputShape(const tModelHandle hdl, const int32_t idx) {
  tShape shape = {0};
  if (hdl == 0) {
    return shape;
  }
  tMossModel *model = (tMossModel *)~hdl;
  if (model == NULL || model->flag_ != THINKER_INST_FLAG || idx < 0 ||
      idx >= model->num_output_) {
    return shape;
  }
  return model->output_shape_[idx];
}

tStatus tCreateExecutor(const tModelHandle model_hdl, tExecHandle *hdl,
                        const tMemory *memory_list, const int32_t num_memory) {
  if (hdl == NULL || memory_list == NULL || num_memory <= 0 || model_hdl == 0) {
    return T_ERR_INVALID_PARA;
  }
  tMossModel *model = (tMossModel *)~model_hdl;
  if (model == NULL || model->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }

  tMemory inst_memory = {0};
  int32_t inst_size = moss_exec_inst_size(model->desc_);
  for (int32_t i = 0; i < num_memory; ++i) {
    if (memory_list[i].mem_type_ == 1) {
      if (memory_list[i].size_ < (uint32_t)inst_size) {
        return T_ERR_INVALID_PARA;
      }
      inst_memory = memory_list[i];
      break;
    }
  }
  if (inst_memory.dptr_ == 0) {
    return T_ERR_INVALID_PARA;
  }

  int8_t *ptr = (int8_t *)inst_memory.dptr_;
  tMossExecInst *inst = (tMossExecInst *)ptr;
  memset(inst, 0, sizeof(*inst));
  inst->flag_ = THINKER_INST_FLAG;
  inst->inst_memory_ = inst_memory;
  inst->model_ = model;
  ptr += ALIGN16(sizeof(tMossExecInst));
  inst->inputs_ = (MossTensor *)ptr;
  ptr += ALIGN16(model->num_input_ * sizeof(MossTensor));
  inst->outputs_ = (MossTensor *)ptr;
  ptr += ALIGN16(model->num_output_ * sizeof(MossTensor));
  tMossLegacyModel *legacy_model = (tMossLegacyModel *)ptr;
  ptr += ALIGN16(sizeof(tMossLegacyModel));
  void **legacy_args = (void **)ptr;

  inst->legacy_model_ = moss_create_legacy_model(model->desc_, legacy_model,
                                                 legacy_args);
  if (inst->legacy_model_ == NULL) {
    return T_ERR_INVALID_INST;
  }

  if (model->desc_->weight_arg_idx >= 0 &&
      mSetModuleWeightBasePtr(inst->legacy_model_,
                              (int8_t *)model->weight_ptr_) != 0) {
    tReleaseExecutor(~((tExecHandle)inst));
    return T_ERR_INVALID_PARA;
  }

  if (model->desc_->workspace_size > 0) {
    addr_type workspace =
        moss_find_memory(memory_list, num_memory, 3, MOSS_MEM_SHARED,
                         moss_align16_u32(model->desc_->workspace_size));
    if (workspace == 0 ||
        mSetModuleWorkSpacePtr(inst->legacy_model_, (int8_t *)workspace) != 0) {
      tReleaseExecutor(~((tExecHandle)inst));
      return T_ERR_INVALID_PARA;
    }
  }

  uint64_t io_offsets[MOSS_MEM_HOST + 1] = {0};
  for (int32_t i = 0; i < model->num_input_; ++i) {
    const MossTensorInfo *info = moss_get_input_info(model, i);
    moss_fill_legacy_tensor(info, &inst->inputs_[i]);
    if (info->mem_space > MOSS_MEM_HOST) {
      tReleaseExecutor(~((tExecHandle)inst));
      return T_ERR_INVALID_DATA;
    }
    uint32_t bytes = moss_tensor_byte_size(info);
    addr_type base = moss_find_memory(memory_list, num_memory,
                                      MOSS_IO_MEMORY_TYPE, info->mem_space,
                                      (uint32_t)(io_offsets[info->mem_space] + bytes));
    if (base == 0) {
      tReleaseExecutor(~((tExecHandle)inst));
      return T_ERR_INVALID_PARA;
    }
    inst->inputs_[i].dataptr = (int8_t *)(base + io_offsets[info->mem_space]);
    io_offsets[info->mem_space] += bytes;
    if (mSetModuleInput(inst->legacy_model_, i, &inst->inputs_[i]) != 0) {
      tReleaseExecutor(~((tExecHandle)inst));
      return T_ERR_INVALID_DATA;
    }
  }
  for (int32_t i = 0; i < model->num_output_; ++i) {
    const MossTensorInfo *info = moss_get_output_info(model, i);
    moss_fill_legacy_tensor(info, &inst->outputs_[i]);
    if (info->mem_space > MOSS_MEM_HOST) {
      tReleaseExecutor(~((tExecHandle)inst));
      return T_ERR_INVALID_DATA;
    }
    uint32_t bytes = moss_tensor_byte_size(info);
    addr_type base = moss_find_memory(memory_list, num_memory,
                                      MOSS_IO_MEMORY_TYPE, info->mem_space,
                                      (uint32_t)(io_offsets[info->mem_space] + bytes));
    if (base == 0) {
      tReleaseExecutor(~((tExecHandle)inst));
      return T_ERR_INVALID_PARA;
    }
    inst->outputs_[i].dataptr = (int8_t *)(base + io_offsets[info->mem_space]);
    io_offsets[info->mem_space] += bytes;
    if (mSetModuleOutput(inst->legacy_model_, i, &inst->outputs_[i]) != 0) {
      tReleaseExecutor(~((tExecHandle)inst));
      return T_ERR_INVALID_DATA;
    }
  }

  *hdl = ~((tExecHandle)inst);
  return T_SUCCESS;
}

tStatus tReleaseExecutor(tExecHandle hdl) {
  if (hdl == 0) {
    return T_ERR_INVALID_INST;
  }
  tMossExecInst *inst = (tMossExecInst *)~hdl;
  if (inst == NULL || inst->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }
  moss_destroy_legacy_model(inst->legacy_model_);
  inst->legacy_model_ = NULL;
  return T_SUCCESS;
}

tStatus tSetInput(const tExecHandle hdl, const int32_t idx,
                  const tData *input) {
  if (hdl == 0) {
    return T_ERR_INVALID_INST;
  }
  tMossExecInst *inst = (tMossExecInst *)~hdl;
  if (inst == NULL || inst->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }
  if (idx < 0 || idx >= inst->model_->num_input_) {
    return T_ERR_INDEX_OF_BOUND;
  }
  if (input == NULL || input->dptr_ == NULL) {
    return T_ERR_INVALID_DATA;
  }

  const MossTensorInfo *info = moss_get_input_info(inst->model_, idx);
  tDType dtype = moss_to_thinker_dtype(info->dtype);
  if (input->dtype_ != (uint16_t)dtype) {
    return T_ERR_INVALID_DATATYPE;
  }
  tShape max_shape = moss_to_thinker_shape(info);
  if (input->shape_.ndim_ != max_shape.ndim_) {
    return T_ERR_INVALID_DATA;
  }
  for (uint32_t i = 0; i < input->shape_.ndim_; ++i) {
    if (input->shape_.dims_[i] > max_shape.dims_[i]) {
      return T_ERR_INVALID_DATA;
    }
    inst->inputs_[idx].shape[i] = (int16_t)input->shape_.dims_[i];
  }

  uint32_t bytes = moss_tensor_byte_size(info);
  memcpy(inst->inputs_[idx].dataptr, input->dptr_, bytes);
  return T_SUCCESS;
}

tStatus tSetInputByName(const tExecHandle hdl, const char *name,
                        const tData *input) {
  if (hdl == 0) {
    return T_ERR_INVALID_INST;
  }
  tMossExecInst *inst = (tMossExecInst *)~hdl;
  if (inst == NULL || inst->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }
  if (name == NULL || input == NULL) {
    return T_ERR_INVALID_PARA;
  }
  for (int32_t i = 0; i < inst->model_->num_input_; ++i) {
    if (strcmp(tGetInputName(~((tModelHandle)inst->model_), i), name) == 0) {
      return tSetInput(hdl, i, input);
    }
  }
  return T_ERR_INVALID_PARA;
}

tStatus tGetOutput(const tExecHandle hdl, const int32_t idx, tData *output) {
  if (hdl == 0) {
    return T_ERR_INVALID_INST;
  }
  tMossExecInst *inst = (tMossExecInst *)~hdl;
  if (inst == NULL || inst->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }
  if (output == NULL) {
    return T_ERR_INVALID_PARA;
  }
  if (idx < 0 || idx >= inst->model_->num_output_) {
    return T_ERR_INDEX_OF_BOUND;
  }
  *output = inst->model_->output_info_[idx];
  output->dptr_ = inst->outputs_[idx].dataptr;
  return T_SUCCESS;
}

tStatus tGetOutputByName(const tExecHandle hdl, const char *name,
                         tData *output) {
  if (hdl == 0) {
    return T_ERR_INVALID_INST;
  }
  tMossExecInst *inst = (tMossExecInst *)~hdl;
  if (inst == NULL || inst->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }
  if (name == NULL || output == NULL) {
    return T_ERR_INVALID_PARA;
  }
  for (int32_t i = 0; i < inst->model_->num_output_; ++i) {
    if (strcmp(tGetOutputName(~((tModelHandle)inst->model_), i), name) == 0) {
      return tGetOutput(hdl, i, output);
    }
  }
  return T_ERR_INVALID_PARA;
}

tStatus tForward(const tExecHandle hdl) {
  if (hdl == 0) {
    return T_ERR_INVALID_INST;
  }
  tMossExecInst *inst = (tMossExecInst *)~hdl;
  if (inst == NULL || inst->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }
  if (inst->force_stop_flag == T_FORCE_STOP_VALUE) {
    return T_FORCE_STOP_VALUE;
  }
  return (mModelForward(inst->legacy_model_) == 0) ? T_SUCCESS : T_ERR_FAIL;
}

tStatus tUpdateShape(tExecHandle hdl, const char **axis_names,
                     const uint32_t *axis_sizes, int32_t num) {
  (void)axis_names;
  (void)axis_sizes;
  if (hdl == 0) {
    return T_ERR_INVALID_INST;
  }
  if (num < 0) {
    return T_ERR_INVALID_PARA;
  }
  tMossExecInst *inst = (tMossExecInst *)~hdl;
  if (inst == NULL || inst->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }
  return T_SUCCESS;
}

tStatus tExecutorStart(tExecHandle hdl) {
  if (hdl == 0) {
    return T_ERR_INVALID_INST;
  }
  tMossExecInst *inst = (tMossExecInst *)~hdl;
  if (inst == NULL || inst->flag_ != THINKER_INST_FLAG) {
    return T_ERR_INVALID_INST;
  }
  inst->force_stop_flag = 0;
  return T_SUCCESS;
}

tStatus tExecutorStop(tExecHandle hdl) {
  if (hdl == 0) {
    return T_ERR_INVALID_INST;
  }
  tMossExecInst *inst = (tMossExecInst *)~hdl;
  if (inst == NULL || inst->flag_ != THINKER_INST_FLAG) {
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
  g_api.tUpdateShape = tUpdateShape;
  g_api.tGetOutput = tGetOutput;
  g_api.tGetOutputByName = tGetOutputByName;
  g_api.tForward = tForward;
  g_api.tExecutorStart = tExecutorStart;
  g_api.tExecutorStop = tExecutorStop;
  g_api.reserve[0] = NULL;
  g_api.reserve[1] = NULL;
  g_api.reserve[2] = NULL;
  return &g_api;
}

#endif /* THINKER_USE_MOSS */
