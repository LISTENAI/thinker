#include "shape_infer.h"
#include <string.h>
#include "thinker_log.h"


#define ALIGN16(n) ((n+15)&~15)

static tStatus tScalarGraphInit(const char *ptr, tScalarGraph *graph)
{
  graph->num_input_ = *(int32_t *)ptr;
  ptr += sizeof(int32_t);
  graph->num_output_ = *(int32_t *)ptr;
  ptr += sizeof(int32_t);
  graph->num_node_ = *(int32_t *)ptr;
  ptr += sizeof(int32_t);
  graph->num_scalars_ = *(int32_t *)ptr;
  ptr += sizeof(int32_t);
  if (graph->num_scalars_ == 0) return T_SUCCESS;
  // deserialize input ids
  graph->inputs_ = (int32_t *)ptr;
  ptr += graph->num_input_ * sizeof(int32_t);
  // deserialize output ids
  graph->outputs_ = (int32_t *)ptr;
  ptr += graph->num_output_ * sizeof(int32_t);
  // deserialize input names
  graph->name_max_len = 32;
  graph->input_names_ = (char *)ptr;
  ptr += graph->num_input_ * 32;
  // deserialize scalars
  graph->scalars_ = (double *)ptr;
  ptr += graph->num_scalars_ * sizeof(double);
  // deserialize op metas
  graph->node_metas_ = (int32_t *)ptr;
  ptr += graph->num_node_ * sizeof(int32_t);
  // deserialize op nodes
  graph->nodes_ = (int32_t *)ptr;
  return T_SUCCESS;
}

tStatus tShapeInferInit(const char *res, tShapeInfer *shape_infer)
{
  tStatus ret = T_SUCCESS;
  const tShapeInferHdr shape_hdr = *(tShapeInferHdr *)(res);
  // calculate workspace size
  uint64_t total_size    = ALIGN16(sizeof(tScalarGraph));
  total_size             += ALIGN16(shape_hdr.graph_size_);
  total_size             += ALIGN16(shape_hdr.num_id_pair_ * sizeof(tTenDimPair));
  total_size             += ALIGN16(shape_hdr.num_dy_axis_ * sizeof(tDyAxisInfo));

  // xMemory inst_memory = { total_size, 0, 0, 0};

  // xMemoryMalloc(&inst_memory);
  // char *ptr                 = (char*)(inst_memory.dptr_);
  // shape_infer->inst_memory_ = inst_memory;
  assert(shape_infer->inst_memory_.size_ >= total_size);
  char *ptr = (char*)shape_infer->inst_memory_.dptr_;
  // init scalar graph
  shape_infer->graph_ = (tScalarGraph *)ptr;
  ptr += ALIGN16(sizeof(tScalarGraph));
  memcpy(ptr, res + shape_hdr.graph_offset_, shape_hdr.graph_size_);
  ret = tScalarGraphInit(ptr, shape_infer->graph_);
  if (ret != T_SUCCESS) return ret;
  ptr += ALIGN16(shape_hdr.graph_size_);
  // init dy id pairs
  shape_infer->tid_pairs_   = (tTenDimPair *)ptr;
  shape_infer->num_id_pair_ = shape_hdr.num_id_pair_;
  memcpy(ptr, res+shape_hdr.id_pair_offset_, shape_hdr.num_id_pair_*sizeof(tTenDimPair));
  ptr += ALIGN16(shape_hdr.num_id_pair_ * sizeof(tTenDimPair));
  // init dy axis info
  shape_infer->dynamic_axis_ = (tDyAxisInfo *)ptr;
  shape_infer->num_dy_axis_  = shape_hdr.num_dy_axis_;
  memcpy(ptr, res+shape_hdr.dy_axis_offset_, shape_hdr.num_dy_axis_*sizeof(tDyAxisInfo));
  ptr += ALIGN16(shape_hdr.num_dy_axis_ * sizeof(tDyAxisInfo));
  return ret;
}

tStatus tShapeInferFini(tShapeInfer *shape_infer)
{
  return T_SUCCESS;
}

static tStatus tScalarGraphForward(tScalarGraph *graph, double *scalars)
{
  // scalar graph inference
  int32_t      ret = 0;
  int32_t      *op_ptr = graph->nodes_;
  ScalarOpType *op_type = NULL;
  int32_t      *input_ids = NULL;
  int32_t      *output_ids = NULL;
  int32_t      node_io_ids[8];
  int          input_num = 0;
  for (int i = 0, cnt = 0; i < graph->num_node_; ++i) {
    input_num = graph->node_metas_[i] - 2;
    CHECK_LE(input_num, 8);
    op_type    = (ScalarOpType *)(op_ptr++);
    input_ids  = (int32_t *)op_ptr;
    op_ptr     += input_num;
    output_ids = (int32_t *)(op_ptr++);
    for (int index = 0; index < input_num; ++index)
      node_io_ids[index] = input_ids[index];
    node_io_ids[input_num] = output_ids[0];
    ret = ScalarFunc(scalars, op_type, node_io_ids, input_num);
    if (ret != 0) return T_ERR_FAIL;
  }
  return T_SUCCESS;
}

tStatus tSetShapeInferInputByTensors(tShapeInfer *shape_infer, double *scalars, tTensor *tensors)
{
  tScalarGraph *graph = shape_infer->graph_;
  CHECK_LE(graph->num_input_, 8);
  uint8_t set_flag[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  // check all inputs axis size are same on the corresponding dynamic axis.
  for (int i = 0; i < shape_infer->num_dy_axis_; ++i)
  {
    tDyAxisInfo dy_info    = shape_infer->dynamic_axis_[i];
    tTensor     *tensor    = &tensors[dy_info.tensor_id_];
    double    size       = (double)tensor->shape_.dims_[dy_info.dy_dim_id_];
    uint8_t     input_id   = dy_info.scalar_input_id_;
    uint32_t    scalar_id  = graph->inputs_[input_id];
    if (set_flag[input_id] == 1)
    {
      if (scalars[scalar_id] != size) return T_ERR_INVALID_DATA;
    }
    else
    {
      scalars[scalar_id] = size;
      set_flag[input_id] = 1;
    }
  }
  return T_SUCCESS;
}

tStatus tSetShapeInferInputByNames(tShapeInfer *shape_infer, double *scalars, 
                                   const char **axis_names, const uint32_t *axis_sizes, int num)
{
  tScalarGraph *graph = shape_infer->graph_;
  CHECK_LE(num, graph->num_input_);
  uint8_t  match_flag = 0;
  const char *name    = NULL;
  for (int i = 0; i < num; ++i)
  {
    match_flag = 0;
    name = shape_infer->graph_->input_names_;
    for (int j = 0; j < graph->num_input_; ++j)
    {
      if (strcmp(axis_names[i], name) == 0)
      {
        match_flag = 1;
        scalars[graph->inputs_[j]] = (double)axis_sizes[i];
      }
      name += shape_infer->graph_->name_max_len;
    }
    if (match_flag == 0) return T_ERR_INVALID_DATA;
  }
  return T_SUCCESS;
}

tStatus tShapeInferForward(tShapeInfer *shape_infer, double *scalars, tTensor *tensors)
{
  tScalarGraph *graph = shape_infer->graph_;
  tStatus ret = tScalarGraphForward(graph, scalars);
  if (ret != T_SUCCESS) return ret;
  // assign scalar graph outputs to tensors shape by tTenDimPair.
  CHECK_EQ(shape_infer->num_id_pair_, graph->num_output_);
  for (int i = 0; i < shape_infer->num_id_pair_; ++i)
  {
    uint32_t size = (uint32_t)(scalars[graph->outputs_[i]] + 0.5f);
    tTenDimPair pair = shape_infer->tid_pairs_[i];
    tensors[pair.tensor_id_].shape_.dims_[pair.dim_id_] = size;
  }
  return T_SUCCESS;
}