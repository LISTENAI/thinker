#undef __OP__
#define __OP__ Transpose
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_VENUS
#include "./venus/transpose.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/transpose.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/transpose.h"
#endif


// Coalesce adjacent input axes that remain adjacent and ordered in the output.
int32_t merge_transpose_axes(uint32_t *axes, uint32_t *shape, uint32_t *dims) {
    if (axes == NULL || shape == NULL || dims == NULL || *dims < 2 || *dims > 5)
      return T_ERR_INVALID_PARA;

    uint32_t positions[5];
    uint32_t group_ids[5];
    uint32_t merged_shape[5] = {0};
    uint32_t merged_axes[5];
    bool seen[5] = {false};
    for (uint32_t i = 0; i < *dims; ++i) {
      if (axes[i] >= *dims || seen[axes[i]]) return T_ERR_INVALID_PARA;
      seen[axes[i]] = true;
      positions[axes[i]] = i;
    }

    uint32_t group_count = 1;
    group_ids[0] = 0;
    merged_shape[0] = shape[0];
    for (uint32_t axis = 1; axis < *dims; ++axis) {
      if (positions[axis] != positions[axis - 1] + 1) {
        group_count++;
        merged_shape[group_count - 1] = shape[axis];
      } else {
        merged_shape[group_count - 1] *= shape[axis];
      }
      group_ids[axis] = group_count - 1;
    }

    uint32_t output_count = 0;
    for (uint32_t i = 0; i < *dims; ++i) {
      uint32_t group = group_ids[axes[i]];
      if (i == 0 || group != group_ids[axes[i - 1]])
        merged_axes[output_count++] = group;
    }
    if (output_count != group_count) return T_ERR_INVALID_PARA;
    for (uint32_t i = 0; i < group_count; ++i) {
      axes[i] = merged_axes[i];
      shape[i] = merged_shape[i];
    }
    *dims = group_count;
    return T_SUCCESS;
}

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
  (void)list;
#if THINKER_PARAM_CHECK
  if (op == NULL || tensors == NULL || op->num_input_ != 1 ||
                      op->num_output_ != 1 || (num_tensor != 2 && num_tensor != 3)) {
      return (T_ERR_INVALID_PARA);
  }
#endif
  TransposeAttrs *attrs = (TransposeAttrs *)((int8_t *)op + op->attr_offset_);
  tTensor *X = tensors[0];
  tTensor *Y = tensors[1];
  tTensor *workspace = NULL;
#if THINKER_PARAM_CHECK
  if (X == NULL || Y == NULL || attrs->ndim_ < 2 ||
                      attrs->ndim_ != X->shape_.ndim_ || attrs->ndim_ > 5 ||
                      Y->shape_.ndim_ != X->shape_.ndim_) {
      return (T_ERR_INVALID_PARA);
  }

  if (X->dptr_ == 0 || Y->dptr_ == 0 ||
                      X->dtype_ != Y->dtype_ || X->byte_ != Y->byte_) {
      return (T_ERR_INVALID_DATATYPE);
  }

  if (X->scale_ != Y->scale_ || X->zero_ != Y->zero_) {
      return (T_ERR_INVALID_PARA);
  }
#endif

  bool seen[5] = {false};
  for (uint32_t i = 0; i < attrs->ndim_; ++i) {
    int32_t axis = attrs->axes_[i];
#if THINKER_PARAM_CHECK
    if (axis < 0 || axis >= attrs->ndim_ || seen[axis] ||
                        X->shape_.dims_[i] == 0) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    seen[axis] = true;
    #if THINKER_PARAM_CHECK
    if (Y->shape_.dims_[i] != X->shape_.dims_[axis]) {
        return (T_ERR_INVALID_DATA);
    }
#endif
  }

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
  uint64_t start_t = tick_count();
#endif

  if (num_tensor == 3) {
    workspace = tensors[2];
#if THINKER_PARAM_CHECK
    if (workspace == NULL || workspace->dptr_ == 0 ||
                        workspace->mem_.type_ != 2 || workspace->dtype_ != Int8 ||
                        workspace->byte_ != 1) {
        return (T_ERR_INVALID_PARA);
    }
#endif
  }
#if THINKER_PARAM_CHECK
  if (workspace != NULL &&
                      (workspace->dptr_ == X->dptr_ || workspace->dptr_ == Y->dptr_)) {
      return (T_ERR_INVALID_PARA);
  }
#endif

  uint32_t axes[5];
  for (int32_t i = 0; i < attrs->ndim_; i++) {
    axes[i] = attrs->axes_[i];
  }
  uint32_t shape[5];
  for (int32_t i = 0; i < attrs->ndim_; i++) {
    shape[i] = X->shape_.dims_[i];
  }

  uint32_t new_dims = attrs->ndim_;
  THINKER_RET_CHECK(merge_transpose_axes(axes, shape, &new_dims), "merge_transpose_axes");
  bool memcpy_only = new_dims == 2 && (shape[0] == 1 || shape[1] == 1);
  #if THINKER_RUNTIME_CHECK
  if (!memcpy_only &&
                        (X->mem_.type_ != 2 || Y->mem_.type_ != 2) &&
                        workspace == NULL) {
      return (T_ERR_NO_WORKSPACE);
  }
#endif
  THINKER_RET_CHECK(transpose_luna(X, Y, workspace, new_dims, axes, shape), "transpose_luna");

#if THINKER_PROFILE
  uint64_t finish_t = tick_count();
	uint32_t total_t = (uint32_t)(finish_t - start_t);
  printf("%8s | %u | (","transpose", total_t);  
#endif  
#else
  return T_ERR_NO_IMPLEMENTED;
#endif

  return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
