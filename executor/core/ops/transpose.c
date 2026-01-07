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


// 合并连续且递增的轴索引块，重新排序axes，并更新shape
int32_t merge_transpose_axes(uint32_t *axes, uint32_t *shape, uint32_t *dims) {
    if (*dims <= 1 || *dims > 5)
      return *dims <= 1 ? T_SUCCESS : T_ERR_INVALID_PARA;

    // 临时数组存储合并后的结果
    int32_t temp_axes[5] = {0};
    uint32_t temp_shape[5] = {0};
    uint32_t temp_count = 0;

    uint32_t read_idx = 0;
    uint32_t start_axis = -1;
    uint32_t continue_len = 0;
    for (uint32_t read_idx = 0; read_idx < *dims;) {  
      uint32_t start = read_idx;

      while (read_idx < *dims - 1 && axes[read_idx + 1] == axes[read_idx] + 1) {
        read_idx++;
      }
      if (start != read_idx) {
        start_axis = axes[start];
        continue_len = read_idx - start;
      }
      temp_axes[temp_count] = axes[start];
      read_idx++;
      temp_count++;
    }

    temp_count = 0;
    if (continue_len) {
      for (int32_t i = 0; i < *dims; i++) {
        if (i == start_axis) {
          temp_shape[temp_count] = 1;
          for (int32_t j = 0; j <= continue_len; j++) {
            temp_shape[temp_count] *= shape[j + start_axis];
          }
          i += continue_len;
        }
        else {
          temp_shape[temp_count] = shape[i];
        }
        temp_count++;
      }
    }

    for (int32_t i = 0; i < temp_count; i++) {
      if (temp_axes[i] <= start_axis) {
          axes[i] = temp_axes[i];
      }
      else {
        axes[i] = temp_axes[i] - continue_len;
      }
      shape[i] = temp_shape[i];
    }
    *dims = (temp_count != 0) ? temp_count : *dims;
    
    return T_SUCCESS;
}

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
  // Validate tensor count
  CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
  // Get binary operation attributes
  TransposeAttrs *attrs = (TransposeAttrs *)((int8_t *)op + op->attr_offset_);
  tTensor *X = ((tTensor **)tensors)[0];
  tTensor *Y = ((tTensor **)tensors)[op->num_input_];
  tTensor *workspace = NULL;
  int32_t workspace_size = 0;

  tStatus ret = T_ERR_NO_IMPLEMENTED;
  if ((attrs->ndim_ != X->shape_.ndim_) || (attrs->ndim_ > 5))
    return ret;

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
  uint64_t start_t = tick_count();
#endif

  if (num_tensor == op->num_input_ + op->num_output_ + 1) {
    workspace = ((tTensor**)tensors)[op->num_input_ + op->num_output_];
    workspace_size = workspace->shape_.dims_[0];
  }

  int32_t size = getShapeSize(&X->shape_);

  int32_t in_is_psram = (X->mem_.type_ != 2) ? 1 : 0;
  int32_t ou_is_psram = (Y->mem_.type_ != 2) ? 1 : 0;

  uint32_t axes[5];
  for (int32_t i = 0; i < attrs->ndim_; i++) {
    axes[i] = attrs->axes_[i];
  }
  uint32_t shape[5];
  for (int32_t i = 0; i < attrs->ndim_; i++) {
    shape[i] = X->shape_.dims_[i];
  }

  uint32_t new_dims = attrs->ndim_;
  ret = merge_transpose_axes(axes, shape, &new_dims);
  ret |= transpose_luna(X, Y, workspace, new_dims, axes, shape);

#if THINKER_PROFILE
  uint64_t finish_t = tick_count();
	uint32_t total_t = (uint32_t)(finish_t - start_t);
  printf("%8s | %u | (","transpose", total_t);  
#endif  
#endif

  return ret;
}

#include "core/operator_template.h"
#undef __OP__
