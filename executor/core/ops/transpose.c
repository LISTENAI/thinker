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
static int32_t merge_transpose_axes(uint32_t *axes, uint32_t *shape, uint32_t *dims) {
    if (*dims <= 1)
        return T_SUCCESS;
    else if (*dims > 4)
      return T_ERR_INVALID_PARA;

    // 临时数组存储合并后的结果
    int32_t temp_axes[4];
    uint32_t temp_shape[4];
    uint32_t temp_count = 0;

    uint32_t read_idx = 0;

    while (read_idx < *dims) {
        uint32_t start = read_idx;
        
        // 找到连续递增的块
        while (read_idx < *dims - 1 && axes[read_idx + 1] == axes[read_idx] + 1) {
            read_idx++;
        }
        
        // 计算当前块的最小轴值（作为合并后的轴代表）
        int32_t min_axis = axes[start]; // 由于是连续递增，起始值就是最小值
        
        // 计算当前块的shape乘积
        uint32_t merged_shape = 1;
        for (int32_t i = start; i <= read_idx; i++) {
            merged_shape *= shape[i];
        }
        
        // 存储合并结果
        temp_axes[temp_count] = min_axis;
        temp_shape[temp_count] = merged_shape;
        temp_count++;
        
        read_idx++;
    }
    
    // 按原始顺序重新编码
    // 创建映射：原轴值 -> 新轴值
    int max_axis = -1;
    for (int i = 0; i < temp_count; i++) {
        if (temp_axes[i] > max_axis) {
            max_axis = temp_axes[i];
        }
    }
    
    // 创建映射数组，标记哪些轴值存在
    int exists[max_axis + 1];
    memset(exists, 0, sizeof(exists));
    for (int i = 0; i < temp_count; i++) {
        exists[temp_axes[i]] = 1;
    }
    
    // 按顺序分配新索引
    int new_index = 0;
    for (int i = 0; i <= max_axis; i++) {
        if (exists[i]) {
            exists[i] = new_index++;  // 存储新索引
        }
    }
    
    // 应用映射
    for (int i = 0; i < temp_count; i++) {
        temp_axes[i] = exists[temp_axes[i]];  // 获取新索引
    }
    
    // 将排序后的结果复制回原数组
    for (int32_t i = 0; i < temp_count; i++) {
        axes[i] = temp_axes[i];
        shape[i] = temp_shape[i];
    }
    *dims = temp_count;
    
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

  if ((attrs->ndim_ != X->shape_.ndim_) || (attrs->ndim_ >4) || (X->dtype_ != Int8))
    return T_ERR_INVALID_PARA;
  tStatus ret = T_ERR_NO_IMPLEMENTED;

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

  uint32_t axes[4];
  for (int32_t i = 0; i < attrs->ndim_; i++) {
    axes[i] = attrs->axes_[i];
  }
  uint32_t shape[4];
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
