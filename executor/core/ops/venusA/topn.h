#ifndef _TOPN_LUNA_H_
#define _TOPN_LUNA_H_

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

#include "thinker_status.h"

/**
 * @brief Find top-N values and indices along specified axis
 * @param X Input tensor
 * @param Index Index tensor (contains offset)
 * @param Y Output tensor containing values and indices
 * @param work_space Workspace buffer
 * @param attrs TopN attributes including dimension and max number
 * @return Operation status
 */
int32_t topn_luna(tTensor *X, tTensor *Index, tTensor *Y, tTensor *work_space, topNAttrs *attrs) 
{
  int32_t ret = T_ERR_NO_IMPLEMENTED;
  
  int32_t axis = attrs->dim;
  int32_t n = attrs->max_num;
  int32_t n_dims = X->shape_.ndim_;
  int32_t once_size = axis == -1 ? X->shape_.dims_[n_dims - 1] : X->shape_.dims_[axis];
  
  int32_t leading = 1;
  int32_t *p_tmp = (int32_t *)work_space->dptr_;
  int64_t idx_offset = *(int64_t *)Index->dptr_;

  // Calculate leading dimension for batch processing
  if (-1 == axis || (n_dims - 1) == axis)
  {
    for (int i = 0; i < (n_dims - 1); i++)
    {
      leading *= X->shape_.dims_[i];
    }
  } 

  // Handle top-1 case only (as per switch statement)
  switch (n) {
    case 1: //top 1
    {
      int32_t *p_dst_val = (int32_t *)Y->dptr_;
      int32_t *p_dst_idx = (int32_t *)Y->dptr_ + leading;
      for (int i = 0; i < leading; i++)
      {
        int8_t *p_src = (int8_t *)X->dptr_ + i * once_size;     
        ret = API_LIB(max_i8o32)(p_src, p_tmp, once_size);
        p_dst_val[i] = (int32_t)p_tmp[0];
        p_dst_idx[i] = (int32_t)(p_tmp[1] + idx_offset);
      }
    }
    break;
    default:
      break;
  }

  return ret;
}

#endif
