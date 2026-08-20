#ifndef _TOPN2_LUNA_H_
#define _TOPN2_LUNA_H_

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
 * @brief Find top-N values from pre-computed sorted data
 * @param X Input tensor with pre-sorted values and indices
 * @param Y Output tensor containing top-N results
 * @param work_space Workspace buffer
 * @param attrs TopN attributes including dimension and max number
 * @return Operation status
 */
int32_t topn2_luna(tTensor *X, tTensor *Y, tTensor *work_space, topNAttrs *attrs) { 
  int32_t axis = attrs->dim;
  int32_t n = attrs->max_num;
  int32_t n_dims = X->shape_.ndim_;
  if (axis < 0) axis += n_dims;
  if (X->dtype_ != Int32 || Y->dtype_ != Int32 || n_dims != 3 || axis != n_dims - 1 || n != 1 ||
      X->shape_.dims_[0] != 2 || Y->shape_.ndim_ != 3 || Y->shape_.dims_[0] != 2 ||
      Y->shape_.dims_[1] != X->shape_.dims_[1] || Y->shape_.dims_[2] != 1 ||
      work_space == NULL || work_space->dptr_ == 0 || work_space->shape_.dims_[0] < 16) {
    return T_ERR_INVALID_PARA;
  }
  int32_t once_size = X->shape_.dims_[axis];
  
  int32_t leading = 1;
  int32_t *p_tmp = (int32_t *)work_space->dptr_;

#if THINKER_PARAM_CHECK
if (X->dtype_ != Int32 || Y->dtype_ != Int32) {
    return (T_ERR_INVALID_DATATYPE);
}

if (n != 1 || !(-1 == axis || (n_dims - 1) == axis)) {
    return (T_ERR_INVALID_PARA);
}
#endif

  // Calculate leading dimension for batch processing
  if ((n_dims - 1) == axis)
  {
    leading = X->shape_.dims_[n_dims - 2];
  }

  // Handle top-1 case only (as per switch statement)
  switch (n) {
    case 1: //top 1
    {
      int32_t *p_src_val = (int32_t *)X->dptr_;
      int32_t *p_src_idx = (int32_t *)X->dptr_ + leading * once_size;
      int32_t *p_dst_val = (int32_t *)Y->dptr_;
      int32_t *p_dst_idx = (int32_t *)Y->dptr_ + leading;
      for (int i = 0; i < leading; i++)
      {
        int32_t *p_src_val_tmp = (int32_t *)p_src_val + i * once_size;
        int32_t *p_src_idx_tmp = (int32_t *)p_src_idx + i * once_size;     
        THINKER_RET_CHECK(API_LIB(max_i32o32)(p_src_val_tmp, p_tmp, once_size), "luna_max_i32o32");
        p_dst_val[i] = (int32_t)p_tmp[0];
        p_dst_idx[i] = p_src_idx_tmp[p_tmp[1]];
      }
    }
    break;
    default:
#if THINKER_PARAM_CHECK
if (1) {
    return (T_ERR_INVALID_PARA);
}
#endif
      break;
  }

  if (Y->mem_.type_ == 1) {
    thinker_psram_write_complete((void *)Y->dptr_, getTensorDataSize(Y));
  }

  return T_SUCCESS;
}

#endif
