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

typedef struct _AxisInfo {
  int32_t axis_;
  int32_t idx_; 
  int32_t* p_shape_;
  int32_t num_;
}AxisInfo;

static inline int cmp(const void* a, const void* b) {
    AxisInfo* p1 = (AxisInfo*)a;
    AxisInfo* p2 = (AxisInfo*)b;
    return p1->axis_ - p2->axis_;
}

static void squeeze_transpose_axes(const int32_t* shape, const int32_t* perm, const int32_t ndim, int32_t* squeezed_shape,
                                  int32_t* squeezed_perm, int32_t* squeezed_ndim) {
  int32_t new_perm[8];
  int32_t new_ndim = 0;

  for (int i = 0; i < ndim; ++i) {
    if (shape[i] > 1) {
      squeezed_shape[new_ndim] = shape[i];
      new_perm[i] = new_ndim;
      ++new_ndim;
    } else {
      new_perm[i] = -1;
    }

    int32_t t_ndim = 0;
    for (int i = 0; i < ndim; ++i) {
      int k = perm[i];
      if (new_perm[k] >= 0) {
        squeezed_perm[t_ndim++] = new_perm[k];
      }
    }
  }

  *squeezed_ndim = new_ndim;
}

static void merge_transpose_axis(const int32_t* ori_shape, const int32_t* perm, const int32_t ndim, int32_t* new_shape, 
                                  int32_t* new_perm, int32_t* new_ndim) {
  int32_t perm_groups[ndim][ndim];
  int g_cnt = 0;

  int concec_perm[ndim];
  int concec_size = 0; 
  int last_perm = -10;

  for (int i = 0; i < ndim; ++i) {
    if (perm[i] == last_perm+1) {
      concec_perm[concec_size++] = perm[i];
    }else {
      if (concec_size > 0) {
        perm_groups[g_cnt][0] = concec_size;
#ifdef THINKER_USE_VENUS
        memcpy(&perm_groups[g_cnt][1], concec_perm, concec_size * sizeof(int32_t));
#elif defined THINKER_USE_ARCS
        opi_psram_cpy_out(&perm_groups[g_cnt][1], concec_perm, concec_size * sizeof(int32_t));
#endif
        g_cnt++;
      }
      concec_perm[0] = perm[i];
      concec_size = 1;
    }
    last_perm = perm[i];
  }
  if (concec_size > 0) {
    perm_groups[g_cnt][0] = concec_size;
#ifdef THINKER_USE_VENUS
    memcpy(&perm_groups[g_cnt][1], concec_perm, concec_size * sizeof(int32_t));
#elif defined THINKER_USE_ARCS
    opi_psram_cpy_out(&perm_groups[g_cnt][1], concec_perm, concec_size * sizeof(int32_t));
#endif
    g_cnt++;
  } 

  AxisInfo min_perms_sorted[g_cnt];
  for (int i = 0; i < g_cnt; ++i) {
    min_perms_sorted[i].axis_ = perm_groups[i][1];
    min_perms_sorted[i].idx_ = i;
    min_perms_sorted[i].p_shape_ = &((int32_t *)ori_shape)[perm_groups[i][1]];
    min_perms_sorted[i].num_ = perm_groups[i][0];
  }

  qsort(min_perms_sorted, g_cnt, sizeof(AxisInfo), cmp);
  for (int i = 0; i < g_cnt; ++i) {
    new_perm[i] = min_perms_sorted[i].idx_;
  }

  for (int i = 0; i < g_cnt; ++i) {
    int32_t acc = 1;
    for (int j = 0; j < min_perms_sorted[i].num_; j++) {
      acc *= min_perms_sorted[i].p_shape_[j];
    }
    new_shape[i] = acc;
  }
  *new_ndim = g_cnt;
}

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) 
{
  CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
  TransposeAttrs *attrs = (TransposeAttrs *)((int8_t *)op + op->attr_offset_);
  tTensor *X = ((tTensor **)tensors)[0];
  tTensor *Y = ((tTensor **)tensors)[op->num_input_];
  tTensor *Temp = NULL;
  int32_t workspace_size = 0;
  if (num_tensor == op->num_input_ + op->num_output_ + 1) {
    Temp = ((tTensor**)tensors)[op->num_input_ + op->num_output_];
    workspace_size = Temp->shape_.dims_[0];
  }

  tStatus ret = T_ERR_NO_IMPLEMENTED;
  tShape ishape = X->shape_;
  tShape ostride = calcStride(&Y->shape_);
  tShape stride;
  stride.ndim_ = Y->shape_.ndim_;

  for (int32_t i = 0; i < ishape.ndim_; i++)
    stride.dims_[attrs->axes_[i]] = ostride.dims_[i];

  int32_t size = getShapeSize(&X->shape_);

  int32_t in_is_psram = (X->mem_.type_ != 2) ? 1 : 0;
  int32_t ou_is_psram = (Y->mem_.type_ != 2) ? 1 : 0;

  int32_t dtype = X->dtype_;
  int32_t n_dims = X->shape_.ndim_;
  int32_t *in_shape = (int32_t *)ishape.dims_;
  // int32_t *axis = (int32_t *)attrs->axes_;
  void *src = (void *)X->dptr_;
  void *dst = (void *)Y->dptr_;
  int32_t axis[6];
  for (int32_t i = 0; i < attrs->ndim_; i++) {
    axis[i] = attrs->axes_[i];
  }

#if (THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA)
#if THINKER_PROFILE
  uint64_t start_t = tick_count();
#endif
  if ((!in_is_psram) & (!ou_is_psram))
  {
    int32_t dtype = X->dtype_;
    int32_t n_dims = X->shape_.ndim_;
    int32_t *in_shape = (int32_t *)ishape.dims_;
    void *src = (void *)X->dptr_;
    void *dst = (void *)Y->dptr_;

    int32_t axis[6];
    int32_t new_shape[6];
    int32_t new_perm[6];
    int32_t new_ndim = 0;
    for (int32_t i = 0; i < attrs->ndim_; i++) {
      axis[i] = attrs->axes_[i];
    }
    if (attrs->ndim_ > 3) {
      merge_transpose_axis(in_shape, axis, attrs->ndim_, new_shape, new_perm, &new_ndim);
    } 
    else {
#ifdef THINKER_USE_VENUS
      memcpy(new_perm, axis, attrs->ndim_ * sizeof(int32_t));
      memcpy(new_shape, in_shape, attrs->ndim_ * sizeof(int32_t));
#elif (THINKER_USE_ARCS || THINKER_USE_VENUSA)
      opi_psram_cpy_out(new_perm, axis, attrs->ndim_ * sizeof(int32_t));
      opi_psram_cpy_out(new_shape, in_shape, attrs->ndim_ * sizeof(int32_t));
#endif
      new_ndim = attrs->ndim_;
    }

    switch (new_ndim) {
      case 2: {
        int32_t row = new_shape[0];
        int32_t col = new_shape[1];
        ret = transpose_luna(dtype, dst, src, row, col);
        break;
      }
      case 3: {
        ret = transpose_axis_luna(dtype, src, dst, new_shape, new_perm, new_ndim);
        break;
      }
      case 4:  // only support (0 == new_perm[0])
      {
        if (0 == new_perm[0]) {
          int32_t new_axis[3];
          int32_t new_in_shape[3];
          int32_t batch = new_shape[0];
          int32_t dbyte = 0xF & dtype;
          int32_t one_batch_size = new_shape[1] * new_shape[2] * new_shape[3] * dbyte;
          new_ndim = 3;
          for (int32_t n = 0; n < new_ndim; n++) {
            new_axis[n] = new_perm[n + 1] - 1;
            new_in_shape[n] = new_shape[n + 1];
          }
          for (int32_t i = 0; i < batch; i++) {
            void *tsrc = (void *)((int8_t *)src + i * one_batch_size);
            void *tdst = (void *)((int8_t *)dst + i * one_batch_size);
            ret = transpose_axis_luna(dtype, (void *)src, (void *)tdst,
                                       new_in_shape, new_axis, new_ndim);
          }
        }
        else {
          return T_ERR_NO_IMPLEMENTED;
        }
      } break;
      default:
        return T_ERR_NO_IMPLEMENTED;
        break;
    }
  }
#endif
#if defined(THINKER_USE_VENUS)
  else {
    return T_ERR_INVALID_PLATFROM;
  }
#endif
#if defined(THINKER_USE_ARCS)
  else if ((in_is_psram) | (ou_is_psram)) { // 有一个在psram
    if (workspace_size >= size * X->byte_) {
      if (in_is_psram) {
        opi_psram_cpy_in((int8_t *)Temp->dptr_, src, size * X->byte_);
        src = (int8_t *)Temp->dptr_;
      }
      else {
        dst = (int8_t *)Temp->dptr_;
      }

      if (attrs->ndim_ == 2) {
          int32_t row = in_shape[0];
          int32_t col = in_shape[1];
          ret = transpose_luna(dtype, dst, src, row, col);
        }
      else if (attrs->ndim_ == 3) {
          ret = transpose_axis_luna(dtype, src, dst, in_shape, axis, X->shape_.ndim_);
      }
      else if (attrs->ndim_ == 4) { // only support (0 == axis[0])
        if (0 == axis[0]) {
          int32_t new_axis[3];
          int32_t new_in_shape[3];
          int32_t batch = in_shape[0];
          int32_t dbyte = 0xF & dtype;
          int32_t one_batch_size = in_shape[1] * in_shape[2] * in_shape[3] * dbyte;
          int32_t n_dims = 3;
          for (int32_t n = 0; n < n_dims; n++) {
            new_axis[n] = axis[n + 1] - 1;
            new_in_shape[n] = in_shape[n + 1];
          }
          for (int32_t i = 0; i < batch; i++) {
            void *tsrc = (void *)((int8_t *)src + i * one_batch_size);
            void *tdst = (void *)((int8_t *)dst + i * one_batch_size);
            ret = transpose_axis_luna(dtype, (void *)src, (void *)tdst, new_in_shape, new_axis, n_dims);
          }
        }
        else {
          printf("Do not support this type!\n");
          return T_ERR_NO_IMPLEMENTED;
        }
      }
      if (ou_is_psram) {
        opi_psram_cpy_in((int8_t *)Y->dptr_, dst, size * X->byte_);
      }
    }
    else {
      if (attrs->ndim_ == 2) {
        int32_t row = in_shape[0];
        int32_t col = in_shape[1];
        if (col * 2 <= workspace_size) {
          int32_t split_col = ceil(workspace_size / row);
          ret = split_transpose_luna(Y, X, Temp, row, col, split_col);
        }
        else
          ret = T_ERR_NO_IMPLEMENTED;
      }
      else if (attrs->ndim_ == 3) {
        ret = transpose_axis_luna(dtype, src, dst, in_shape, axis, X->shape_.ndim_);
      }
      else if (attrs->ndim_ == 4) { // only support (0 == axis[0])
        if (0 == axis[0]) {
          int32_t new_axis[3];
          int32_t new_in_shape[3];
          int32_t batch = in_shape[0];
          int32_t dbyte = 0xF & dtype;
          int32_t one_batch_size = in_shape[1] * in_shape[2] * in_shape[3] * dbyte;
          int32_t n_dims = 3;
          for (int32_t n = 0; n < n_dims; n++) {
            new_axis[n] = axis[n + 1] - 1;
            new_in_shape[n] = in_shape[n + 1];
          }
          for (int32_t i = 0; i < batch; i++) {
            void *tsrc = (void *)((int8_t *)src + i * one_batch_size);
            void *tdst = (void *)((int8_t *)dst + i * one_batch_size);
            ret = transpose_axis_luna(dtype, (void *)src, (void *)tdst, new_in_shape, new_axis, n_dims);
          }
        }
      }
    }
  }
  else if ((in_is_psram) & (ou_is_psram)) { // 两个都在psram
    if (workspace_size >= size  * 2 * X->byte_) {
      opi_psram_cpy_in((int8_t *)Temp->dptr_, src, size * X->byte_);
      src = (int8_t *)Temp->dptr_;
      dst = (int8_t *)Temp->dptr_ + size * X->byte_;
      if (attrs->ndim_ == 2) {
          int32_t row = in_shape[0];
          int32_t col = in_shape[1];
          ret = transpose_luna(dtype, dst, src, row, col);
        }
      else if (attrs->ndim_ == 3) {
          ret = transpose_axis_luna(dtype, src, dst, in_shape, axis, X->shape_.ndim_);
      }
      else if (attrs->ndim_ == 4) { // only support (0 == axis[0])
        if (0 == axis[0]) {
          int32_t new_axis[3];
          int32_t new_in_shape[3];
          int32_t batch = in_shape[0];
          int32_t dbyte = 0xF & dtype;
          int32_t one_batch_size = in_shape[1] * in_shape[2] * in_shape[3] * dbyte;
          int32_t n_dims = 3;
          for (int32_t n = 0; n < n_dims; n++) {
            new_axis[n] = axis[n + 1] - 1;
            new_in_shape[n] = in_shape[n + 1];
          }
          for (int32_t i = 0; i < batch; i++) {
            void *tsrc = (void *)((int8_t *)src + i * one_batch_size);
            void *tdst = (void *)((int8_t *)dst + i * one_batch_size);
            ret = transpose_axis_luna(dtype, (void *)src, (void *)tdst, new_in_shape, new_axis, n_dims);
          }
        }
      }
      opi_psram_cpy_in((int8_t *)Y->dptr_, dst, size * X->byte_);
    }
    else if (workspace_size >= 16384) {
      int32_t split_num = ceil(size / 16384.0);
      if (workspace_size >= size * X->byte_) {

    }
  }
}
#if THINKER_PROFILE
  uint64_t finish_t = tick_count();
	uint32_t total_t = (uint32_t)(finish_t - start_t);
  printf("%8s | %u | (","transpose", total_t);  
#endif  
#endif
#if defined(THINKER_USE_VENUSA)
  else if ((in_is_psram) | (ou_is_psram)) { // 有一个在psram
    if (workspace_size >= size * X->byte_) {
      if (in_is_psram) {
    	  ret = luna_psrammemcpy_i8o8((int8_t *)Temp->dptr_, src, size * X->byte_);
        src = (int8_t *)Temp->dptr_;
      }

      if (attrs->ndim_ == 2) {
          int32_t row = in_shape[0];
          int32_t col = in_shape[1];
          ret = transpose_luna(dtype, dst, src, row, col);
        }
      else if (attrs->ndim_ == 3) {
          ret = transpose_axis_luna(dtype, src, dst, in_shape, axis, X->shape_.ndim_);
      }
      else if (attrs->ndim_ == 4) { // only support (0 == axis[0])
        if (0 == axis[0]) {
          int32_t new_axis[3];
          int32_t new_in_shape[3];
          int32_t batch = in_shape[0];
          int32_t dbyte = 0xF & dtype;
          int32_t one_batch_size = in_shape[1] * in_shape[2] * in_shape[3] * dbyte;
          int32_t n_dims = 3;
          for (int32_t n = 0; n < n_dims; n++) {
            new_axis[n] = axis[n + 1] - 1;
            new_in_shape[n] = in_shape[n + 1];
          }
          for (int32_t i = 0; i < batch; i++) {
            void *tsrc = (void *)((int8_t *)src + i * one_batch_size);
            void *tdst = (void *)((int8_t *)dst + i * one_batch_size);
            ret = transpose_axis_luna(dtype, (void *)src, (void *)tdst, new_in_shape, new_axis, n_dims);
          }
        }
        else {
          printf("Do not support this type!\n");
          return T_ERR_NO_IMPLEMENTED;
        }
      }
      if (ou_is_psram) {
        opi_psram_cpy_out((int8_t *)Y->dptr_, dst, size * X->byte_);
      }
    }
    else {
      if (attrs->ndim_ == 2) {
        int32_t row = in_shape[0];
        int32_t col = in_shape[1];
        if (col * 2 <= workspace_size) {
          int32_t split_col = ceil(workspace_size / row);
          ret = split_transpose_luna(Y, X, Temp, row, col, split_col);
        }
        else
          ret = T_ERR_NO_IMPLEMENTED;
      }
      else if (attrs->ndim_ == 3) {
        ret = transpose_axis_luna(dtype, src, dst, in_shape, axis, X->shape_.ndim_);
      }
      else if (attrs->ndim_ == 4) { // only support (0 == axis[0])
        if (0 == axis[0]) {
          if (1 == axis[1]) {
			  int32_t new_axis[3];
			  int32_t new_in_shape[3];
			  int32_t batch = in_shape[0];
			  int32_t dbyte = 0xF & dtype;
			  int32_t one_batch_size = in_shape[2] * in_shape[3] * dbyte;
			  int32_t n_dims = 2;
			  int8_t *tsrc = NULL;
			  int8_t *tdst = NULL;
			  for (int32_t n = 0; n < n_dims; n++) {
				new_axis[n] = axis[n + 2] - 2;
				new_in_shape[n] = in_shape[n + 2];
			  }
			  if (workspace_size >= one_batch_size) {
				  for (int32_t i = 0; i < in_shape[1]; i++) {
					tsrc = (int8_t *)src + i * one_batch_size;
					tdst = (int8_t *)dst + i * one_batch_size;
				    if (in_is_psram) {
              ret = luna_psrammemcpy_i8o8((int8_t *)Temp->dptr_, (int8_t *)src + i * one_batch_size, one_batch_size);
              tsrc = (int8_t *)Temp->dptr_;
				    }
				    if (ou_is_psram)
				    	tdst = (int8_t *)Temp->dptr_;

				    ret = transpose_luna(dtype, (void *)tdst, (void *)tsrc, new_in_shape[0], new_in_shape[1]);

				    if (ou_is_psram) {
					    opi_psram_cpy_out((int8_t *)dst + i * one_batch_size, tdst, one_batch_size);
				    }
				  }
#if !(defined(WIN32)|| defined(linux))
		HAL_FlushDCache_by_Addr((uint32_t *)(Y->dptr_), in_shape[1] * one_batch_size);
#endif
			   }
      }
      else {
			  int32_t new_axis[3];
			  int32_t new_in_shape[3];
			  int32_t batch = in_shape[0];
			  int32_t dbyte = 0xF & dtype;
			  int32_t one_batch_size = in_shape[1] * in_shape[2] * in_shape[3] * dbyte;
			  int32_t n_dims = 3;
			  for (int32_t n = 0; n < n_dims; n++) {
          new_axis[n] = axis[n + 1] - 1;
          new_in_shape[n] = in_shape[n + 1];
			  }
			  for (int32_t i = 0; i < batch; i++) {
          void *tsrc = (void *)((int8_t *)src + i * one_batch_size);
          void *tdst = (void *)((int8_t *)dst + i * one_batch_size);
          ret = transpose_axis_luna(dtype, (void *)src, (void *)tdst, new_in_shape, new_axis, n_dims);
			  }
          }
        }
      }
    }
  }
  else if ((in_is_psram) & (ou_is_psram)) { // 两个都在psram
    if (workspace_size >= size  * 2 * X->byte_) {
      ret = luna_psrammemcpy_i8o8((int8_t *)Temp->dptr_, src, size * X->byte_);
      src = (int8_t *)Temp->dptr_;
      dst = (int8_t *)Temp->dptr_ + size * X->byte_;
      if (attrs->ndim_ == 2) {
          int32_t row = in_shape[0];
          int32_t col = in_shape[1];
          ret = transpose_luna(dtype, dst, src, row, col);
        }
      else if (attrs->ndim_ == 3) {
          ret = transpose_axis_luna(dtype, src, dst, in_shape, axis, X->shape_.ndim_);
      }
      else if (attrs->ndim_ == 4) { // only support (0 == axis[0])
        if (0 == axis[0]) {
          int32_t new_axis[3];
          int32_t new_in_shape[3];
          int32_t batch = in_shape[0];
          int32_t dbyte = 0xF & dtype;
          int32_t one_batch_size = in_shape[1] * in_shape[2] * in_shape[3] * dbyte;
          int32_t n_dims = 3;
          for (int32_t n = 0; n < n_dims; n++) {
            new_axis[n] = axis[n + 1] - 1;
            new_in_shape[n] = in_shape[n + 1];
          }
          for (int32_t i = 0; i < batch; i++) {
            void *tsrc = (void *)((int8_t *)src + i * one_batch_size);
            void *tdst = (void *)((int8_t *)dst + i * one_batch_size);
            ret = transpose_axis_luna(dtype, (void *)src, (void *)tdst, new_in_shape, new_axis, n_dims);
          }
        }
      }
      opi_psram_cpy_out((int8_t *)Y->dptr_, dst, size * X->byte_);
    }
    else if (workspace_size >= 16384) {
      int32_t split_num = ceil(size / 16384.0);
      if (workspace_size >= size * X->byte_) {

    }
  }
}
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
