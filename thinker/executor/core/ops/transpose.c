#undef __OP__
#define __OP__ Transpose
#include <stdio.h>
#include <string.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_VENUS
#include "./venus/transpose.h"
#endif

#define BLOCK_M 32
#define BLOCK_N 32

void trans_block_buff_stride_fp32(float *dst, const float *src, int32_t row,
                                  int32_t col, int32_t stride) {
  for (int32_t i = 0; i < row; ++i) {
    for (int32_t j = 0; j < col; ++j) {
      memcpy(dst + (j * row + i) * stride, src + (i * col + j) * stride,
             stride * sizeof(float));
    }
  }
}

void trans_block_buff_stride_int8(int8_t *dst, const int8_t *src, int32_t row,
                                  int32_t col, int32_t stride) {
  for (int32_t i = 0; i < row; ++i) {
    for (int32_t j = 0; j < col; ++j) {
      memcpy(dst + (j * row + i) * stride, src + (i * col + j) * stride,
             stride * sizeof(int8_t));
    }
  }
}

void trans_block_buff_stride(int16_t dtype, void *dst, const void *src,
                             int32_t row, int32_t col, int32_t stride) {
  if (dtype == Float32) {
    trans_block_buff_stride_fp32((float *)dst, (const float *)src, row, col,
                                 stride);
  } else {
    trans_block_buff_stride_int8((int8_t *)dst, (const int8_t *)src, row, col,
                                 stride);
  }
}

void trans_block_buff_fp32(float *dst, const float *src, int32_t row,
                           int32_t col) {
  float buff[BLOCK_M * BLOCK_N];
  int32_t i = 0, j = 0;
  int32_t BLOCK_M_HALF = BLOCK_M;
  int32_t BLOCK_N_HALF = BLOCK_N;
  while (BLOCK_M_HALF > 1 && BLOCK_N_HALF > 1) {
    if (i + BLOCK_M_HALF > col) {
      BLOCK_M_HALF >>= 1;
      continue;
    }
    if (j + BLOCK_N_HALF > row) {
      BLOCK_N_HALF >>= 1;
      continue;
    }

    for (int32_t ii = i; ii + BLOCK_M_HALF <= col; ii += BLOCK_M_HALF) {
      for (int32_t jj = 0; jj + BLOCK_N_HALF <= j; jj += BLOCK_N_HALF) {
        for (int32_t n = 0; n < BLOCK_N_HALF; n++) {
          float *buff_tmp = &buff[n * BLOCK_M_HALF];
          const float *src_tmp = &src[(n + jj) * col + ii];
          for (int32_t m = 0; m < BLOCK_M_HALF; m++) {
            buff_tmp[m] = src_tmp[m];
          }
        }
        for (int32_t m = 0; m < BLOCK_M_HALF; m++) {
          float *dst_tmp = &dst[(m + ii) * row + jj];
          const float *buff_tmp = &buff[m];
          for (int32_t n = 0; n < BLOCK_N_HALF; n++) {
            dst_tmp[n] = buff_tmp[n * BLOCK_M_HALF];
          }
        }
      }
    }
    i += (col - i) / BLOCK_M_HALF * BLOCK_M_HALF;
    for (int32_t ii = 0; ii + BLOCK_M_HALF <= col; ii += BLOCK_M_HALF) {
      for (int32_t jj = j; jj + BLOCK_N_HALF <= row; jj += BLOCK_N_HALF) {
        for (int32_t n = 0; n < BLOCK_N_HALF; n++) {
          float *buff_tmp = &buff[n * BLOCK_M_HALF];
          const float *src_tmp = &src[(n + jj) * col + ii];
          for (int32_t m = 0; m < BLOCK_M_HALF; m++) {
            buff_tmp[m] = src_tmp[m];
          }
        }
        for (int32_t m = 0; m < BLOCK_M_HALF; m++) {
          float *dst_tmp = &dst[(m + ii) * row + jj];
          const float *buff_tmp = &buff[m];
          for (int32_t n = 0; n < BLOCK_N_HALF; n++) {
            dst_tmp[n] = buff_tmp[n * BLOCK_M_HALF];
          }
        }
      }
    }
    j += (row - j) / BLOCK_N_HALF * BLOCK_N_HALF;
  }
  for (int32_t c = i; c < col; ++c) {
    for (int32_t r = 0; r < j; ++r) {
      dst[c * row + r] = src[r * col + c];
    }
  }
  for (int32_t c = 0; c < col; ++c) {
    for (int32_t r = j; r < row; ++r) {
      dst[c * row + r] = src[r * col + c];
    }
  }
}

void trans_block_buff_int8(int8_t *dst, const int8_t *src, int32_t row,
                           int32_t col) {
  int8_t buff[BLOCK_M * BLOCK_N];
  int32_t i = 0, j = 0;
  int32_t BLOCK_M_HALF = BLOCK_M;
  int32_t BLOCK_N_HALF = BLOCK_N;
  while (BLOCK_M_HALF > 1 && BLOCK_N_HALF > 1) {
    if (i + BLOCK_M_HALF > col) {
      BLOCK_M_HALF >>= 1;
      continue;
    }
    if (j + BLOCK_N_HALF > row) {
      BLOCK_N_HALF >>= 1;
      continue;
    }

    for (int32_t ii = i; ii + BLOCK_M_HALF <= col; ii += BLOCK_M_HALF) {
      for (int32_t jj = 0; jj + BLOCK_N_HALF <= j; jj += BLOCK_N_HALF) {
        for (int32_t n = 0; n < BLOCK_N_HALF; n++) {
          int8_t *buff_tmp = &buff[n * BLOCK_M_HALF];
          const int8_t *src_tmp = &src[(n + jj) * col + ii];
          for (int32_t m = 0; m < BLOCK_M_HALF; m++) {
            buff_tmp[m] = src_tmp[m];
          }
        }
        for (int32_t m = 0; m < BLOCK_M_HALF; m++) {
          int8_t *dst_tmp = &dst[(m + ii) * row + jj];
          const int8_t *buff_tmp = &buff[m];
          for (int32_t n = 0; n < BLOCK_N_HALF; n++) {
            dst_tmp[n] = buff_tmp[n * BLOCK_M_HALF];
          }
        }
      }
    }
    i += (col - i) / BLOCK_M_HALF * BLOCK_M_HALF;
    for (int32_t ii = 0; ii + BLOCK_M_HALF <= col; ii += BLOCK_M_HALF) {
      for (int32_t jj = j; jj + BLOCK_N_HALF <= row; jj += BLOCK_N_HALF) {
        for (int32_t n = 0; n < BLOCK_N_HALF; n++) {
          int8_t *buff_tmp = &buff[n * BLOCK_M_HALF];
          const int8_t *src_tmp = &src[(n + jj) * col + ii];
          for (int32_t m = 0; m < BLOCK_M_HALF; m++) {
            buff_tmp[m] = src_tmp[m];
          }
        }
        for (int32_t m = 0; m < BLOCK_M_HALF; m++) {
          int8_t *dst_tmp = &dst[(m + ii) * row + jj];
          const int8_t *buff_tmp = &buff[m];
          for (int32_t n = 0; n < BLOCK_N_HALF; n++) {
            dst_tmp[n] = buff_tmp[n * BLOCK_M_HALF];
          }
        }
      }
    }
    j += (row - j) / BLOCK_N_HALF * BLOCK_N_HALF;
  }
  for (int32_t c = i; c < col; ++c) {
    for (int32_t r = 0; r < j; ++r) {
      dst[c * row + r] = src[r * col + c];
    }
  }
  for (int32_t c = 0; c < col; ++c) {
    for (int32_t r = j; r < row; ++r) {
      dst[c * row + r] = src[r * col + c];
    }
  }
}

void trans_block_buff(int16_t dtype, void *dst, const void *src, int32_t row,
                      int32_t col) {
  if (dtype == Float32) {
    trans_block_buff_fp32((float *)dst, (const float *)src, row, col);
  } else {
    trans_block_buff_int8((int8_t *)dst, (const int8_t *)src, row, col);
  }
}

int32_t calc_out_index(int32_t oindex, const int32_t index, int32_t dims,
                       const int32_t *ishape, const int32_t *stride) {
  if (dims == 1) {
    oindex += index * stride[0];
  } else {
    oindex += index % ishape[dims - 1] * stride[dims - 1];
    oindex = calc_out_index(oindex, index / ishape[dims - 1], dims - 1, ishape,
                            stride);
  }
  return oindex;
}

void transpose_kernel(uint16_t dtype, const void *p_input, void *p_output,
                      const int32_t size, const int32_t ndim,
                      const int32_t *ishape, const int32_t *stride) {
  if (dtype == Float32) {
    float *in = (float *)p_input;
    float *out = (float *)p_output;
    for (int32_t i = 0; i < size; i++) {
      int32_t oindex = 0;
      oindex = calc_out_index(oindex, i, ndim, ishape, stride);
      out[oindex] = in[i];
    }
  } else {
    int8_t *in = (int8_t *)p_input;
    int8_t *out = (int8_t *)p_output;
    for (int32_t i = 0; i < size; i++) {
      int32_t oindex = 0;
      oindex = calc_out_index(oindex, i, ndim, ishape, stride);
      out[oindex] = in[i];
    }
  }
}

void transpose_axis_dim3(uint16_t dtype, const void *p_input, void *p_output,
                         const int32_t *in_shape, const int32_t *stride,
                         int32_t size, const int8_t *axis) {
  // (1, 2, 0)
  if (axis[0] == 1 && axis[1] == 2 && axis[2] == 0) {
    trans_block_buff(dtype, (void *)p_output, (const void *)p_input,
                     in_shape[0], in_shape[1] * in_shape[2]);
  }  // (0, 2, 1)
  else if (axis[0] == 0 && axis[1] == 2 && axis[2] == 1) {
    int32_t n_dim12 = in_shape[1] * in_shape[2];
    int32_t c = in_shape[0];
    int32_t byte = 0x000F & dtype;
    // OMP_PARALLEL_FOR_
    for (int32_t i = 0; i < c; ++i) {
      int64_t tdst = (int64_t)((int8_t *)p_output + i * n_dim12 * byte);
      int64_t tsrc = (int64_t)((int8_t *)p_input + i * n_dim12 * byte);
      trans_block_buff(dtype, (void *)tdst, (const void *)tsrc, in_shape[1],
                       in_shape[2]);
    }
  }  // (1, 0, 2)
  else if (axis[0] == 1 && axis[1] == 0 && axis[2] == 2) {
    trans_block_buff_stride(dtype, p_output, p_input, in_shape[0], in_shape[1],
                            in_shape[2]);
  } else {
    transpose_kernel(dtype, p_input, p_output, size, 3, in_shape, stride);
  }
}

void transpose_axis_dim4(uint16_t dtype, const void *p_input, void *p_output,
                         const int32_t *in_shape, const int32_t *stride,
                         int32_t size, const int8_t *axis) {
  // 0 1 3 2
  if (axis[1] == 1 && axis[2] == 3 && axis[3] == 2) {
    int32_t n_dim23 = in_shape[2] * in_shape[3];
    int32_t c = in_shape[0] * in_shape[1];
    int32_t byte = 0x000F & dtype;
    // OMP_PARALLEL_FOR_
    for (int32_t i = 0; i < c; ++i) {
      int64_t tdst = (int64_t)((int8_t *)p_output + i * n_dim23 * byte);
      int64_t tsrc = (int64_t)((int8_t *)p_input + i * n_dim23 * byte);
      trans_block_buff(dtype, (void *)tdst, (const void *)tsrc, in_shape[2],
                       in_shape[3]);
    }
  }
  // 1 0 2 3
  else if (axis[1] == 0 && axis[2] == 2 && axis[3] == 3) {
    trans_block_buff_stride(dtype, p_output, p_input, in_shape[0], in_shape[1],
                            in_shape[2] * in_shape[3]);
  }
  // 0 2 1 3
  else if (axis[1] == 2 && axis[2] == 1 && axis[3] == 3) {
    int32_t n = in_shape[0];
    int32_t n_dim123 = in_shape[1] * in_shape[2] * in_shape[3];
    int32_t byte = 0x000F & dtype;
    // OMP_PARALLEL_FOR_
    for (int32_t i = 0; i < n; ++i) {
      int64_t tdst = (int64_t)((int8_t *)p_output + i * n_dim123 * byte);
      int64_t tsrc = (int64_t)((int8_t *)p_input + i * n_dim123 * byte);
      trans_block_buff_stride(dtype, (void *)tdst, (void *)tsrc, in_shape[1],
                              in_shape[2], in_shape[3]);
    }
  }
  // 0 2 3 1
  else if (axis[1] == 2 && axis[2] == 3 && axis[3] == 1) {
    int32_t n_dim123 = in_shape[1] * in_shape[2] * in_shape[3];
    int32_t c = in_shape[0];
    int32_t byte = 0x000F & dtype;
    // OMP_PARALLEL_FOR_
    for (int32_t i = 0; i < c; ++i) {
      int64_t tdst = (int64_t)((int8_t *)p_output + i * n_dim123 * byte);
      int64_t tsrc = (int64_t)((int8_t *)p_input + i * n_dim123 * byte);
      trans_block_buff(dtype, (void *)tdst, (void *)tsrc, in_shape[1],
                       in_shape[2] * in_shape[3]);
    }
  }
  // 0 3 1 2
  else if (axis[1] == 3 && axis[2] == 1 && axis[3] == 2) {
    int32_t n_dim123 = in_shape[1] * in_shape[2] * in_shape[3];
    int32_t c = in_shape[0];
    int32_t byte = 0x000F & dtype;
    // OMP_PARALLEL_FOR_
    for (int32_t i = 0; i < c; ++i) {
      int64_t tdst = (int64_t)((int8_t *)p_output + i * n_dim123 * byte);
      int64_t tsrc = (int64_t)((int8_t *)p_input + i * n_dim123 * byte);
      trans_block_buff(dtype, (void *)tdst, (void *)tsrc,
                       in_shape[1] * in_shape[2], in_shape[3]);
    }
  } else {
    transpose_kernel(dtype, p_input, p_output, size, 4, in_shape, stride);
  }
}

void transpose_axis_dim5(uint16_t dtype, const void *p_input, void *p_output,
                         const int32_t *in_shape, const int32_t *stride,
                         int32_t size, const int8_t *axis) {
  if (axis[1] == 2 && axis[2] == 1 && axis[3] == 3 && axis[4] == 4) {
    int32_t n_dim1234 = in_shape[1] * in_shape[2] * in_shape[3] * in_shape[4];
    int32_t c = in_shape[0];
    int8_t new_axis[4] = {1, 0, 2, 3};
    int32_t byte = 0x000F & dtype;
    for (int32_t i = 0; i < c; ++i) {
      int64_t tdst = (int64_t)((int8_t *)p_output + i * n_dim1234 * byte);
      int64_t tsrc = (int64_t)((int8_t *)p_input + i * n_dim1234 * byte);
      transpose_axis_dim4(dtype, (void *)tsrc, (void *)tdst, in_shape + 1,
                          stride + 1, n_dim1234, new_axis);
    }
  } else {
    transpose_kernel(dtype, p_input, p_output, size, 5, in_shape, stride);
  }
}

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                   tDMA_List *list) {
  CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));
  tTensor *X = ((tTensor **)tensors)[0];
  TransposeAttrs *attrs = (TransposeAttrs *)((int8_t *)op + op->attr_offset_);
  tTensor *Y = ((tTensor **)tensors)[op->num_input_];

  tShape ishape = X->shape_;
  tShape ostride = calcStride(&Y->shape_);
  tShape stride;
  stride.ndim_ = Y->shape_.ndim_;

  for (int32_t i = 0; i < ishape.ndim_; i++)
    stride.dims_[attrs->axes_[i]] = ostride.dims_[i];

  int32_t size = getShapeSize(&X->shape_);

#ifdef THINKER_USE_VENUS
  {
    int32_t ret = -1;
    int32_t dtype = X->dtype_;
    int32_t n_dims = X->shape_.ndim_;
    int32_t *in_shape = (int32_t *)ishape.dims_;
    // int32_t *axis = (int32_t *)attrs->axes_;
    void *src = (void *)X->dptr_;
    void *dst = (void *)Y->dptr_;
    int32_t axis[4];
    for (int32_t i = 0; i < attrs->ndim_; i++) {
      axis[i] = attrs->axes_[i];
    }
    switch (attrs->ndim_) {
      case 2: {
        int32_t row = in_shape[0];
        int32_t col = in_shape[1];
        ret = transpose_luna(dtype, src, dst, row, col);
        break;
      }
      case 3: {
        ret = transpose_axis_luna(dtype, src, dst, in_shape, axis, n_dims);
        break;
      }
      case 4:  // only support (0 == axis[0])
      {
        if (0 == axis[0]) {
          int32_t new_axis[3];
          int32_t new_in_shape[3];
          int32_t batch = in_shape[0];
          int32_t dbyte = 0xF & dtype;
          int32_t one_batch_size =
              in_shape[1] * in_shape[2] * in_shape[3] * dbyte;
          n_dims = 3;
          for (int32_t n = 0; n < n_dims; n++) {
            new_axis[n] = axis[n + 1] - 1;
            new_in_shape[n] = in_shape[n + 1];
          }
          for (int32_t i = 0; i < batch; i++) {
            void *tsrc = (void *)((int8_t *)src + i * one_batch_size);
            void *tdst = (void *)((int8_t *)dst + i * one_batch_size);
            ret = transpose_axis_luna(dtype, (void *)src, (void *)tdst,
                                       new_in_shape, new_axis, n_dims);
          }
        }
      } break;
      default:
        break;
    }
    if (0 == ret) {
      return ret;
    }
  }
#endif

  switch (attrs->ndim_) {
    case 1:
      memcpy((void *)Y->dptr_, (void *)X->dptr_, size * X->byte_);
      break;
    case 2:
      if (attrs->axes_[0] == 1 && attrs->axes_[1] == 0)
        transpose_kernel(X->dtype_, (void *)X->dptr_, (void *)Y->dptr_, size, 2,
                         (int32_t *)ishape.dims_, (int32_t *)stride.dims_);
      else
        memcpy((void *)Y->dptr_, (void *)X->dptr_, size * X->byte_);
      break;
    case 3:
      transpose_axis_dim3(X->dtype_, (void *)X->dptr_, (void *)Y->dptr_,
                          (int32_t *)ishape.dims_, (int32_t *)stride.dims_,
                          size, attrs->axes_);
      break;
    case 4:
      transpose_axis_dim4(X->dtype_, (void *)X->dptr_, (void *)Y->dptr_,
                          (int32_t *)ishape.dims_, (int32_t *)stride.dims_,
                          size, attrs->axes_);
      break;
    case 5:
      transpose_axis_dim5(X->dtype_, (void *)X->dptr_, (void *)Y->dptr_,
                          (int32_t *)ishape.dims_, (int32_t *)stride.dims_,
                          size, attrs->axes_);
      break;
    case 6:
      transpose_kernel(X->dtype_, (void *)X->dptr_, (void *)Y->dptr_, size, 6,
                       (int32_t *)ishape.dims_, (int32_t *)stride.dims_);
      break;
    default:
      return -1;
      break;
  }
  return 0;
}

#include "core/operator_template.h"
#undef __OP__
