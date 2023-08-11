#ifndef _MATMUL_LUNA_H_
#define _MATMUL_LUNA_H_

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"

static void mtx_mpyf(void *pScr, float *z, const float *x, const float *y,
                     int32_t M, int32_t N, int32_t P) {
  for (int32_t i = 0; i < M; i++) {
    for (int32_t j = 0; j < P; j++) {
      float sum = 0;
      for (int32_t k = 0; k < N; k++) {
        sum += x[i * N + k] * y[k * P + j];
      }
      z[i * P + j] = sum;
    }
  }
}

int32_t calc_matmul_hifi_float(tTensor *lhs, tTensor *rhs, tTensor *out,
                               int32_t batch) {
  int32_t n_dim = lhs->shape_.ndim_;
  int32_t M = lhs->shape_.dims_[n_dim - 2];
  int32_t N = lhs->shape_.dims_[n_dim - 1];
  int32_t L = rhs->shape_.dims_[n_dim - 1];
  float *src1 = (float *)lhs->dptr_;
  float *src2 = (float *)rhs->dptr_;
  float *dst = (float *)out->dptr_;
  int32_t src1_offset = M * N;
  int32_t src2_offset = N * L;
  int32_t dst_offset = M * L;
  for (int32_t i = 0; i < batch; i++) {
    float *tsrc1 = src1 + i * src1_offset;
    float *tsrc2 = src2 + i * src2_offset;
    float *tdst = dst + i * dst_offset;
    mtx_mpyf(NULL, tdst, tsrc1, tsrc2, M, N, L);
  }

  return 0;
}

int32_t matmul_luna(tTensor *lhs, tTensor *rhs, tTensor *out) {
  int32_t ret = T_ERR_FAIL;

  if ((lhs->shape_.ndim_ != rhs->shape_.ndim_) ||
      (lhs->dtype_ != rhs->dtype_)) {
    return ret;
  }

  if (Float32 == lhs->dtype_) {
    int32_t batch = 1;
    int32_t n_dim = lhs->shape_.ndim_;
    if (3 == n_dim) {
      batch = lhs->shape_.dims_[0];
    }
    ret = calc_matmul_hifi_float(lhs, rhs, out, batch);

    return ret;
  }

  return ret;
}
#