#ifndef _LINEARINT_LUNA_H_
#define _LINEARINT_LUNA_H_
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "luna/luna_math.h"
#include "thinker_define.h"

typedef int32_t (*FC_MAT_MUL_LUNA_API)(int8_t *src1, int8_t *src2, void *dst,
                                       int32_t row, int32_t col, int32_t col2,
                                       int32_t shift);
typedef int32_t (*FC_SPLIT_MAT_MUL_LUNA_API)(int8_t *src1, int8_t *src2,
                                             void *dst, int32_t split_num,
                                             int32_t row, int32_t col,
                                             int32_t col2, int32_t shift);
typedef int32_t (*FC_VEC_ADD_LUNA_API)(void *src1, void *src2, void *dst,
                                       int32_t size, int32_t shift);

struct fc_luna_api_item {
  void *luna_api;
};

struct fc_luna_api_item fc_luna_api_list[][3] = {
    {{luna_mat_mul_q7_int8}, {luna_mat_mul_q7_int16}, {luna_mat_mul_q7_int32}},
    {{luna_split_mat_mul_q7_int8},
     {luna_split_mat_mul_q7_int16},
     {luna_split_mat_mul_q7_int32}},
    {{luna_add_q7_int8}, {luna_add_q7_int16}, {luna_add_q7_int32}},
    {{luna_add_q15_int8}, {luna_add_q15_int16}, {luna_add_q15_int32}},
    {{luna_add_q31_int8}, {luna_add_q31_int16}, {luna_add_q31_int32}}};

static int32_t luna_ceil(int32_t x, int32_t shift) {
  if (x & ~(0xFFFFFFFF << shift)) {
    return (x >> shift) + 1;
  } else {
    return (x >> shift);
  }
}

static int32_t calc_linearint_luna(tTensor *input, tTensor *weight,
                                   tTensor *bias, tTensor *output,
                                   tTensor *tmp) {
  int32_t ret = T_ERR_FAIL;
  tShape new_shape;
  if (1 == input->shape_.ndim_) {
    new_shape.ndim_ = 2;
    new_shape.dims_[1] = input->shape_.dims_[0];
    new_shape.dims_[0] = 1;
  } else if (input->shape_.ndim_ == 3) {
    new_shape.ndim_ = 2;
    new_shape.dims_[0] = input->shape_.dims_[0] * input->shape_.dims_[1];
    new_shape.dims_[1] = input->shape_.dims_[2];
  } else {
    memcpy(&new_shape, &input->shape_, sizeof(tShape));
  }

  int32_t out_is_psram = 0;
  const int32_t left_limit = 64 * 1024;
  const int32_t right_limit = 32 * 1024;
  // int32_t in_idx = (input->dtype_ & 0xF) >> 1;
  int32_t ou_idx = (output->dtype_ & 0xF) >> 1;
  int32_t bias_idx = 0;
  int32_t n_dim = new_shape.ndim_;
  int32_t M = new_shape.dims_[n_dim - 2];
  int32_t N = new_shape.dims_[n_dim - 1];
  int32_t L = weight->shape_.dims_[n_dim - 1];
  int8_t *p_in = (int8_t *)input->dptr_;
  int8_t *p_weight = (int8_t *)weight->dptr_;
  void *p_out = (void *)output->dptr_;
  void *p_bis = NULL;
  int32_t q_i = (int32_t)input->scale_;
  int32_t q_w = (int32_t)weight->scale_;
  int32_t q_o = (int32_t)output->scale_;
  int32_t shift = q_i + q_w - q_o;
  int32_t has_bias = 0;

  if ((1 == output->mem_.type_ || 3 == output->mem_.type_) && (tmp) &&
      (!bias))  // need copy psram to share
  {
    out_is_psram = 1;
    p_out = (void *)tmp->dptr_;
  }

  if (bias) {
    bias_idx = (bias->dtype_ & 0xF) >> 1;
    p_bis = (void *)bias->dptr_;
    has_bias = (bias->shape_.ndim_) ? 1 : 0;
  }

  if (shift < 0) {
    return ret;
  }

  switch (input->dtype_) {
    case Int8:  // weight only suppoer int8
    {
      int32_t int8_condition_l =
          (luna_ceil(M, 2) << 2) * (luna_ceil(N, 3) << 3);  // right:4x8
      if (int8_condition_l > left_limit) {
        return ret;
      }
      int32_t int8_condition_r =
          (luna_ceil(N, 3) << 3) * (luna_ceil(L, 2) << 2);  // right:8x4
      if (int8_condition_r <= right_limit) {
        FC_MAT_MUL_LUNA_API luna_mat_mul_api;
        if (has_bias) {
          int32_t *p_tmp = (int32_t *)tmp->dptr_;
          luna_mat_mul_api =
              (FC_MAT_MUL_LUNA_API)fc_luna_api_list[0][bias_idx].luna_api;
          ret = luna_mat_mul_api(p_in, p_weight, p_tmp, M, N, L, 0);
        } else {
          luna_mat_mul_api =
              (FC_MAT_MUL_LUNA_API)fc_luna_api_list[0][ou_idx].luna_api;
          ret = luna_mat_mul_api(p_in, p_weight, p_out, M, N, L, shift);
        }
      } else  // big martrix split on col
      {
        int32_t split_num = 2;
        int32_t split_L = L / split_num;
        int8_condition_r =
            (luna_ceil(N, 3) << 3) * (luna_ceil(split_L, 2) << 2);  // right:8x4
        while (int8_condition_r > right_limit || (0 != (L % split_num))) {
          split_num++;
          split_L = L / split_num;
          int8_condition_r = (luna_ceil(N, 3) << 3) *
                             (luna_ceil(split_L, 2) << 2);  // right:8x4
        }
        {
          FC_SPLIT_MAT_MUL_LUNA_API luna_mat_mul_api;
          if (has_bias) {
            int32_t *p_tmp = (int32_t *)tmp->dptr_;
            luna_mat_mul_api =
                (FC_SPLIT_MAT_MUL_LUNA_API)fc_luna_api_list[1][bias_idx]
                    .luna_api;
            ret =
                luna_mat_mul_api(p_in, p_weight, p_tmp, split_num, M, N, L, 0);
          } else {
            luna_mat_mul_api =
                (FC_SPLIT_MAT_MUL_LUNA_API)fc_luna_api_list[1][ou_idx].luna_api;
            ret = luna_mat_mul_api(p_in, p_weight, p_out, split_num, M, N, L,
                                   shift);
          }
        }
      }

      if (has_bias) {
        for (int32_t i = 0; i < M; i++)  // add bias
        {
          int32_t *p_tmp = (int32_t *)tmp->dptr_;
          FC_VEC_ADD_LUNA_API luna_add_api =
              (FC_VEC_ADD_LUNA_API)fc_luna_api_list[2 + bias_idx][ou_idx]
                  .luna_api;
          int8_t *tsrc1 = (int8_t *)p_tmp + i * L * (bias->dtype_ & 0xF);
          int8_t *tdst = (int8_t *)p_out + i * L * (output->dtype_ & 0xF);
          ret |= luna_add_api(tsrc1, p_bis, tdst, L, shift);
        }
      }
    } break;
  }

  if (out_is_psram) {
    memcpy((void *)output->dptr_, p_out, M * L);
  }

  return ret;
}

int32_t linearint_luna(tTensor *input, tTensor *weight, tTensor *bias,
                       LinearIntAttrs *attrs, tTensor *workspace,
                       tTensor *output) {
  int32_t ret = T_ERR_FAIL;

  //    if ((2 != input->shape_.ndim_) || (input->shape_.ndim_ !=
  //    weight->shape_.ndim_) || (input->dtype_ != weight->dtype_))
  //    {
  //        return ret;
  //    }
  ret = calc_linearint_luna(input, weight, bias, output, workspace);

  return ret;
}

#endif
