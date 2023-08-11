#ifndef _BMMINT_VENUS_H_
#define _BMMINT_VENUS_H_

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "luna/luna_math.h"

typedef int32_t (*BMM_MAT_MUL_LUNA_API)(void *src1, void *src2, void *dst,
                                        int32_t row, int32_t col, int32_t col2,
                                        int32_t shift);
typedef int32_t (*BMM_SPLIT_MAT_MUL_LUNA_API)(void *src1, void *src2, void *dst,
                                              int32_t split_num, int32_t row,
                                              int32_t col, int32_t col2,
                                              int32_t shift);

struct luna_bmm_item {
  void *luna_api;
};

struct luna_bmm_item luna_bmm_api_list[][3] = {
    {{luna_mat_mul_q7_int8}, {luna_mat_mul_q7_int16}, {luna_mat_mul_q7_int32}},
    {{luna_mat_mul_q15_int8},
     {luna_mat_mul_q15_int16},
     {luna_mat_mul_q15_int32}},
    {{luna_mat_mul_q31_int8},
     {luna_mat_mul_q31_int16},
     {luna_mat_mul_q31_int32}},
    {{luna_split_mat_mul_q7_int8},
     {luna_split_mat_mul_q7_int16},
     {luna_split_mat_mul_q7_int32}},
    {{luna_split_mat_mul_q15_int8},
     {luna_split_mat_mul_q15_int16},
     {luna_split_mat_mul_q15_int32}},
    {{luna_split_mat_mul_q31_int8},
     {luna_split_mat_mul_q31_int16},
     {luna_split_mat_mul_q31_int32}}};

static int32_t luna_ceil(int32_t x, int32_t shift) {
  if (x & ~(0xFFFFFFFF << shift)) {
    return (x >> shift) + 1;
  } else {
    return (x >> shift);
  }
}

int32_t bmmint_luna(tTensor *lhs, tTensor *rhs, tTensor *out, tTensor *workspace) {
  int32_t ret = T_ERR_FAIL;

  if ((lhs->shape_.ndim_ != rhs->shape_.ndim_) ||
      (lhs->dtype_ != rhs->dtype_)) {
    return ret;
  }
  const int32_t left_limit = 64 * 1024;
  const int32_t right_limit = 32 * 1024;
  int32_t batch = 1;
  int32_t lhs_is_psram = 0;
  int32_t out_is_psram = 0;
  int32_t n_dim = lhs->shape_.ndim_;
  int32_t in_idx = (lhs->dtype_ & 0xF) >> 1;
  int32_t ou_idx = (out->dtype_ & 0xF) >> 1;
  int32_t M = lhs->shape_.dims_[n_dim - 2];
  int32_t N = lhs->shape_.dims_[n_dim - 1];
  int32_t L = rhs->shape_.dims_[n_dim - 1];
  void *src1 = (void *)lhs->dptr_;
  void *src2 = (void *)rhs->dptr_;
  void *dst = (void *)out->dptr_;
  void *tmp_buf = NULL;
  int32_t workspace_size = 0;
  if (workspace != NULL){
    tmp_buf = (void *)workspace->dptr_;
    workspace_size = workspace->shape_.dims_[0];
  }
  int32_t src1_offset = M * N * lhs->byte_;
  int32_t src2_offset = N * L * rhs->byte_;
  int32_t dst_offset = M * L * out->byte_;
  int32_t q_l = (int32_t)lhs->scale_;
  int32_t q_r = (int32_t)rhs->scale_;
  int32_t q_o = (int32_t)out->scale_;
  int32_t shift = q_l + q_r - q_o;
  if ((shift < 0) || (n_dim < 2) || (n_dim > 3)) {
    return ret;
  }

  if ((0 != in_idx) || (0 != ou_idx))
    return ret;

  if (3 == n_dim) {
    batch = lhs->shape_.dims_[0];
  }

  if (2 != rhs->mem_.type_)  // not support rhs on psram
  {
    return ret;
  }
  if (2 != lhs->mem_.type_)  // need copy psram to share
  {
    lhs_is_psram = 1;
  }
  
  if (2 != out->mem_.type_)  // need copy psram to share
  {
    out_is_psram = 1;
  }

  for (int32_t i = 0; i < batch; i++) {
    void *tsrc1 = (void *)((int8_t *)src1 + i * src1_offset);
    void *tsrc2 = (void *)((int8_t *)src2 + i * src2_offset);
    void *tdst = (void *)((int8_t *)dst + i * dst_offset);

    // switch (in_idx) {
    //   case 0: { // 8bit
    int32_t int8_condition_l = (luna_ceil(M, 2) << 2) * (luna_ceil(N, 3) << 3);  // right:4x8
    int32_t int8_condition_r = (luna_ceil(N, 3) << 3) * (luna_ceil(L, 2) << 2);  // right:8x4
    if (int8_condition_l > left_limit) {  //split left martrix
      int s_num = 2;
      int32_t split_M = (0 == (M % s_num)) ? (M / s_num) : (M / s_num + 1);
      int32_t final_s_M = 0;
      int8_condition_l = (luna_ceil(split_M, 2) << 2) * (luna_ceil(N, 3) << 3);  // right:4x8
      while (int8_condition_l > left_limit) {
        s_num++;
        split_M = (0 == (M % s_num)) ? (M / s_num) : (M / s_num + 1);
        int8_condition_l = (luna_ceil(split_M, 2) << 2) * (luna_ceil(N, 3) << 3);  // right:4x8
      }
      final_s_M = (0 == (M % s_num)) ? (split_M) : ( M - (split_M * (s_num - 1)));

      int32_t split_left_size = split_M * N;
      int32_t split_out_size = split_M * L;

      if (int8_condition_r <= right_limit) {
        BMM_MAT_MUL_LUNA_API luna_mat_mul_api =
          (BMM_MAT_MUL_LUNA_API)luna_bmm_api_list[in_idx][ou_idx].luna_api;            
        for (int32_t j = 0; j < s_num; j++) {              
          int8_t *p_tmp_src1 = tsrc1 + split_left_size * j;
          int8_t *p_tmp_dst = tdst + split_out_size * j;
          int32_t in_oft = split_left_size * j;
          int32_t ou_oft = split_out_size * j;
          int32_t tmp_size = 0;
          if (j == (s_num - 1)) {
            split_M = final_s_M;
            split_left_size = split_M * N;
            split_out_size = split_M * L;
          }
          if (lhs_is_psram) {
            p_tmp_src1 = (int8_t *)tmp_buf;
            tmp_size = split_left_size;
            memcpy(p_tmp_src1, tsrc1 + in_oft, split_left_size);                
          }
          if (out_is_psram) {
            p_tmp_dst = (int8_t *)tmp_buf + tmp_size;
            tmp_size += split_out_size;
            if (tmp_size > workspace_size)
              return -1;
            ret = luna_mat_mul_api(p_tmp_src1, tsrc2, p_tmp_dst, split_M, N, L, shift);
            memcpy(tdst + ou_oft, p_tmp_dst, split_out_size);                
          }
          else
          {
            ret = luna_mat_mul_api(p_tmp_src1, tsrc2, p_tmp_dst, split_M, N, L, shift);
          }
        }
      }
      else {  // big martrix split on col
        int32_t split_num = 2;
        int32_t split_L = L / split_num;
        int8_condition_r = (luna_ceil(N, 3) << 3) * (luna_ceil(split_L, 2) << 2);  // right:8x4
        while (int8_condition_r > right_limit || (0 != (L % split_num))) {
          split_num++;
          split_L = L / split_num;
          int8_condition_r = (luna_ceil(N, 3) << 3) * (luna_ceil(split_L, 2) << 2);  // right:8x4
        }

        BMM_SPLIT_MAT_MUL_LUNA_API luna_mat_mul_api =
              (BMM_SPLIT_MAT_MUL_LUNA_API)luna_bmm_api_list[in_idx + 3][ou_idx].luna_api;
        for (int32_t j = 0; j < s_num; j++) {
          int8_t *p_tmp_src1 = tsrc1 + split_left_size * j;
          int8_t *p_tmp_dst = tdst + split_out_size * j;
          int32_t in_oft = split_left_size * j;
          int32_t ou_oft = split_out_size * j;
          int32_t tmp_size = 0;
          if (j == (s_num - 1)) {
            split_M = final_s_M;
            split_left_size = split_M * N;
            split_out_size = split_M * L;
          }
          if (lhs_is_psram) {
            p_tmp_src1 = (int8_t *)tmp_buf;
            tmp_size = split_left_size;
            memcpy(p_tmp_src1, tsrc1 + in_oft, split_left_size);                
          }
          if (out_is_psram) {
            p_tmp_dst = (int8_t *)tmp_buf + tmp_size;
            tmp_size += split_out_size;
            if (tmp_size > workspace_size)
              return -1;
            ret = luna_mat_mul_api(p_tmp_src1, tsrc2, p_tmp_dst, split_num, split_M, N, L, shift);
            memcpy(tdst + ou_oft, p_tmp_dst, split_out_size);                
          }
          else
          {
            ret = luna_mat_mul_api(p_tmp_src1, tsrc2, p_tmp_dst, split_num, split_M, N, L, shift);
          }
        }
      }
    }
    else {  // left martrix <= 64KB
      int8_t *p_tmp_src1 = tsrc1;
      int8_t *p_tmp_dst = tdst;
      int32_t tmp_size = 0;
      if (lhs_is_psram) {
        p_tmp_src1 = (int8_t *)tmp_buf;
        tmp_size = M * N;
        memcpy(p_tmp_src1, tsrc1, M * N);
      }
      if (out_is_psram) {
        p_tmp_dst = (int8_t *)tmp_buf + tmp_size;
        tmp_size += M * L;
        if (tmp_size > workspace_size)
        {
          printf("workspace exceed!\n");
          return -1;
        }
      }
      if (int8_condition_r <= right_limit) {
        BMM_MAT_MUL_LUNA_API luna_mat_mul_api =
            (BMM_MAT_MUL_LUNA_API)luna_bmm_api_list[in_idx][ou_idx].luna_api;
        ret = luna_mat_mul_api(p_tmp_src1, tsrc2, p_tmp_dst, M, N, L, shift);
      }
      else {  // big martrix split on col
        int32_t split_num = 2;
        int32_t split_L = L / split_num;
        int8_condition_r = (luna_ceil(N, 3) << 3) * (luna_ceil(split_L, 2) << 2);  // right:8x4
        while (int8_condition_r > right_limit || (0 != (L % split_num))) {
          split_num++;
          split_L = L / split_num;
          int8_condition_r = (luna_ceil(N, 3) << 3) * (luna_ceil(split_L, 2) << 2);  // right:8x4
        }
        {
          BMM_SPLIT_MAT_MUL_LUNA_API luna_mat_mul_api =
              (BMM_SPLIT_MAT_MUL_LUNA_API)luna_bmm_api_list[in_idx + 3][ou_idx].luna_api;
          ret = luna_mat_mul_api(p_tmp_src1, tsrc2, p_tmp_dst, split_num, M, N, L, shift);
        }
      }
      if (out_is_psram) {
        memcpy(tdst, p_tmp_dst, M * L);     
      }
      //   }
      // } break;
  #if 0 //not support 16bit and 32bit
      case 1: {
        int32_t int16_condition_l = (luna_ceil(M, 2) << 2) *
                                    (luna_ceil(N, 1) << 1) *
                                    sizeof(int16_t);  // right:4x2
        if (int16_condition_l > left_limit) {
          return ret;
        }
        int32_t int16_condition_r = (luna_ceil(N, 1) << 1) *
                                    (luna_ceil(L, 2) << 2) *
                                    sizeof(int16_t);  // right:2x4
        if (int16_condition_r <= right_limit) {
          BMM_MAT_MUL_LUNA_API luna_mat_mul_api =
              (BMM_MAT_MUL_LUNA_API)luna_bmm_api_list[in_idx][ou_idx].luna_api;
          ret = luna_mat_mul_api(src1, src2, dst, M, N, L, shift);
        } else  // big martrix split on col
        {
          int32_t split_num = 2;
          int32_t split_L = L / split_num;
          int16_condition_r = (luna_ceil(N, 1) << 1) *
                              (luna_ceil(split_L, 2) << 2) *
                              sizeof(int16_t);  // right:2x4
          while (int16_condition_r > right_limit) {
            split_num++;
            split_L = L / split_num;
            int16_condition_r = (luna_ceil(N, 1) << 1) *
                                (luna_ceil(split_L, 2) << 2) *
                                sizeof(int16_t);  // right:2x4
          }
          if (0 != (L % split_num)) {
            return ret;
          }
          {
            BMM_SPLIT_MAT_MUL_LUNA_API luna_mat_mul_api =
                (BMM_SPLIT_MAT_MUL_LUNA_API)luna_bmm_api_list[in_idx + 3][ou_idx]
                    .luna_api;
            ret = luna_mat_mul_api(src1, src2, dst, split_num, M, N, L, shift);
          }
        }
      } break;
      case 2: {
        int32_t int32_condition_l = (luna_ceil(M, 1) << 1) *
                                    (luna_ceil(N, 1) << 1) *
                                    sizeof(int32_t);  // right:2x2
        if (int32_condition_l > left_limit) {
          return ret;
        }
        int32_t int32_condition_r = (luna_ceil(N, 1) << 1) *
                                    (luna_ceil(L, 1) << 1) *
                                    sizeof(int32_t);  // right:2x2
        if (int32_condition_r <= right_limit) {
          BMM_MAT_MUL_LUNA_API luna_mat_mul_api =
              (BMM_MAT_MUL_LUNA_API)luna_bmm_api_list[in_idx][ou_idx].luna_api;
          ret = luna_mat_mul_api(src1, src2, dst, M, N, L, shift);
        } else  // big martrix split on col
        {
          int32_t split_num = 2;
          int32_t split_L = L / split_num;
          int32_condition_r = (luna_ceil(N, 1) << 1) *
                              (luna_ceil(split_L, 1) << 1) *
                              sizeof(int32_t);  // right:2x2
          while (int32_condition_r > right_limit) {
            split_num++;
            split_L = L / split_num;
            int32_condition_r = (luna_ceil(N, 1) << 1) *
                                (luna_ceil(split_L, 1) << 1) *
                                sizeof(int32_t);  // right:2x2
          }
          if (0 != (L % split_num)) {
            return ret;
          }
          {
            BMM_SPLIT_MAT_MUL_LUNA_API luna_mat_mul_api =
                (BMM_SPLIT_MAT_MUL_LUNA_API)luna_bmm_api_list[in_idx + 3][ou_idx]
                    .luna_api;
            ret = luna_mat_mul_api(src1, src2, dst, split_num, M, N, L, shift);
          }
        }
      } break;
  #endif
      // default:
      //   break;
    }
  }

  return ret;
}
#endif  //_BMMINT_VENUS_H_
