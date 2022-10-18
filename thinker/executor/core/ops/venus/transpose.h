
#include <stdio.h>
#include <string.h>

#include "core/comm/utils.h"
#include "luna/luna_math.h"

typedef int32_t (*MAT_TRANS_LUNA_API)(void *src, void *dst, int32_t row,
                                      int32_t col);
typedef int32_t (*MAT_TRANS_SPLIT_LUNA_API)(void *src, void *dst, int32_t row,
                                            int32_t col, int32_t split_num);
typedef int32_t (*MAT_TRANS_AXIS_LUNA_API)(void *src, void *dst,
                                           int32_t *in_shape, int32_t *axis,
                                           int32_t n_dims);

struct luna_mat_trans_item {
  void *luna_api;
};

struct luna_mat_trans_item luna_mat_trans_api_list[][3] = {
    {{luna_mat_trans_q7}, {luna_mat_trans_q15}, {luna_mat_trans_q31}},
    {{luna_split_mat_trans_q7},
     {luna_split_mat_trans_q15},
     {luna_split_mat_trans_q31}},
    {{luna_trans_axis_q7}, {luna_trans_axis_q15}, {luna_trans_axis_q31}}};

static int32_t luna_ceil(int32_t x, int32_t shift) {
  if (x & ~(0xFFFFFFFF << shift)) {
    return (x >> shift) + 1;
  } else {
    return (x >> shift);
  }
}

int32_t check_trans_size_legal(
    int32_t row, int32_t col,
    int32_t dtype) {  // 8bit:16x4  16bit:8x4  32bit:4x4
  const int32_t max_size = 64 * 1024;
  int32_t size = 0;
  int32_t d_byte = (dtype & 0xF);
  int32_t row_bit = 4 - (d_byte >> 1);
  int32_t col_bit = 2;
  size = (luna_ceil(row, row_bit) << row_bit) *
         (luna_ceil(col, col_bit) << col_bit) * d_byte;
  if (size > max_size) {
    return 1;
  }
  return 0;
}

int32_t check_trans_axis_size_legal(int32_t *in_shape, int32_t *axis,
                                    int32_t dtype) {  // only support 3d
  const int32_t max_size = 64 * 1024;
  const int32_t max_special_size = 32 * 1024;
  int32_t c = in_shape[0];
  int32_t h = in_shape[1];
  int32_t w = in_shape[2];
  int32_t size = 0;
  int32_t special_size = 0;
  int32_t d_byte = (dtype & 0xF);
  int32_t row_bit = 4 - (d_byte >> 1);
  int32_t col_bit = 2;

  if (0 == axis[0] && 2 == axis[1] && 1 == axis[2])  // 0,2,1
  {
    size = (luna_ceil(h, row_bit) << row_bit) *
           (luna_ceil(w, col_bit) << col_bit) * d_byte;
  } else if (1 == axis[0] && 0 == axis[1] && 2 == axis[2])  // 1,0,2
  {
    size = (luna_ceil(h, row_bit) << row_bit) *
           (luna_ceil(w, col_bit) << col_bit) * d_byte;
  } else if (1 == axis[0] && 2 == axis[1] && 0 == axis[2])  // 1,2,0
  {
    size = (luna_ceil(c, row_bit) << row_bit) *
           (luna_ceil(w, col_bit) << col_bit) * d_byte;
  } else if (2 == axis[0] && 0 == axis[1] && 1 == axis[2])  // 2,0,1
  {
    size = (luna_ceil(h, row_bit) << row_bit) *
           (luna_ceil(w, col_bit) << col_bit) * d_byte;
  } else if (2 == axis[0] && 1 == axis[1] && 0 == axis[2])  // 2,1,0
  {
    size = (luna_ceil(c, row_bit) << row_bit) *
           (luna_ceil(w, col_bit) << col_bit) * d_byte;
  }
  special_size = h * w;
  if ((size > max_size) || (special_size > max_size)) {
    return 1;
  }
  return 0;
}

int32_t transpose_luna(int16_t dtype, void *dst, const void *src, int32_t row,
                       int32_t col) {
  // memcpy((void *)dst, (void *)src, row * col * (dtype & 0xF));
  // return 0;

  int32_t ret = T_ERR_FAIL;
  int32_t idx = (dtype & 0xF) >> 1;
  if (0 == check_trans_size_legal(row, col, dtype))  // size <= 64K
  {
    MAT_TRANS_LUNA_API luna_trans_api =
        (MAT_TRANS_LUNA_API)luna_mat_trans_api_list[0][idx].luna_api;
    luna_trans_api((void *)src, dst, row, col);
  } else  // size larger than 64K, need split, only support split row
  {
    int32_t split_num = 2;
    int32_t split_row = row / split_num;
    while (check_trans_size_legal(split_row, col, dtype)) {
      split_num++;
      split_row = row / split_num;
    }
    if (0 != (row % split_num)) {
      return ret;
    }
    {
      MAT_TRANS_SPLIT_LUNA_API luna_trans_api =
          (MAT_TRANS_SPLIT_LUNA_API)luna_mat_trans_api_list[1][idx].luna_api;
      ret = luna_trans_api((void *)src, dst, row, col, split_num);
    }
  }

  return ret;
}

int32_t transpose_axis_luna(int16_t dtype, void *src, void *dst,
                            int32_t *in_shape, int32_t *axis, uint32_t n_dims) {
  int32_t ret = T_ERR_FAIL;

  if ((3 != n_dims) || check_trans_axis_size_legal(in_shape, axis, dtype)) {
    return ret;
  }
  int32_t idx = (dtype & 0xF) >> 1;
  MAT_TRANS_AXIS_LUNA_API luna_trans_api =
      (MAT_TRANS_AXIS_LUNA_API)luna_mat_trans_api_list[2][idx].luna_api;
  ret = luna_trans_api(src, dst, in_shape, axis, n_dims);

  return ret;
}
