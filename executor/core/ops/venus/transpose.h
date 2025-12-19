#ifndef _TRANSPOSE_LUNA_H_
#define _TRANSPOSE_LUNA_H_

#include <stdio.h>
#include <string.h>
#include "core/comm/utils.h"
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief API function pointer type for matrix transpose operations
 */
typedef int32_t (*MAT_TRANS_LUNA_API)(void *src, void *dst, int32_t row, int32_t col);

/**
 * @brief API function pointer type for split matrix transpose operations
 */
typedef int32_t (*MAT_TRANS_SPLIT_LUNA_API)(void *src, void *dst, int32_t row, int32_t col, int32_t split_num);

/**
 * @brief API function pointer type for matrix transpose with axis permutation
 */
typedef int32_t (*MAT_TRANS_AXIS_LUNA_API)(void *src, void *dst, int32_t *in_shape, int32_t *axis, int32_t n_dims);

/**
 * @brief Structure to hold matrix transpose API functions
 */
struct luna_mat_trans_item {
    void *luna_api;
};

/**
 * @brief List of matrix transpose API functions for different data types
 */
static struct luna_mat_trans_item luna_mat_trans_api_list[][3] = {
    {{API_LIB(mat_trans_q7)}, {API_LIB(mat_trans_q15)}, {API_LIB(mat_trans_q31)}},
    {{API_LIB(split_mat_trans_q7)}, {API_LIB(split_mat_trans_q15)}, {API_LIB(split_mat_trans_q31)}},
    {{API_LIB(trans_axis_q7)}, {API_LIB(trans_axis_q15)}, {API_LIB(trans_axis_q31)}}
};

/**
 * @brief Calculate the ceiling of x divided by 2^shift
 * @param x Input value
 * @param shift Number of bits to shift
 * @return int32_t Ceiling value
 */
static int32_t luna_ceil(int32_t x, int32_t shift) {
    if (x & ~(0xFFFFFFFF << shift)) {
        return (x >> shift) + 1;
    } else {
        return (x >> shift);
    }
}

/**
 * @brief Check if the matrix dimensions are within the allowed size limits
 * @param row Number of rows
 * @param col Number of columns
 * @param dtype Data type (e.g., Int8, Int16, Int32)
 * @return int32_t 0 if size is valid, 1 otherwise
 */
static int32_t check_trans_size_legal(int32_t row, int32_t col, int32_t dtype) {
    const int32_t max_size = 64 * 1024;
    int32_t d_byte = (dtype & 0xF);
    int32_t row_bit = 4 - (d_byte >> 1);
    int32_t col_bit = 2;
    int32_t size = (luna_ceil(row, row_bit) << row_bit) * (luna_ceil(col, col_bit) << col_bit) * d_byte;
    return size > max_size ? 1 : 0;
}

/**
 * @brief Check if the matrix dimensions and axis permutation are within the allowed size limits
 * @param in_shape Input tensor shape
 * @param axis Axis permutation
 * @param dtype Data type (e.g., Int8, Int16, Int32)
 * @return int32_t 0 if size is valid, 1 otherwise
 */
static int32_t check_trans_axis_size_legal(int32_t *in_shape, int32_t *axis, int32_t dtype) {
    const int32_t max_size = 64 * 1024;
    const int32_t max_special_size = 32 * 1024;
    int32_t c = in_shape[0];
    int32_t h = in_shape[1];
    int32_t w = in_shape[2];
    int32_t d_byte = (dtype & 0xF);
    int32_t row_bit = 4 - (d_byte >> 1);
    int32_t col_bit = 2;
    int32_t size = 0;
    int32_t special_size = h * w;

    if ((0 == axis[0] && 2 == axis[1] && 1 == axis[2]) ||
        (1 == axis[0] && 0 == axis[1] && 2 == axis[2]) ||
        (1 == axis[0] && 2 == axis[1] && 0 == axis[2]) ||
        (2 == axis[0] && 0 == axis[1] && 1 == axis[2]) ||
        (2 == axis[0] && 1 == axis[1] && 0 == axis[2])) {
        size = (luna_ceil(h, row_bit) << row_bit) * (luna_ceil(w, col_bit) << col_bit) * d_byte;
    }

    return (size > max_size) || (special_size > max_special_size) ? 1 : 0;
}

/**
 * @brief Perform matrix transpose operation
 * @param dtype Data type (e.g., Int8, Int16, Int32)
 * @param dst Output matrix pointer
 * @param src Input matrix pointer
 * @param row Number of rows
 * @param col Number of columns
 * @return int32_t Operation status
 */
int32_t transpose_luna(int16_t dtype, void *dst, const void *src, int32_t row, int32_t col) {
    int32_t ret = T_ERR_FAIL;
    int32_t idx = (dtype & 0xF) >> 1;

    if (!check_trans_size_legal(row, col, dtype)) {  // Size <= 64K
        MAT_TRANS_LUNA_API luna_trans_api = (MAT_TRANS_LUNA_API)luna_mat_trans_api_list[0][idx].luna_api;
        luna_trans_api((void *)src, dst, row, col);
        ret = T_SUCCESS;
    } else {  // Size > 64K, need to split
        int32_t split_num = 2;
        int32_t split_row = row / split_num;
        while (check_trans_size_legal(split_row, col, dtype)) {
            split_num++;
            split_row = row / split_num;
        }
        if (row % split_num != 0) {
            return ret;
        }
        MAT_TRANS_SPLIT_LUNA_API luna_trans_api = (MAT_TRANS_SPLIT_LUNA_API)luna_mat_trans_api_list[1][idx].luna_api;
        ret = luna_trans_api((void *)src, dst, row, col, split_num);
    }

    return ret;
}

/**
 * @brief Perform matrix transpose with axis permutation
 * @param dtype Data type (e.g., Int8, Int16, Int32)
 * @param src Input tensor pointer
 * @param dst Output tensor pointer
 * @param in_shape Input tensor shape
 * @param axis Axis permutation
 * @param n_dims Number of dimensions
 * @return int32_t Operation status
 */
int32_t transpose_axis_luna(int16_t dtype, void *src, void *dst, int32_t *in_shape, int32_t *axis, uint32_t n_dims) {
    int32_t ret = T_ERR_FAIL;

    if (n_dims != 3 || check_trans_axis_size_legal(in_shape, axis, dtype)) {
        return ret;
    }

    int32_t idx = (dtype & 0xF) >> 1;
    MAT_TRANS_AXIS_LUNA_API luna_trans_api = (MAT_TRANS_AXIS_LUNA_API)luna_mat_trans_api_list[2][idx].luna_api;
    ret = luna_trans_api(src, dst, in_shape, axis, n_dims);

    return ret;
}

#endif