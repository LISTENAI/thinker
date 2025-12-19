/***************************************************************************
 * luna_matrix_math.h                                                       *
 *                                                                         *
 * Copyright (C) 2020 listenai Co.Ltd                   			       *
 * All rights reserved.                                                    *
 ***************************************************************************/
#ifndef __LUNA_MATRIX_MATH_H__
#define __LUNA_MATRIX_MATH_H__

#include "luna_math_types.h"

int32_t luna_mat_mul_i8i8o8(const int8_t *src1, const int8_t *src2, int8_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_mat_mul_i8i8o16(const int8_t *src1, const int8_t *src2, int16_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_mat_mul_i8i8o32(const int8_t *src1, const int8_t *src2, int32_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_mat_mul_i16i16o8(const int16_t *src1, const int16_t *src2, int8_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_mat_mul_i16i16o16(const int16_t *src1, const int16_t *src2, int16_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_mat_mul_i16i16o32(const int16_t *src1, const int16_t *src2, int32_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_mat_mul_i32i32o8(const int32_t *src1, const int32_t *src2, int8_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_mat_mul_i32i32o16(const int32_t *src1, const int32_t *src2, int16_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_mat_mul_i32i32o32(const int32_t *src1, const int32_t *src2, int32_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t luna_mat_mul_inv_i8i8o8(const int8_t *src1, const int8_t *src2, int8_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);
int32_t luna_mat_mul_inv_i8i8o16(const int8_t *src1, const int8_t *src2, int16_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);
int32_t luna_mat_mul_inv_i8i8o32(const int8_t *src1, const int8_t *src2, int32_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);
int32_t luna_mat_mul_inv_i16i16o8(const int16_t *src1, const int16_t *src2, int8_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);
int32_t luna_mat_mul_inv_i16i16o16(const int16_t *src1, const int16_t *src2, int16_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);
int32_t luna_mat_mul_inv_i16i16o32(const int16_t *src1, const int16_t *src2, int32_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);
int32_t luna_mat_mul_inv_i32i32o8(const int32_t *src1, const int32_t *src2, int8_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);
int32_t luna_mat_mul_inv_i32i32o16(const int32_t *src1, const int32_t *src2, int16_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);
int32_t luna_mat_mul_inv_i32i32o32(const int32_t *src1, const int32_t *src2, int32_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);

int32_t luna_split_mat_mul_i8i8o8(const int8_t *src1, const int8_t *src2, int8_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_i8i8o16(const int8_t *src1, const int8_t *src2, int16_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_i8i8o32(const int8_t *src1, const int8_t *src2, int32_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_i16i16o8(const int16_t *src1, const int16_t *src2, int8_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_i16i16o16(const int16_t *src1, const int16_t *src2, int16_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_i16i16o32(const int16_t *src1, const int16_t *src2, int32_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_i32i32o8(const int32_t *src1, const int32_t *src2, int8_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_i32i32o16(const int32_t *src1, const int32_t *src2, int16_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_i32i32o32(const int32_t *src1, const int32_t *src2, int32_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t luna_group_mat_mul_i8i8o8(const int8_t *src1, const int8_t *src2, int8_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_group_mat_mul_i8i8o16(const int8_t *src1, const int8_t *src2, int16_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_group_mat_mul_i8i8o32(const int8_t *src1, const int8_t *src2, int32_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_group_mat_mul_i16i16o8(const int16_t *src1, const int16_t *src2, int8_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_group_mat_mul_i16i16o16(const int16_t *src1, const int16_t *src2, int16_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_group_mat_mul_i16i16o32(const int16_t *src1, const int16_t *src2, int32_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_group_mat_mul_i32i32o8(const int32_t *src1, const int32_t *src2, int8_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_group_mat_mul_i32i32o16(const int32_t *src1, const int32_t *src2, int16_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_group_mat_mul_i32i32o32(const int32_t *src1, const int32_t *src2, int32_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t luna_split_mat_mul_bias_i8i8i32o8(const int8_t *src1, const int8_t *src2, const int32_t *bias, int8_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_bias_i8i8i32o16(const int8_t *src1, const int8_t *src2, const int32_t *bias, int16_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_bias_i8i8i32o32(const int8_t *src1, const int8_t *src2, const int32_t *bias, int32_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_bias_i16i16i32o8(const int16_t *src1, const int16_t *src2, const int32_t *bias, int8_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_bias_i16i16i32o16(const int16_t *src1, const int16_t *src2, const int32_t *bias, int16_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_bias_i16i16i32o32(const int16_t *src1, const int16_t *src2, const int32_t *bias, int32_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_bias_i32i32i32o8(const int32_t *src1, const int32_t *src2, const int32_t *bias, int8_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_bias_i32i32i32o16(const int32_t *src1, const int32_t *src2, const int32_t *bias, int16_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_bias_i32i32i32o32(const int32_t *src1, const int32_t *src2, const int32_t *bias, int32_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_bias_i4i8i32o8(const int8_t *src1, const int8_t *src2, const int32_t *bias, int8_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_bias_i4i8i32o16(const int8_t *src1, const int8_t *src2, const int32_t *bias, int16_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_bias_i4i8i32o32(const int8_t *src1, const int8_t *src2, const int32_t *bias, int32_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_bias_i8i4i32o8(const int8_t *src1, const int8_t *src2, const int32_t *bias, int8_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_bias_i8i4i32o16(const int8_t *src1, const int8_t *src2, const int32_t *bias, int16_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_bias_i8i4i32o32(const int8_t *src1, const int8_t *src2, const int32_t *bias, int32_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t luna_mat_trans_i8o8(const int8_t *src1, int8_t *dst, uint32_t row, uint32_t col);
int32_t luna_mat_trans_i16o16(const int16_t *src1, int16_t *dst, uint32_t row, uint32_t col);
int32_t luna_mat_trans_i32o32(const int32_t *src1, int32_t *dst, uint32_t row, uint32_t col);

int32_t luna_mat_trans_inv_i8o8(const int8_t *src, int8_t *dst, uint32_t row, uint32_t col, uint32_t i_inv, uint32_t o_inv);
int32_t luna_mat_trans_inv_i16o16(const int16_t *src, int16_t *dst, uint32_t row, uint32_t col, uint32_t i_inv, uint32_t o_inv);
int32_t luna_mat_trans_inv_i32o32(const int32_t *src, int32_t *dst, uint32_t row, uint32_t col, uint32_t i_inv, uint32_t o_inv);

int32_t luna_split_mat_trans_i8o8(const int8_t *src, int8_t *dst, uint32_t row, uint32_t col);
int32_t luna_split_mat_trans_i16o16(const int16_t *src, int16_t *dst, uint32_t row, uint32_t col);
int32_t luna_split_mat_trans_i32o32(const int32_t *src, int32_t *dst, uint32_t row, uint32_t col);

int32_t luna_trans_axis_i8o8(const int8_t *src, int8_t *dst, uint32_t *in_shape, uint32_t *axis, uint32_t n_dims);
int32_t luna_trans_axis_i16o16(const int16_t *src, int16_t *dst, uint32_t *in_shape, uint32_t *axis, uint32_t n_dims);
int32_t luna_trans_axis_i32o32(const int32_t *src, int32_t *dst, uint32_t *in_shape, uint32_t *axis, uint32_t n_dims);

int32_t luna_mat_copy_i8o8(int8_t* src, int8_t* dst, uint32_t channel, uint32_t row, uint32_t col, 
	uint32_t i_planar_inv, uint32_t i_row_inv, uint32_t o_planar_inv, uint32_t o_row_inv);
int32_t luna_mat_copy_i16o16(int16_t* src, int16_t* dst, uint32_t channel, uint32_t row, uint32_t col, 
	uint32_t i_planar_inv, uint32_t i_row_inv, uint32_t o_planar_inv, uint32_t o_row_inv);
int32_t luna_mat_copy_i32o32(int32_t* src, int32_t* dst, uint32_t channel, uint32_t row, uint32_t col, 
	uint32_t i_planar_inv, uint32_t i_row_inv, uint32_t o_planar_inv, uint32_t o_row_inv);

#endif // __LUNA_MATRIX_MATH_H__
