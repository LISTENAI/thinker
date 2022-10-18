/***************************************************************************
 * luna_matrix_math.h                                                       *
 *                                                                         *
 * Copyright (C) 2020 listenai Co.Ltd                   			       *
 * All rights reserved.                                                    *
 ***************************************************************************/
#ifndef __LUNA_MATRIX_MATH_H__
#define __LUNA_MATRIX_MATH_H__


#include "luna_math_types.h"

/**
 * @brief Multiplication of two q7 matrices.
 * @param[in]       *src1 points to the first input matrix.
 * @param[in]       *src2 points to the second input matrix.
 * @param[out]      *dst  points to the output matrix.
 * @param[in]       row   number of the first input matrix rows.
 * @param[in]       col   number of the first input matrix columns.
 * @param[in]       col2  number of the second input matrix columns.
 * @return none.
 *
 * <b>Function notes:</b>
 *
 * The 1.15 format input is multiplied yields a 2.30 format, and then added
 * without saturation to a 64-bit accumulator in 34.30 format. Finally,
 * the added output is truncated to 34.15 format by discarding the lower 15
 * bits, and then saturated to yield a result in 1.15 format.
 */
int32_t luna_mat_mul_q7_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_mat_mul_q7_int16(const q7_t *src1, const q7_t *src2, q15_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_mat_mul_q7_int32(const q7_t *src1, const q7_t *src2, q31_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

/**
 * @brief Multiplication of two q15 matrices.
 * @param[in]       *src1 points to the first input matrix.
 * @param[in]       *src2 points to the second input matrix.
 * @param[out]      *dst  points to the output matrix.
 * @param[in]       row   number of the first input matrix rows.
 * @param[in]       col   number of the first input matrix columns.
 * @param[in]       col2  number of the second input matrix columns.
 * @return none.
 *
 * <b>Function notes:</b>
 *
 * The 1.15 format input is multiplied yields a 2.30 format, and then added
 * without saturation to a 64-bit accumulator in 34.30 format. Finally,
 * the added output is truncated to 34.15 format by discarding the lower 15
 * bits, and then saturated to yield a result in 1.15 format.
 */
int32_t luna_mat_mul_q15_int8(const q15_t *src1, const q15_t *src2, q7_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_mat_mul_q15_int16(const q15_t *src1, const q15_t *src2, q15_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_mat_mul_q15_int32(const q15_t *src1, const q15_t *src2, q31_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);


/**
 * @brief Multiplication of two q31 matrices.
 * @param[in]       *src1 points to the first input matrix.
 * @param[in]       *src2 points to the second input matrix.
 * @param[out]      *dst  points to the output matrix.
 * @param[in]       row   number of the first input matrix rows.
 * @param[in]       col   number of the first input matrix columns.
 * @param[in]       col2  number of the second input matrix columns.
 * @return none.
 *
 * <b>Function notes:</b>
 *
 * The 1.15 format input is multiplied yields a 2.30 format, and then added
 * without saturation to a 64-bit accumulator in 34.30 format. Finally,
 * the added output is truncated to 34.15 format by discarding the lower 15
 * bits, and then saturated to yield a result in 1.15 format.
 */
int32_t luna_mat_mul_q31_int8(const q31_t *src1, const q31_t *src2, q7_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_mat_mul_q31_int16(const q31_t *src1, const q31_t *src2, q15_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_mat_mul_q31_int32(const q31_t *src1, const q31_t *src2, q31_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t luna_mat_mul_q7q3_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_mat_mul_q7q3_int16(const q7_t *src1, const q7_t *src2, q15_t *dst,	uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_mat_mul_q7q3_int32(const q7_t *src1, const q7_t *src2, q31_t *dst,	uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_mat_mul_q31q15_int32(const q31_t *src1, const q15_t *src2, q31_t *dst,	uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t luna_mat_trans_q7(const q7_t *src1, q7_t *dst, uint32_t row, uint32_t col);
int32_t luna_mat_trans_q15(const q15_t *src1, q15_t *dst, uint32_t row, uint32_t col);
int32_t luna_mat_trans_q31(const q31_t *src1, q31_t *dst, uint32_t row, uint32_t col);

int32_t luna_group_mat_mul_q7_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_group_mat_mul_q7_int16(const q7_t *src1, const q7_t *src2, q15_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_group_mat_mul_q7_int32(const q7_t *src1, const q7_t *src2, q31_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_group_mat_mul_q15_int32(const q15_t *src1, const q15_t *src2, q31_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_group_mat_mul_q7q3_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_group_mat_mul_q7q3_int16(const q7_t *src1, const q7_t *src2, q15_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_group_mat_mul_q7q3_int32(const q7_t *src1, const q7_t *src2, q31_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t luna_group_mat_mul(const void *src1, const void *src2, void *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift, uint32_t in_bit, uint32_t out_bit);

int32_t luna_mat_trans_inv_q7(const q7_t *src, q7_t *dst, uint32_t row, uint32_t col, uint32_t i_inv, uint32_t o_inv);
int32_t luna_mat_trans_inv_q15(const q15_t *src, q15_t *dst, uint32_t row, uint32_t col, uint32_t i_inv, uint32_t o_inv);
int32_t luna_mat_trans_inv_q31(const q31_t *src, q31_t *dst, uint32_t row, uint32_t col, uint32_t i_inv, uint32_t o_inv);
int32_t luna_mat_trans_col234_inv_q31(const q31_t *src, q31_t *dst, uint32_t row, uint32_t col, uint32_t i_inv, uint32_t o_inv);

int32_t luna_mat_mul_inv_q7_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);
int32_t luna_mat_mul_inv_q7_int16(const q7_t *src1, const q7_t *src2, q15_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);
int32_t luna_mat_mul_inv_q7_int32(const q7_t *src1, const q7_t *src2, q31_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);
int32_t luna_mat_mul_inv_q15_int8(const q15_t *src1, const q15_t *src2, q7_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);
int32_t luna_mat_mul_inv_q15_int16(const q15_t *src1, const q15_t *src2, q15_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);
int32_t luna_mat_mul_inv_q15_int32(const q15_t *src1, const q15_t *src2, q31_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);
int32_t luna_mat_mul_inv_q31_int8(const q31_t *src1, const q31_t *src2, q7_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);
int32_t luna_mat_mul_inv_q31_int16(const q31_t *src1, const q31_t *src2, q15_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);
int32_t luna_mat_mul_inv_q31_int32(const q31_t *src1, const q31_t *src2, q31_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);
int32_t luna_mat_mul_inv_q7q3_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);
int32_t luna_mat_mul_inv_q7q3_int16(const q7_t *src1, const q7_t *src2, q15_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);
int32_t luna_mat_mul_inv_q7q3_int32(const q7_t *src1, const q7_t *src2, q31_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);

int32_t luna_split_mat_mul_q7_int8(const q7_t *src1, const q7_t *src2, int8_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_q7_int16(const q7_t *src1, const q7_t *src2, int16_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_q7_int32(const q7_t *src1, const q7_t *src2, int32_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_q15_int8(const q15_t *src1, const q15_t *src2, int8_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_q15_int16(const q15_t *src1, const q15_t *src2, int16_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_q15_int32(const q15_t *src1, const q15_t *src2, int32_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_q31_int8(const q31_t *src1, const q31_t *src2, int8_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_q31_int16(const q31_t *src1, const q31_t *src2, int16_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_q31_int32(const q31_t *src1, const q31_t *src2, int32_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_q7q3_int8(const q7_t *src1, const q7_t *src2, int8_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_q7q3_int16(const q7_t *src1, const q7_t *src2, int16_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
int32_t luna_split_mat_mul_q7q3_int32(const q7_t *src1, const q7_t *src2, int32_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t luna_split_mat_trans_q7(const q7_t *src, q7_t *dst, uint32_t row, uint32_t col, uint32_t split_num);
int32_t luna_split_mat_trans_q15(const q15_t *src, q15_t *dst, uint32_t row, uint32_t col, uint32_t split_num);
int32_t luna_split_mat_trans_q31(const q31_t *src, q31_t *dst, uint32_t row, uint32_t col, uint32_t split_num);

int32_t luna_trans_axis_q7(const q7_t *src, q7_t *dst, uint32_t *in_shape, uint32_t *axis, uint32_t n_dims);
int32_t luna_trans_axis_q15(const q15_t *src, q15_t *dst, uint32_t *in_shape, uint32_t *axis, uint32_t n_dims);
int32_t luna_trans_axis_q31(const q31_t *src, q31_t *dst, uint32_t *in_shape, uint32_t *axis, uint32_t n_dims);

#endif // __LUNA_MATRIX_MATH_H__
