#ifndef __NNBLAS_MATRIX_MATH_H__
#define __NNBLAS_MATRIX_MATH_H__

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
int32_t nnblas_mat_mul_q7_int8(const q7_t *src1, const q7_t *src2, q7_t *dst,
		uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_mat_mul_q7_int16(const q7_t *src1, const q7_t *src2, q15_t *dst,
		uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_mat_mul_q7_int32(const q7_t *src1, const q7_t *src2, q31_t *dst,
		uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

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
int32_t nnblas_mat_mul_q15_int8(const q15_t *src1, const q15_t *src2, q7_t *dst,
		uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_mat_mul_q15_int16(const q15_t *src1, const q15_t *src2, q15_t *dst,
		uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_mat_mul_q15_int32(const q15_t *src1, const q15_t *src2, q31_t *dst,
		uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
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
int32_t nnblas_mat_mul_q31_int8(const q31_t *src1, const q31_t *src2, q7_t *dst,
		uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_mat_mul_q31_int16(const q31_t *src1, const q31_t *src2, q15_t *dst,
		uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_mat_mul_q31_int32(const q31_t *src1, const q31_t *src2, q31_t *dst,
		uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

/*//col2 % 2 == 0
int32_t nnblas_mat_mul_q7q3_int8(const q7_t *src1, const q7_t *src2, q7_t *dst,
		uint32_t row, uint32_t col, uint32_t col2, uint32_t shift)
{
	return nnblas_mat_mul((void *)src1, (void *)src2, (void *)dst, row, col, col2, shift, (uint32_t *)luna_matrix_mul_b4bit2b);
}

//col2 % 2 == 0
int32_t nnblas_mat_mul_q7q3_int16(const q7_t *src1, const q7_t *src2, q15_t *dst,
		uint32_t row, uint32_t col, uint32_t col2, uint32_t shift)
{
	return nnblas_mat_mul((void *)src1, (void *)src2, (void *)dst, row, col, col2, shift, (uint32_t *)luna_matrix_mul_b4bit2h);
}

//col2 % 2 == 0
int32_t nnblas_mat_mul_q7q3_int32(const q7_t *src1, const q7_t *src2, q31_t *dst,
		uint32_t row, uint32_t col, uint32_t col2, uint32_t shift)
{
	return nnblas_mat_mul((void *)src1, (void *)src2, (void *)dst, row, col, col2, shift, (uint32_t *)luna_matrix_mul_b4bit2w);
}

int32_t nnblas_mat_mul_q31q15_int32(const q31_t *src1, const q15_t *src2, q31_t *dst,
		uint32_t row, uint32_t col, uint32_t col2, uint32_t shift)
{
	return nnblas_mat_mul((void *)src1, (void *)src2, (void *)dst, row, col, col2, shift, (uint32_t *)luna_matrix_mul_wh2w);
}
*/

// int32_t nnblas_mat_trans_com(const void *src, void *dst, uint32_t row, uint32_t col, uint32_t bits)
// {
// 	int32_t ret = 0;

// 	matrix_params.src1 = LUNA_SHARE_ADDR_OFFSET(src);
// 	matrix_params.src2 = 0;
// 	matrix_params.dst = LUNA_SHARE_ADDR_OFFSET(dst);
// 	matrix_params.row = row;
// 	matrix_params.col = col;
// 	matrix_params.col2 = 0;
// 	matrix_params.shift = 0;
// 	matrix_params.data_type = bits;

// 	ret = luna_execute_cmd(luna_matrix_transpose, (void *)&matrix_params, sizeof(LunaMatrixParams_t));

// 	return ret;
// }

int32_t nnblas_mat_trans_q7(const q7_t *src1, q7_t *dst, uint32_t row, uint32_t col);

int32_t nnblas_mat_trans_q15(const q15_t *src1, q15_t *dst, uint32_t row, uint32_t col);

int32_t nnblas_mat_trans_q31(const q31_t *src1, q31_t *dst, uint32_t row, uint32_t col);

#if !USE_GROUP_MAT_INV

int32_t nnblas_group_mat_mul_q7_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_group_mat_mul_q7_int16(const q7_t *src1, const q7_t *src2, q15_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_group_mat_mul_q7_int32(const q7_t *src1, const q7_t *src2, q31_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_group_mat_mul_q15_int32(const q15_t *src1, const q15_t *src2, q31_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_group_mat_mul(const void *src1, const void *src2, void *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift, uint32_t in_bit, uint32_t out_bit);

#endif 

int32_t nnblas_mat_trans_inv_q7(const q7_t *src, q7_t *dst, uint32_t row, uint32_t col, uint32_t i_inv, uint32_t o_inv);

int32_t nnblas_mat_trans_inv_q15(const q15_t *src, q15_t *dst, uint32_t row, uint32_t col, uint32_t i_inv, uint32_t o_inv);

int32_t nnblas_mat_trans_inv_q31(const q31_t *src, q31_t *dst, uint32_t row, uint32_t col, uint32_t i_inv, uint32_t o_inv);

/*
int32_t nnblas_mat_trans_col234_inv_q31(const q31_t *src, q31_t *dst, uint32_t row, uint32_t col, uint32_t i_inv, uint32_t o_inv)
{
	int32_t ret = 0;
	uint32_t dtype = 0;
	void* api;

	if (col != 2 && col != 3 && col != 4)
	{
		LUNA_LOG("col must be 2/3/4\r\n");
		return -1;
	}

	matrixinv_params.src1 = LUNA_SHARE_ADDR_OFFSET(src);
	matrixinv_params.src2 = 0;
	matrixinv_params.dst = LUNA_SHARE_ADDR_OFFSET(dst);
	matrixinv_params.row = row;
	matrixinv_params.col = col;
	matrixinv_params.col2 = 0;
	matrixinv_params.shift = 0;
	matrixinv_params.i_inv1 = i_inv*(32>>3);
	matrixinv_params.i_inv2 = 0;
	matrixinv_params.o_inv = o_inv*(32>>3);

	ret = luna_execute_cmd(luna_matrix_transpose_col234_inv_w2w, (void *)&matrixinv_params, sizeof(matrixinv_params));

	return ret;
}*/

int32_t nnblas_mat_mul_inv_q7_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);

int32_t nnblas_mat_mul_inv_q7_int16(const q7_t *src1, const q7_t *src2, q15_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);

int32_t nnblas_mat_mul_inv_q7_int32(const q7_t *src1, const q7_t *src2, q31_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);

int32_t nnblas_mat_mul_inv_q15_int8(const q15_t *src1, const q15_t *src2, q7_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);

int32_t nnblas_mat_mul_inv_q15_int16(const q15_t *src1, const q15_t *src2, q15_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);

int32_t nnblas_mat_mul_inv_q15_int32(const q15_t *src1, const q15_t *src2, q31_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);

int32_t nnblas_mat_mul_inv_q31_int8(const q31_t *src1, const q31_t *src2, q7_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);

int32_t nnblas_mat_mul_inv_q31_int16(const q31_t *src1, const q31_t *src2, q15_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);

int32_t nnblas_mat_mul_inv_q31_int32(const q31_t *src1, const q31_t *src2, q31_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift);

// //col2 % 2 == 0 && i_inv2 % 2 == 0
// int32_t nnblas_mat_mul_inv_q7q3_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift)
// {
// 	return nnblas_mat_mul_inv_com(src1, src2, dst, row, col, col2, i_inv1, i_inv2, o_inv, shift, 4, 8);
// }

// //col2 % 2 == 0 && i_inv2 % 2 == 0
// int32_t nnblas_mat_mul_inv_q7q3_int16(const q7_t *src1, const q7_t *src2, q15_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift)
// {
// 	return nnblas_mat_mul_inv_com(src1, src2, dst, row, col, col2, i_inv1, i_inv2, o_inv, shift, 4, 16);
// }

// //col2 % 2 == 0 && i_inv2 % 2 == 0
// int32_t nnblas_mat_mul_inv_q7q3_int32(const q7_t *src1, const q7_t *src2, q31_t *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t i_inv1, uint32_t i_inv2, uint32_t o_inv, uint32_t shift)
// {
// 	return nnblas_mat_mul_inv_com(src1, src2, dst, row, col, col2, i_inv1, i_inv2, o_inv, shift, 4, 32);
// }

#if USE_GROUP_MAT_INV

int32_t nnblas_group_mat_mul_q7_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_group_mat_mul_q7_int16(const q7_t *src1, const q7_t *src2, q15_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_group_mat_mul_q7_int32(const q7_t *src1, const q7_t *src2, q31_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_group_mat_mul_q15_int32(const q15_t *src1, const q15_t *src2, q31_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

// int32_t nnblas_group_mat_mul(const void *src1, const void *src2, void *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift, uint32_t in_bit, uint32_t out_bit)
// {
// 	if ((in_bit == 8) && (out_bit == 8))
// 	{
// 		nnblas_group_mat_mul_q7_int8((const q7_t *)src1, (const q7_t *)src2, (q7_t *)dst, group_num, row, col, col2, shift);
// 	}
// 	else if ((in_bit == 16) && (out_bit == 32))
// 	{
// 		nnblas_group_mat_mul_q15_int32((const q15_t *)src1, (const q15_t *)src2, (q31_t *)dst, group_num, row, col, col2, shift);
// 	}
// 	else if ((in_bit == 8) && (out_bit == 32))
// 	{
// 		nnblas_group_mat_mul_q7_int32((const q7_t *)src1, (const q7_t *)src2, (q31_t *)dst, group_num, row, col, col2, shift);
// 	}
// 	else
// 	{
// 		printf("nnblas_group_mat_mul >> type of input or output error\n\r");
// 		return -1;
// 	}
// 	return 0;
// }


// int32_t nnblas_group_mat_mul_com(const void *src1, const void *src2, void *dst, uint32_t row, uint32_t col, uint32_t col2, uint32_t group, uint32_t shift, uint32_t i_precision, uint32_t o_precision)
// {
// 	int32_t ret = 0;
// 	uint32_t dtype = 0;
// 	uint32_t i_inv1,i_inv2,o_inv;
// 	uint32_t *api;

// 	i_inv1 = col * group;
// 	i_inv2 = col2;
// 	o_inv = col2 * group;

// 	matrixinv_params.src1 = LUNA_SHARE_ADDR_OFFSET(src1);
// 	matrixinv_params.src2 = LUNA_SHARE_ADDR_OFFSET(src2);
// 	matrixinv_params.dst = LUNA_SHARE_ADDR_OFFSET(dst);
// 	matrixinv_params.row = row;
// 	matrixinv_params.col = col;
// 	matrixinv_params.col2 = col2;
// 	matrixinv_params.shift = shift;
// 	matrixinv_params.i_inv1 = (4 == i_precision) ? i_inv1 : (i_inv1*(i_precision>>3));
// 	matrixinv_params.i_inv2 = (4 == i_precision) ? (i_inv2 >> 1) : (i_inv2*(i_precision>>3));
// 	matrixinv_params.o_inv = ((group<<16)|(o_inv*(o_precision>>3)));

// 	dtype = ((i_precision>>3)<<4)|(o_precision>>3);
// 	switch (dtype)
// 	{
// 		case 0x01: api = (uint32_t *)luna_matmul_group_b4bit2b;	break;
// 		case 0x02: api = (uint32_t *)luna_matmul_group_b4bit2h;	break;
// 		case 0x04: api = (uint32_t *)luna_matmul_group_b4bit2w;	break;
// 		case 0x11: api = (uint32_t *)luna_matrix_mul_group_b2b; break;
// 		case 0x12: api = (uint32_t *)luna_matrix_mul_group_b2h; break;
// 		case 0x14: api = (uint32_t *)luna_matrix_mul_group_b2w; break;
// 		case 0x21: api = (uint32_t *)luna_matrix_mul_group_h2b; break;
// 		case 0x22: api = (uint32_t *)luna_matrix_mul_group_h2h; break;
// 		case 0x24: api = (uint32_t *)luna_matrix_mul_group_h2w; break;
// 		case 0x41: api = (uint32_t *)luna_matrix_mul_group_w2b; break;
// 		case 0x42: api = (uint32_t *)luna_matrix_mul_group_w2h; break;
// 		case 0x44: api = (uint32_t *)luna_matrix_mul_group_w2w; break;
// 		default: 
// 		{
// 			LUNA_LOG("nnblas_mat_mul_group_com unkonw dtype, dtype = %d\r\n", dtype);
// 			return -1;
// 		}
// 		break;
// 	}
	
// 	ret = luna_execute_cmd(api, (void *)&matrixinv_params, sizeof(matrixinv_params));

// 	return ret;
// }
/*
int32_t nnblas_group_mat_mul(const void *src1, const void *src2, void *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift, uint32_t in_bit, uint32_t out_bit)
{
	return nnblas_group_mat_mul_com(src1, src2, dst, row, col, col2, group_num, shift, in_bit, out_bit);
}

int32_t nnblas_group_mat_mul_q7q3_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift)
{
	return nnblas_group_mat_mul_com(src1, src2, dst, row, col, col2, group_num, shift, 4, 8);
}

int32_t nnblas_group_mat_mul_q7q3_int16(const q7_t *src1, const q7_t *src2, q15_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift)
{
	return nnblas_group_mat_mul_com(src1, src2, dst, row, col, col2, group_num, shift, 4, 16);
}

int32_t nnblas_group_mat_mul_q7q3_int32(const q7_t *src1, const q7_t *src2, q31_t *dst, uint32_t group_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift)
{
	return nnblas_group_mat_mul_com(src1, src2, dst, row, col, col2, group_num, shift, 4, 32);
}
*/
#endif

int32_t nnblas_split_mat_trans_q7(const q7_t *src, q7_t *dst, uint32_t row, uint32_t col, uint32_t split_num);

int32_t nnblas_split_mat_trans_q15(const q15_t *src, q15_t *dst, uint32_t row, uint32_t col, uint32_t split_num);

int32_t nnblas_split_mat_trans_q31(const q31_t *src, q31_t *dst, uint32_t row, uint32_t col, uint32_t split_num);

int32_t nnblas_trans_axis_q7(const q7_t *src, q7_t *dst, uint32_t *in_shape, uint32_t *axis, uint32_t n_dims);

int32_t nnblas_trans_axis_q15(const q15_t *src, q15_t *dst, uint32_t *in_shape, uint32_t *axis, uint32_t n_dims);

int32_t nnblas_trans_axis_q31(const q31_t *src, q31_t *dst, uint32_t *in_shape, uint32_t *axis, uint32_t n_dims);

int32_t nnblas_split_mat_mul_q7_int8(const q7_t *src1, const q7_t *src2, q7_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_split_mat_mul_q7_int16(const q7_t *src1, const q7_t *src2, q15_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_split_mat_mul_q7_int32(const q7_t *src1, const q7_t *src2, q31_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_split_mat_mul_q15_int8(const q15_t *src1, const q15_t *src2, q7_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_split_mat_mul_q15_int16(const q15_t *src1, const q15_t *src2, q15_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_split_mat_mul_q15_int32(const q15_t *src1, const q15_t *src2, q31_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_split_mat_mul_q31_int8(const q31_t *src1, const q31_t *src2, q7_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_split_mat_mul_q31_int16(const q31_t *src1, const q31_t *src2, q15_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);

int32_t nnblas_split_mat_mul_q31_int32(const q31_t *src1, const q31_t *src2, q31_t *dst, uint32_t split_num, uint32_t row, uint32_t col, uint32_t col2, uint32_t shift);
#endif

