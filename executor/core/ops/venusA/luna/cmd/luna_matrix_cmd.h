#ifndef __LUNA_MATRIX_H__
#define __LUNA_MATRIX_H__

#include "luna/luna.h"

typedef struct LunaMatrixTransposeSettings
{
	uint32_t i_addr;
	uint32_t o_addr;
	uint32_t precision;
	uint32_t row;
	uint32_t col;
	uint32_t i_inv;
	uint32_t o_inv;

	uint32_t loop_num;
	uint32_t i_addr_inc;
	uint32_t o_addr_inc;
	uint32_t loop_num2;
	uint32_t i_addr_inc2;
	uint32_t o_addr_inc2;

	uint32_t row_last;
	uint32_t col_last;
}LunaMatrixTransposeSettings_t;

typedef struct LunaMatrixCopySettings
{
	uint32_t i_addr; 
	uint32_t o_addr; 
	uint32_t precision;  

	uint32_t h;
	uint32_t w;
	uint32_t c;

	uint32_t loop_num;
	uint32_t i_inv;
	uint32_t o_inv; 
	uint32_t i_addr_inc; 
	uint32_t o_addr_inc;
}LunaMatrixCopySettings_t;

typedef struct LunaMatrixSettings
{
	uint32_t i_addr_m0;
	uint32_t i_addr_m1;
	uint32_t i_addr_bias;  
	uint32_t o_addr;
	uint32_t M;
	uint32_t N;   
	uint32_t L;
	uint32_t i_interval0; 
	uint32_t i_interval1;
	uint32_t o_interval; 

	uint32_t i_precision;
	uint32_t o_precision;
	uint32_t m0_in_flash;
	uint32_t m0_bit4_en;
	uint32_t m1_in_flash;
	uint32_t m1_bit4_en;
	uint32_t bias_en;
	uint32_t bias_in_flash;

	uint32_t loop_num_0;
	uint32_t loop_num_1; 
	uint32_t i_addr_m0_inc;
	uint32_t i_addr_m1_inc;
	uint32_t o_addr_inc_0;
	uint32_t o_addr_inc_1; 
	uint32_t i_addr_bias_inc; 	
	uint32_t i_addr_bias_inc2; 
	
	uint32_t LL_dstm0_slave0_r18;
	uint32_t LL_dstm0_slave0_r19;
	uint32_t LL_ares_burst_r8;
	uint32_t LL_ares_burst_r9;
	uint32_t LL_ares_burst_r13;
	uint32_t LL_ares_burst_r12;

	uint32_t LR_dstm0_slave0_r18;
	uint32_t LR_dstm0_slave0_r19;
	uint32_t LR_ares_burst_r8;
	uint32_t LR_ares_burst_r9;
	uint32_t LR_ares_burst_r13;
	uint32_t LR_ares_burst_r12;

	uint32_t CC_master0_select0_row0_r8;
	uint32_t CC_master0_select0_row0_r9;
	uint32_t CC_master0_select0_row1_r8;
	uint32_t CC_master0_select0_row1_r9;
	uint32_t CC_master0_select0_row2_r8;
	uint32_t CC_master0_select2_0_r12;
	uint32_t CC_master0_select2_1_r10;
	uint32_t CC_master0_select2_2_r11;
	uint32_t CC_master0_select2_2_r12;
	uint32_t CC_master0_select2_2_r13;
	uint32_t CC_master0_select2_2_r14;
	uint32_t CC_master0_select2_2_r17;
	uint32_t CC_master0_select3_0_r15;
	uint32_t CC_master0_select3_0_r16;
	uint32_t CC_master0_select3_1_r15;
	uint32_t CC_master0_select3_1_r16;
	uint32_t CC_dprc_r23;
	uint32_t CC_dprc_r26;
	uint32_t CC_dprc_r27;
	uint32_t CC_iowr_r20;
	uint32_t CC_iowr_r18;
	uint32_t CC_iowr_r19;
	uint32_t CC_iowr_r21;

	uint32_t CC_iowr_r19_H;
	// M_last
	uint32_t LL_dstm0_slave0_r19_M_last;
	uint32_t LL_ares_burst_r13_M_last;
	uint32_t CC_master0_select0_row2_r8_M_last_o_cnt;
	uint32_t CC_iowr_r19_L_M_last;
	// L_last 
	uint32_t LR_dstm0_slave0_r18_L_last;
	uint32_t LR_ares_burst_r12_L_last;
	uint32_t CC_master0_select0_row1_r9_L_last_m_cnt;
	uint32_t CC_iowr_r18_L_last;
	uint32_t CC_iowr_r19_H_last_iowr_last_vld_bp;

} LunaMatrixSettings_t;

extern __luna_cmd_attr__ uint32_t luna_matrix_mul_cmd[];
extern __luna_cmd_attr__ uint32_t luna_matrix_transpose_cmd[];
extern __luna_cmd_attr__ uint32_t luna_matrix_transpose_3d_wch_cmd[];

#endif

