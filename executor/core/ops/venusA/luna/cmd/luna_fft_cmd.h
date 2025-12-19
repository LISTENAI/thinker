#ifndef __LUNA_API_FFT_H__
#define __LUNA_API_FFT_H__

#include "luna/luna.h"

typedef struct LunaFFTSettings
{
	uint32_t i_addr;  // offset 0
	uint32_t o_addr; // offset 4
	uint32_t fft_length; // offset 8
	uint32_t fft_log2; // offset 12

	uint32_t shift; 	// offset 16
	uint32_t mode0;   // mode >= 1, {'FFT','IFFT','CR256-FFT','CR256-IFFT','CR257-FFT','CR257-IFFT'};
	uint32_t mode1;   // mode >= 1, {'NORMAL','SHORTEN','REALONLY'};
	uint32_t mode2;	  // mode >= 1, {'SHIFT'};

	uint32_t complex_mode0; // offset 32

	// load input data stage 0
	uint32_t lid_stage0_r8_L; // offset 36
	uint32_t lid_stage0_r20_L; // offset 40
	uint32_t lid_stage0_r20_H; // offset 44
	uint32_t lid_stage0_r15_L; // offset 48

	// load input data stage 1
	uint32_t lid_stage1_r18_L; // offset 52
	
	// load input data stage 2
	uint32_t lid_stage2_r8_L; // offset 56
	uint32_t lid_stage2_r20_L; // offset 60
	uint32_t lid_stage2_r20_H; // offset 64
	uint32_t lid_stage2_r15_L; // offset 68

	// load input data stage 3
	uint32_t lid_stage3_r18_L; // offset 72
	uint32_t lid_stage3_r19_H; // offset 76
	uint32_t lid_stage3_r18; // offset 80
	uint32_t lid_stage3_r9; // offset 84  no use

	// slave common configuration
	uint32_t ares_mas0_sel3_r18_L; // offset 88

	// last stage process
	uint32_t bl; // offset 92
	uint32_t bh; // offset 96
	uint32_t ares_mas0_sel0_r20_L; // offset 100
	uint32_t ares_mas0_sel0_r20_H; // offset 104
	uint32_t ares_mas0_sel2_r20_H; // // offset 108
	
	// slave config
	uint32_t rst_reg1_reg2_reg3_r18_L; // offset 112
	uint32_t rst_reg1_reg2_reg3_r20_L; // offset 116
	uint32_t rst_reg1_reg2_reg3_r20_H; // offset 120

	// output
	uint32_t mnts_mas0_mem_iowr_r8_L; // offset 124
	uint32_t mnts_mas0_mem_iowr_r9_L; // offset 128
	uint32_t ares_mas0_sel0_chs0000_r10_L; // offset 132
	uint32_t ares_mas0_sel0_chs0000_r20_L; // offset 136
	uint32_t ares_mas0_sel0_chs0000_r20_H; // offset 140
	uint32_t ares_mas0_sel0_chs0000_r15_L; // offset 144

	
	// iowr config
	uint32_t iowr_cfg_r20_L; // offset 148
	uint32_t iowr_cfg_r20_H; // offset 152
	uint32_t iowr_cfg_r18_L; // offset 156
	uint32_t iowr_cfg_outband_fix; // offset 160
	
	uint32_t mem_we0_addr_flag;  // offset 164
}LunaFFTSettings_t;

extern __luna_cmd_attr__ uint32_t luna_fft_cmd[];

#endif

