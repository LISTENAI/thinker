#ifndef __LUNA_API_MISC_H__
#define __LUNA_API_MISC_H__

#include "luna/luna.h"

typedef struct LunaMiscActvParams
{
	uint32_t src;
	uint32_t dst;
	uint32_t size;
	uint32_t pos_shift;
	uint32_t neg_shift;
	uint32_t inout_bits;
	uint32_t act_type;
}LunaMiscActvParams_t;

typedef struct LunaMiscMemParams
{
	uint32_t src;
	uint32_t dst;
	uint32_t size;
	uint32_t value;
	uint32_t inbits;
}LunaMiscMemParams_t;

typedef struct LunaExpParams
{
	uint32_t src;
	uint32_t dst;
	uint32_t size;
}LunaExpParams_t;

typedef struct LunaSoftmaxParams
{
	uint32_t i_addr; //0
	uint32_t o_addr;
	uint32_t length;
	uint32_t in_num;

	uint32_t in_num_offs; //16
	uint32_t in_mask;
	uint32_t r8_value_low;
	uint32_t r9_value_low;

	uint32_t ou_num; //32

	uint32_t maxmin_pe_r26;  //36 + 0
	uint32_t maxmin_pe_27;
	uint32_t sum_master_s3_r15_L;
	uint32_t sum_pe_r27;

	uint32_t scale_master_s3_r15_L; //16  //52
	uint32_t scale_pe_r23_H;  //56
	uint32_t src_sel; //60
}LunaSoftmaxParams_t;

typedef struct LunaLutNewParams 
{
	uint32_t i_precision_dat; //0  log2(x)<<16|(x)
	uint32_t i_precision_idx; //4 log2(x)<<16|(x)
	uint32_t i_copy_band; //8 log2(x)<<16|(x)
	uint32_t cfg_size_dat; //12

	uint32_t cfg_size_idx; //16
	uint32_t i_addr_dat; //20
	uint32_t i_addr_idx; //24
	uint32_t o_addr; //28
}LunaLutNewParams_t;

typedef struct LunaLutActParams
{
	uint32_t i_addr;
	uint32_t o_addr;
	uint32_t cfg_size;
	uint32_t act_type;// 0:sigmoid  1:tanh  2:swish  3:gelu 

	uint32_t src_sel; 
	uint32_t i_precision;
	uint32_t o_precision;
	uint32_t cfg_shift;
}LunaLutActParams_t;

typedef struct LunaGpdmaParams
{
	uint32_t chn;
	uint32_t i_addr;
	uint32_t o_addr;
	uint32_t size;
	uint32_t sg_num;
	uint32_t sg_itv;
	uint32_t ds_num;
	uint32_t ds_itv;
}LunaGpdmaParams_t;

typedef struct LunaRGBSub128Params
{
	uint32_t i_addr;
	uint32_t o_addr;
	uint32_t cfg_size;
	uint32_t ahb_en;
}LunaRGBSub128Params_t;

extern __luna_cmd_attr__ uint32_t luna_api_memset[];
extern __luna_cmd_attr__ uint32_t luna_api_memcpy[];
extern __luna_cmd_attr__ uint32_t luna_api_activate[];
extern __luna_cmd_attr__ uint32_t luna_api_lut[];
extern __luna_cmd_attr__ uint32_t luna_api_exp[];
extern __luna_cmd_attr__ uint32_t luna_api_activate_relux[];
extern __luna_cmd_attr__ uint32_t luna_api_softmax[];
extern __luna_cmd_attr__ uint32_t luna_api_psrammemcpy[];
extern __luna_cmd_attr__ uint32_t luna_api_lut_new[];
extern __luna_cmd_attr__ uint32_t luna_api_lut_act[];
extern __luna_cmd_attr__ uint32_t luna_api_gpdma_start[];
extern __luna_cmd_attr__ uint32_t luna_api_gpdma_wait[];
extern __luna_cmd_attr__ uint32_t luna_api_crc32[];
extern __luna_cmd_attr__ uint32_t luna_api_crc32[];
extern __luna_cmd_attr__ uint32_t luna_api_rgb_sub128[];

#endif

