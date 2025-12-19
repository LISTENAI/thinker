#pragma once

#include <stdint.h>

uint32_t nlang_ffn_int_trans_dec_getsize(uint32_t dim_in, uint32_t dim_hidden, uint32_t dim_out, uint32_t seq_len, uint32_t group_num);

int32_t nlang_ffn_int_trans_dec(int8_t *p_input,  // (1,D)
    int8_t *p_weight_m0, int32_t *p_bias_m0, // (8, Dh/8, Di) weight@psram + bias@psram
    int8_t *p_weight_m1, int32_t *p_bias_m1, // (8, D0, Dh/8) weight@psram + bias@psram
    int8_t *p_weight_mask, int32_t *p_bias_mask, //(8, D) weight@psram + bias@psram
    int8_t *p_output, int8_t *p_temp,
    uint32_t dim_in, uint32_t dim_hidden, uint32_t dim_out, uint32_t seq_len, uint32_t group_num,
    int32_t q_input_m0, int32_t q_weight_m0, int32_t q_output_m0, 
    int32_t q_input_m1, int32_t q_weight_m1, int32_t q_output_m1,
    int32_t q_input_mask, int32_t q_weight_mask, int32_t q_output_mask,
    int8_t is_4bit_m0, int8_t is_4bit_m1);