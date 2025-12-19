#pragma once

#include <stdint.h>

int32_t nlang_ffn_int_trans_getsize(uint32_t dim_in, uint32_t dim_hidden, uint32_t dim_out, uint32_t seq_len);

int32_t nlang_ffn_int_trans(int8_t *p_input, // (T, D) 
    int8_t *p_weight_m0, int32_t *p_bias_m0, // (Dh, D) weight@psram + bias@share
    int8_t *p_weight_m1, int32_t *p_bias_m1, // (Do, Dh) weight@psram + bias@share
    int8_t *p_output, int8_t *p_temp,
    uint32_t dim_in, uint32_t dim_hidden, uint32_t dim_out, uint32_t seq_len,
    int32_t q_input_m0, int32_t q_weight_m0, int32_t q_output_m0, 
    int32_t q_input_m1, int32_t q_weight_m1, int32_t q_output_m1);