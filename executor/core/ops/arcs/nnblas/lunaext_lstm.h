#pragma once

#include <stdint.h>

uint32_t nlang_lstm_int_getsize(int32_t input_size, int32_t hidden_size, int32_t seq_size, int32_t go_forward, int32_t split_num);

int32_t nlang_lstm_int(
    int8_t *p_in,
    int8_t  *p_out,
    int8_t  *p_h_in,
    int32_t *p_cell_in,
    int8_t  *p_iw_weight,
    int8_t  *p_hw_weight,
    int32_t *p_ib_bias,
    int32_t *p_hb_bias,
    int8_t  *p_tmp, 
    int32_t input_size, int32_t hidden_size, int32_t seq_size, int32_t go_forward, int32_t split_num,
    int32_t i_q, int32_t h_q, int32_t iw_q, int32_t hw_q);

uint32_t nlang_lstm_seq_n_int_getsize(int32_t input_size, int32_t hidden_size, int32_t seq_size, int32_t go_forward, int32_t split_num);

int32_t nlang_lstm_seq_n_int(
    int8_t *p_in,
    int8_t  *p_out,
    int8_t  *p_h_in,
    int32_t *p_cell_in,
    int8_t  *p_iw_weight,
    int8_t  *p_hw_weight,
    int32_t *p_ib_bias,
    int32_t *p_hb_bias,
    int8_t  *p_tmp, 
    int32_t input_size, int32_t hidden_size, int32_t seq_size, int32_t go_forward, int32_t split_num,
    int32_t i_q, int32_t h_q, int32_t iw_q, int32_t hw_q);

uint32_t nlang_lstm_seq_1_int_getsize(int32_t input_size, int32_t hidden_size, int32_t seq_size, int32_t go_forward, int32_t split_num);

int32_t nlang_lstm_seq_1_int(
    int8_t *p_in,
    int8_t  *p_out,
    int8_t  *p_h_in,
    int32_t *p_cell_in,
    int8_t  *p_iw_weight,
    int8_t  *p_hw_weight,
    int32_t *p_ib_bias,
    int32_t *p_hb_bias,
    int8_t  *p_tmp, 
    int32_t input_size, int32_t hidden_size, int32_t seq_size, int32_t go_forward, int32_t split_num,
    int32_t i_q, int32_t h_q, int32_t iw_q, int32_t hw_q);