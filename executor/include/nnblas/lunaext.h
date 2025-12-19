#ifndef __LUNA_EXT_H__
#define __LUNA_EXT_H__

#include <stdint.h>
#include <stdlib.h>

/**
 * Calculate required memory size for LSTM operation.
 * @param input_size: Input size
 * @param hidden_size: Hidden size
 * @param seq_size: Sequence size
 * @param go_forward: Direction flag (1 for forward, 0 for backward)
 * @param split_num: Split number
 * @return: Required memory size in bytes
 */
uint32_t nlang_lstm_int_getsize(int32_t input_size, int32_t hidden_size, int32_t seq_size, int32_t go_forward, int32_t split_num);

/**
 * Perform LSTM computation.
 * @param p_in: Input tensor
 * @param p_out: Output tensor
 * @param p_h_in: Hidden state input
 * @param p_cell_in: Cell state input
 * @param p_iw_weight: Input weight matrix
 * @param p_hw_weight: Hidden weight matrix
 * @param p_ib_bias: Input bias vector
 * @param p_hb_bias: Hidden bias vector
 * @param p_tmp: Temporary buffer
 * @param input_size: Input size
 * @param hidden_size: Hidden size
 * @param seq_size: Sequence size
 * @param go_forward: Direction flag (1 for forward, 0 for backward)
 * @param split_num: Split number
 * @param i_q: Input quantization level
 * @param h_q: Hidden quantization level
 * @param iw_q: Input weight quantization level
 * @param hw_q: Hidden weight quantization level
 * @return: Status code
 */
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

#endif /* __LUNA_EXT_H__ */