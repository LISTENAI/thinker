#pragma once
#include <stdint.h>

/**
 * Calculate required memory size for FFN operation (trans decoder version).
 * @param dim_in: Input dimension
 * @param dim_hidden: Hidden dimension
 * @param dim_out: Output dimension
 * @param seq_len: Sequence length
 * @param group_num: Group number
 * @return: Required memory size in bytes
 */
uint32_t nlang_ffn_int_trans_dec_getsize(uint32_t dim_in, uint32_t dim_hidden, uint32_t dim_out, uint32_t seq_len, uint32_t group_num);

/**
 * Perform FFN computation (trans decoder version).
 * @param p_input: Input tensor (1, D)
 * @param p_weight_m0: First weight matrix (8, Dh/8, Di)
 * @param p_bias_m0: First bias vector (8, Dh/8, Di)
 * @param p_weight_m1: Second weight matrix (8, D0, Dh/8)
 * @param p_bias_m1: Second bias vector (8, D0, Dh/8)
 * @param p_weight_mask: Mask weight matrix (8, D)
 * @param p_bias_mask: Mask bias vector (8, D)
 * @param p_output: Output tensor
 * @param p_temp: Temporary buffer
 * @param dim_in: Input dimension
 * @param dim_hidden: Hidden dimension
 * @param dim_out: Output dimension
 * @param seq_len: Sequence length
 * @param group_num: Group number
 * @param q_input_m0: First input quantization level
 * @param q_weight_m0: First weight quantization level
 * @param q_output_m0: First output quantization level
 * @param q_input_m1: Second input quantization level
 * @param q_weight_m1: Second weight quantization level
 * @param q_output_m1: Second output quantization level
 * @param q_input_mask: Mask input quantization level
 * @param q_weight_mask: Mask weight quantization level
 * @param q_output_mask: Mask output quantization level
 * @return: Status code
 */
int32_t nlang_ffn_int_trans_dec(int8_t *p_input,
    int8_t *p_weight_m0, int32_t *p_bias_m0,
    int8_t *p_weight_m1, int32_t *p_bias_m1,
    int8_t *p_weight_mask, int32_t *p_bias_mask,
    int8_t *p_output, int8_t *p_temp,
    uint32_t dim_in, uint32_t dim_hidden, uint32_t dim_out, uint32_t seq_len, uint32_t group_num,
    int32_t q_input_m0, int32_t q_weight_m0, int32_t q_output_m0,
    int32_t q_input_m1, int32_t q_weight_m1, int32_t q_output_m1,
    int32_t q_input_mask, int32_t q_weight_mask, int32_t q_output_mask);