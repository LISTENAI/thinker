#pragma once
#include <stdint.h>

/**
 * Calculate required memory size for FFN operation (trans version).
 * @param dim_in: Input dimension
 * @param dim_hidden: Hidden dimension
 * @param dim_out: Output dimension
 * @param seq_len: Sequence length
 * @return: Required memory size in bytes
 */
int32_t nlang_ffn_int_trans_getsize(uint32_t dim_in, uint32_t dim_hidden, uint32_t dim_out, uint32_t seq_len);

/**
 * Perform FFN computation (trans version).
 * @param p_input: Input tensor (T, D)
 * @param p_weight_m0: First weight matrix (Dh, D)
 * @param p_bias_m0: First bias vector (Dh, D)
 * @param p_weight_m1: Second weight matrix (Do, Dh)
 * @param p_bias_m1: Second bias vector (Do, Dh)
 * @param p_output: Output tensor
 * @param p_temp: Temporary buffer
 * @param dim_in: Input dimension
 * @param dim_hidden: Hidden dimension
 * @param dim_out: Output dimension
 * @param seq_len: Sequence length
 * @param q_input_m0: First input quantization level
 * @param q_weight_m0: First weight quantization level
 * @param q_output_m0: First output quantization level
 * @param q_input_m1: Second input quantization level
 * @param q_weight_m1: Second weight quantization level
 * @param q_output_m1: Second output quantization level
 * @return: Status code
 */
int32_t nlang_ffn_int_trans(int8_t *p_input,
    int8_t *p_weight_m0, int32_t *p_bias_m0,
    int8_t *p_weight_m1, int32_t *p_bias_m1,
    int8_t *p_output, int8_t *p_temp,
    uint32_t dim_in, uint32_t dim_hidden, uint32_t dim_out, uint32_t seq_len,
    int32_t q_input_m0, int32_t q_weight_m0, int32_t q_output_m0,
    int32_t q_input_m1, int32_t q_weight_m1, int32_t q_output_m1);