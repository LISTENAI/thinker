#pragma once
#include <stdint.h>

/**
 * Calculate required memory size for self-attention operation (pin-du version).
 * @param dim_in: Input dimension
 * @param dim_out: Output dimension  
 * @param headers: Number of attention heads
 * @param dim_head: Dimension per head
 * @param n: Sequence length
 * @return: Required memory size in bytes
 */
uint32_t nlang_self_attention_int_pindu_getsize(uint32_t dim_in, uint32_t dim_out, uint32_t headers, uint32_t dim_head, uint32_t n);

/**
 * Perform self-attention computation (pin-du version).
 * @param p_input: Input tensor (n, c)
 * @param p_weight_q: Query weight matrix (headers*dim_head, c)
 * @param p_bias_q: Query bias vector (headers*dim_head)
 * @param p_weight_k: Key weight matrix (headers*dim_head, c)
 * @param p_bias_k: Key bias vector (headers*dim_head)
 * @param p_weight_v: Value weight matrix (headers*dim_head, c)
 * @param p_bias_v: Value bias vector (headers*dim_head)
 * @param p_weight_out: Output weight matrix (dim_out, dim_head)
 * @param p_bias_out: Output bias vector (dim_out)
 * @param p_output: Output tensor
 * @param p_temp: Temporary buffer
 * @param dim_in: Input dimension
 * @param dim_out: Output dimension
 * @param headers: Number of attention heads
 * @param dim_head: Dimension per head
 * @param n: Sequence length
 * @param scale: Scaling factor (Q15 format)
 * @param q_input: Input quantization level
 * @param q_weight_q: Query weight quantization level
 * @param q_weight_k: Key weight quantization level
 * @param q_weight_v: Value weight quantization level
 * @param q_output_q: Query output quantization level
 * @param q_output_k: Key output quantization level
 * @param q_output_v: Value output quantization level
 * @param q_output_bmm0: BMM0 output quantization level
 * @param q_weight_scale: Weight scaling quantization level
 * @param q_output_scale: Output scaling quantization level
 * @param q_output_softmax: Softmax output quantization level
 * @param q_output_bmm1: BMM1 output quantization level
 * @param q_weight_o: Output weight quantization level
 * @param q_output: Final output quantization level
 * @return: Status code
 */
int32_t nlang_self_attention_int_pindu(int8_t *p_input, 
    int8_t *p_weight_q, int32_t *p_bias_q,
    int8_t *p_weight_k, int32_t *p_bias_k,
    int8_t *p_weight_v, int32_t *p_bias_v,
    int8_t *p_weight_out, int32_t *p_bias_out,
    int8_t *p_output, int8_t *p_temp,
    uint32_t dim_in, uint32_t dim_out, uint32_t headers, uint32_t dim_head, uint32_t n,
    int32_t scale, 
    int32_t q_input, int32_t q_weight_q, int32_t q_weight_k, int32_t q_weight_v,
    int32_t q_output_q, int32_t q_output_k, int32_t q_output_v,
    int32_t q_output_bmm0,
    int32_t q_weight_scale, int32_t q_output_scale,
    int32_t q_output_softmax,
    int32_t q_output_bmm1,
    int32_t q_weight_o, int32_t q_output);

/**
 * Calculate required memory size for self-attention operation (trans version).
 * @param dim_in: Input dimension
 * @param dim_out: Output dimension
 * @param headers: Number of attention heads
 * @param dim_head: Dimension per head
 * @param n: Sequence length
 * @return: Required memory size in bytes
 */
uint32_t nlang_self_attention_int_trans_getsize(uint32_t dim_in, uint32_t dim_out, uint32_t headers, uint32_t dim_head, uint32_t n);

/**
 * Perform self-attention computation (trans version).
 * @param p_input: Input tensor (n, c)
 * @param p_weight_q: Query weight matrix (headers*dim_head, c)
 * @param p_bias_q: Query bias vector (headers*dim_head)
 * @param p_weight_k: Key weight matrix (headers*dim_head, c)
 * @param p_bias_k: Key bias vector (headers*dim_head)
 * @param p_weight_v: Value weight matrix (headers*dim_head, c)
 * @param p_bias_v: Value bias vector (headers*dim_head)
 * @param p_weight_out: Output weight matrix (dim_out, dim_head)
 * @param p_bias_out: Output bias vector (dim_out)
 * @param p_output: Output tensor
 * @param p_temp: Temporary buffer
 * @param dim_in: Input dimension
 * @param dim_out: Output dimension
 * @param headers: Number of attention heads
 * @param dim_head: Dimension per head
 * @param n: Sequence length
 * @param scale: Scaling factor (Q15 format)
 * @param q_input: Input quantization level
 * @param q_weight_q: Query weight quantization level
 * @param q_weight_k: Key weight quantization level
 * @param q_weight_v: Value weight quantization level
 * @param q_output_q: Query output quantization level
 * @param q_output_k: Key output quantization level
 * @param q_output_v: Value output quantization level
 * @param q_output_bmm0: BMM0 output quantization level
 * @param q_weight_scale: Weight scaling quantization level
 * @param q_output_scale: Output scaling quantization level
 * @param q_output_softmax: Softmax output quantization level
 * @param q_output_bmm1: BMM1 output quantization level
 * @param q_weight_o: Output weight quantization level
 * @param q_output: Final output quantization level
 * @param p_weight_emb_k: Relative position embedding key weights (2*max_rel+1, dim_head)
 * @param p_weight_emb_v: Relative position embedding value weights (2*max_rel+1, dim_head)
 * @param q_x_bmm2: BMM2 x input quantization level
 * @param q_y_bmm2: BMM2 y input quantization level
 * @param q_o_bmm2: BMM2 output quantization level
 * @param q_x_bmm3: BMM3 x input quantization level
 * @param q_y_bmm3: BMM3 y input quantization level
 * @param q_o_bmm3: BMM3 output quantization level
 * @param q_x_add1: Add1 x input quantization level
 * @param q_y_add1: Add1 y input quantization level
 * @param q_o_add1: Add1 output quantization level
 * @param q_x_add2: Add2 x input quantization level
 * @param q_y_add2: Add2 y input quantization level
 * @param q_o_add2: Add2 output quantization level
 * @param max_rel: Maximum relative position
 * @return: Status code
 */
int32_t nlang_self_attention_int_trans(int8_t *p_input,
    int8_t *p_weight_q, int32_t *p_bias_q,
    int8_t *p_weight_k, int32_t *p_bias_k,
    int8_t *p_weight_v, int32_t *p_bias_v,
    int8_t *p_weight_out, int32_t *p_bias_out,
    int8_t *p_output, int8_t *p_temp,
    uint32_t dim_in, uint32_t dim_out, uint32_t headers, uint32_t dim_head, uint32_t n,
    int32_t scale, 
    int32_t q_input, int32_t q_weight_q, int32_t q_weight_k, int32_t q_weight_v,
    int32_t q_output_q, int32_t q_output_k, int32_t q_output_v,
    int32_t q_output_bmm0,
    int32_t q_weight_scale, int32_t q_output_scale,
    int32_t q_output_softmax,
    int32_t q_output_bmm1,
    int32_t q_weight_o, int32_t q_output,
    int8_t *p_weight_emb_k, int8_t *p_weight_emb_v,
    int32_t q_x_bmm2, int32_t q_y_bmm2, int32_t q_o_bmm2,
    int32_t q_x_bmm3, int32_t q_y_bmm3, int32_t q_o_bmm3,
    int32_t q_x_add1, int32_t q_y_add1, int32_t q_o_add1,
    int32_t q_x_add2, int32_t q_y_add2, int32_t q_o_add2,
    int32_t max_rel);