#ifndef _MULTIHEADATTENTION_LUNA_H_
#define _MULTIHEADATTENTION_LUNA_H_

#include <math.h>
#include <stdlib.h>

#include "core/operator_attrs.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "thinker_status.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief Add two int8 tensors with scaling
 * @param p_input1 First input tensor
 * @param p_input2 Second input tensor
 * @param p_output Output tensor
 * @param p_temp Temporary buffer
 * @param size Tensor size
 * @param scale_x Scale factor for first input
 * @param scale_y Scale factor for second input
 * @param scale_o Output scale factor
 * @return Operation status
 */
static int32_t luna_add_int8(int8_t* p_input1, int8_t* p_input2, int8_t* p_output, int8_t* p_temp,
    int32_t size, int32_t scale_x, int32_t scale_y, int32_t scale_o) 
{
    int ret = 0;
    
    // Scale inputs to common format
    if (scale_x > scale_o) {
        ret |= luna_scale_i8i8o8(p_input1, 1, p_input1, size, scale_x - scale_o);
    } else if (scale_x < scale_o) {
        ret |= luna_scale_i8i8o8(p_input1, 1<<(scale_o - scale_x), p_input1, size, 0);
    }
    if (scale_y > scale_o) {
        ret |= luna_scale_i8i8o8(p_input2, 1, p_input2, size, scale_y - scale_o);
    } else if (scale_y < scale_o) {
        ret |= luna_scale_i8i8o8(p_input2, 1<<(scale_o - scale_y), p_input2, size, 0);
    }
    
    // Perform addition
    ret = luna_add_i8i8o8(p_input1, p_input2, p_output, size, 0);
    return ret;
}

/**
 * @brief Apply softmax to int8 tensors
 * @param p_input Input tensor
 * @param p_output Output tensor
 * @param p_temp Temporary buffer
 * @param batch Batch size
 * @param size Element count per batch
 * @param q_x Input scale
 * @param q_o Output scale
 * @return Operation status
 */
static int32_t luna_softmax_int8(int8_t* p_input, int8_t* p_output, int8_t* p_temp, int32_t batch, int32_t size, int32_t q_x, int32_t q_o)
{
    int ret = 0;
    int32_t *p_softmax = (int32_t *)p_temp; p_temp += 2*size*sizeof(int32_t);
    int32_t *p_softmax2 = (int32_t *)p_temp; p_temp += 2*size*sizeof(int32_t); 
    
    for (int32_t j = 0; j < batch; j++) {
        ret |= luna_scale_i8i8o32(p_input + j*size, 1, p_softmax, size, 0);
        ret |= luna_scale_i32i32o32(p_softmax, 1<<(25-q_x), p_softmax, size, 0);
        ret |= luna_softmax_i32o32(p_softmax, p_softmax2, size);  //6.25=>16.15
        ret |= luna_scale_i32i32o8(p_softmax2, 1, p_output + j*size, size, 15 - q_o);
    }
    return ret;
}

/**
 * @brief Batch matrix multiplication for int8 tensors
 * @param p_mat1 First matrix
 * @param p_mat2 Second matrix
 * @param p_out Output matrix
 * @param B Batch size
 * @param M First matrix rows
 * @param K Inner dimension
 * @param N Second matrix columns
 * @param q_x First matrix scale
 * @param q_y Second matrix scale
 * @param q_o Output scale
 * @return Operation status
 */
static int32_t luna_bmm_int8(int8_t* p_mat1, int8_t* p_mat2, int8_t* p_out, 
    int32_t B, int32_t M, int32_t K, int32_t N, 
    int32_t q_x, int32_t q_y, int32_t q_o)
{
    int ret = 0;
    for (int32_t i = 0; i < B; i++) {
        ret |= luna_mat_mul_i8i8o8(p_mat1 + i*M*K, p_mat2 + i*K*N, p_out + i*M*N, M, K, N, q_x+q_y-q_o); 
    }
    return ret;
}

/**
 * @brief Batch matrix multiplication with relative position keys
 * @param p_weight_emb_k Relative position embedding for keys
 * @param p_emb_k Key embeddings
 * @param p_mat2 Second matrix
 * @param p_out Output matrix
 * @param n_q Query sequence length
 * @param n_k Key sequence length
 * @param dim_head Head dimension
 * @param headers Number of attention heads
 * @param q_x First matrix scale
 * @param q_y Second matrix scale
 * @param q_o Output scale
 * @param max_rel Maximum relative position
 * @return Operation status
 */
static int32_t luna_bmm_rel_key_int8(int8_t* p_weight_emb_k, int8_t* p_emb_k, int8_t* p_mat2, int8_t* p_out, 
    int32_t n_q, int32_t n_k, int32_t dim_head, int32_t headers, 
    int32_t q_x, int32_t q_y, int32_t q_o,
    int32_t max_rel)
{
    int ret = 0;
    int8_t *p_emb_k_new;
    
    for (int i = 0; i < n_q; i++) {
        if (0 - i >= -max_rel && n_k - i <= max_rel) {  //not overflow
            p_emb_k_new = p_weight_emb_k + ((0 - i) + max_rel)*dim_head;
        } else {
            for (int j = 0; j < n_k; j++) {
                int rel = j - i;
                if (rel < -max_rel) rel = -max_rel;
                if (rel > max_rel) rel = max_rel;
                rel += max_rel;
                luna_memcpy_i8o8(p_emb_k + j*dim_head, p_weight_emb_k + rel * dim_head, dim_head);
            }
            p_emb_k_new = p_emb_k;
        }
        ret |= luna_split_mat_mul_bias_i8i8i32o8(p_emb_k_new, p_mat2 + i*dim_head*headers, 0, p_out + i*n_k*headers, n_k, dim_head, headers, q_x+q_y-q_o); 
    }
    return ret;  
}

/**
 * @brief Batch matrix multiplication with relative position values
 * @param p_mat1 First matrix
 * @param p_weight_emb_v Relative position embedding for values
 * @param p_emb_v Value embeddings
 * @param p_out Output matrix
 * @param n_q Query sequence length
 * @param headers Number of attention heads
 * @param n_k Key sequence length
 * @param dim_head Head dimension
 * @param q_x First matrix scale
 * @param q_y Second matrix scale
 * @param q_o Output scale
 * @param max_rel Maximum relative position
 * @return Operation status
 */
static int32_t luna_bmm_rel_value_int8(int8_t* p_mat1, int8_t* p_weight_emb_v, int8_t* p_emb_v, int8_t* p_out, 
    int32_t n_q, int32_t headers, int32_t n_k, int32_t dim_head, 
    int32_t q_x, int32_t q_y, int32_t q_o,
    int32_t max_rel)
{
    int ret = 0;
    int8_t *p_emb_v_new;
    
    for (int i = 0; i < n_q; i++) {
        if (0 - i >= -max_rel && n_k - i <= max_rel) {  //not overflow
            p_emb_v_new = p_weight_emb_v + ((0 - i) + max_rel)*dim_head;
        } else {
            for (int j = 0; j < n_k; j++) {
                int rel = j - i;
                if (rel < -max_rel) rel = -max_rel;
                if (rel > max_rel) rel = max_rel;
                rel += max_rel;
                luna_memcpy_i8o8(p_emb_v + (0*n_k + j)*dim_head, p_weight_emb_v + rel * dim_head, dim_head);
            }
            p_emb_v_new = p_emb_v;
        }
        ret |= luna_split_mat_mul_bias_i8i8i32o8(p_mat1 + i*headers*n_k, p_emb_v_new + 0*n_k*dim_head, 0, p_out + i*headers*dim_head, headers, n_k, dim_head, q_x+q_y-q_o); 
    }
    return ret;
}

/**
 * @brief Main self-attention computation for quantized integers
 * @param p_input Input tensor (n, c)
 * @param p_weight_q Weight matrix for queries
 * @param p_bias_q Bias for queries
 * @param p_weight_k Weight matrix for keys
 * @param p_bias_k Bias for keys
 * @param p_weight_v Weight matrix for values
 * @param p_bias_v Bias for values
 * @param p_weight_out Output weight matrix
 * @param p_bias_out Output bias
 * @param p_output Output tensor
 * @param p_temp Temporary buffer
 * @param dim_in Input dimension
 * @param dim_out Output dimension
 * @param headers Number of attention heads
 * @param dim_head Dimension per head
 * @param n Sequence length
 * @param scale Scaling factor
 * @param q_input Input scale
 * @param q_weight_q Query weight scale
 * @param q_weight_k Key weight scale
 * @param q_weight_v Value weight scale
 * @param q_output_q Query output scale
 * @param q_output_k Key output scale
 * @param q_output_v Value output scale
 * @param q_output_bmm0 BMM0 output scale
 * @param q_weight_scale Weight scaling factor
 * @param q_output_scale Output scaling factor
 * @param q_output_softmax Softmax output scale
 * @param q_output_bmm1 BMM1 output scale
 * @param q_weight_o Output weight scale
 * @param q_output Output scale
 * @param p_weight_emb_k Relative position embedding for keys
 * @param p_weight_emb_v Relative position embedding for values
 * @param q_x_bmm2 BMM2 input X scale
 * @param q_y_bmm2 BMM2 input Y scale
 * @param q_o_bmm2 BMM2 output scale
 * @param q_x_bmm3 BMM3 input X scale
 * @param q_y_bmm3 BMM3 input Y scale
 * @param q_o_bmm3 BMM3 output scale
 * @param q_x_add1 Add1 input X scale
 * @param q_y_add1 Add1 input Y scale
 * @param q_o_add1 Add1 output scale
 * @param q_x_add2 Add2 input X scale
 * @param q_y_add2 Add2 input Y scale
 * @param q_o_add2 Add2 output scale
 * @param max_rel Maximum relative position
 * @return Operation status
 */
static int32_t luna_self_attention_int_trans(int8_t *p_input, //(n,c)
    int8_t *p_weight_q, int32_t *p_bias_q, //(headers*dim_head,c),(headers*dim_head) @psram
    int8_t *p_weight_k, int32_t *p_bias_k, //(headers*dim_head,c),(headers*dim_head) @psram
    int8_t *p_weight_v, int32_t *p_bias_v, //(headers*dim_head,c),(headers*dim_head) @psram
    int8_t *p_weight_out, int32_t *p_bias_out,//(dim_out,dim_head) @psram
    int8_t *p_output, int8_t *p_temp,
    uint32_t dim_in, uint32_t dim_out, uint32_t headers, uint32_t dim_head, uint32_t n,
    int32_t scale /* Q15 */, 
    int32_t q_input, int32_t q_weight_q, int32_t q_weight_k, int32_t q_weight_v,
    int32_t q_output_q, int32_t q_output_k, int32_t q_output_v,
    int32_t q_output_bmm0,
    int32_t q_weight_scale, int32_t q_output_scale,
    int32_t q_output_softmax, 
    int32_t q_output_bmm1,
    int32_t q_weight_o, int32_t q_output,
    int8_t *p_weight_emb_k, int8_t *p_weight_emb_v,  // (2*max_rel+1,dim_head), (2*max_rel+1,dim_head) @share
    int32_t q_x_bmm2, int32_t q_y_bmm2, int32_t q_o_bmm2, // rel_key_bmm
    int32_t q_x_bmm3, int32_t q_y_bmm3, int32_t q_o_bmm3, // rel_val_bmm
    int32_t q_x_add1, int32_t q_y_add1, int32_t q_o_add1, // rel_key_add
    int32_t q_x_add2, int32_t q_y_add2, int32_t q_o_add2,
    const int max_rel) // rel_val_add
{
    int32_t ret = 0;
    uint32_t n_q = n; 
    uint32_t n_k = n;

    // Allocate temporary buffers
    int8_t *p_input_T = (int8_t *)(p_temp); p_temp += n_q*dim_in*sizeof(int8_t);
    int8_t *p_q = (int8_t *)(p_temp); p_temp += headers*dim_head*n_q*sizeof(int8_t);
    int8_t *p_k = (int8_t *)(p_temp); p_temp += headers*dim_head*n_k*sizeof(int8_t); 
    int8_t *p_dots = (int8_t *)(p_temp); p_temp += headers*n_q*n_k*sizeof(int8_t);  
    
    // Reuse buffers for efficiency
    int8_t *p_q_emb = p_k;
    int8_t *p_emb_k = (int8_t *)(p_temp); p_temp += n_k*dim_head*sizeof(int8_t); 
    int8_t *p_v = p_k;
    int8_t *p_out = p_q;  
    int8_t *p_emb_v = p_emb_k;
    int8_t *p_out_emb = p_k;
    int8_t *p_out2 = p_input_T; 

    // Allocate softmax buffer
    int32_t *p_softmax = (int32_t *)p_temp;
    int8_t *p_dots_emb = (int8_t *)(p_temp);
    int8_t *p_dots_emb_T = (int8_t *)(p_temp) + headers*n_q*n_k*sizeof(int8_t);
    int8_t *p_out_emb_T = (int8_t *)p_temp;

    p_temp += MAX(2*n_q*n_k*sizeof(int32_t), MAX(2*headers*n_q*n_k*sizeof(int8_t), n_q*headers*dim_in*sizeof(int8_t)));

    uint32_t shape[3], axis[3];

    // Step 1: Project input to query, key, and value representations
    ret |= luna_split_mat_trans_i8o8(p_input, p_input_T, n_q, dim_in); // Transpose input
    ret |= luna_split_mat_mul_bias_i8i8i32o8(p_weight_q, p_input_T, p_bias_q, p_q, headers*dim_head, dim_in, n_q, q_input + q_weight_q - q_output_q);
    ret |= luna_split_mat_mul_bias_i8i8i32o8(p_weight_k, p_input_T, p_bias_k, p_k, headers*dim_head, dim_in, n_k, q_input + q_weight_k - q_output_k);
    
    // Step 2: Scale queries
    ret |= luna_scale_i8i8o8(p_q, (int8_t)scale, p_q, headers*dim_head * n_q, q_output_q+q_weight_scale-q_output_scale);

    // Step 3: Transpose queries for attention computation
    for (uint32_t i = 0; i < headers; i++){
        ret |= luna_mat_trans_i8o8(p_q + i*dim_head*n_q, p_q + i*dim_head*n_q, dim_head, n_q);
    }

    // Step 4: Compute attention scores (Q @ K^T)
    ret |= luna_bmm_int8(p_q, p_k, p_dots, headers, n_q, dim_head, n_k, q_output_scale, q_output_k, q_output_bmm0);

    // Step 5: Add relative position embeddings to attention scores
    shape[0] = headers, shape[1] = n_q, shape[2] = dim_head;
    axis[0] = 1, axis[1] = 2, axis[2] = 0;
    luna_trans_axis_i8o8(p_q, p_q_emb, shape, axis, 3);
    
    luna_bmm_rel_key_int8(p_weight_emb_k, p_emb_k, p_q_emb, p_dots_emb, n_q, n_k, dim_head, headers, q_x_bmm2, q_y_bmm2, q_o_bmm2, max_rel);
    luna_split_mat_trans_i8o8(p_dots_emb, p_dots_emb_T, n_q*n_k, headers);
    
    luna_add_int8(p_dots, p_dots_emb_T, p_dots, 0, headers*n_q*n_k, q_x_add1, q_y_add1, q_o_add1);

    // Step 6: Apply softmax
    for (uint32_t i = 0; i < headers; i++){
        luna_softmax_int8(p_dots + i*n_k*n_q, p_dots + i*n_k*n_q, (int8_t *)p_softmax, n_q, n_k, q_o_add1, q_output_softmax);
    }

    // Step 7: Compute values using attention weights
    ret |= luna_split_mat_mul_bias_i8i8i32o8(p_weight_v, p_input_T, p_bias_v, p_v, headers*dim_head, dim_in, n_k, q_input + q_weight_v - q_output_v);
    
    for (uint32_t i = 0; i < headers; i++){
        int8_t *p_v_h = p_v + i*dim_head*n_k;
        int8_t *p_dots_h = p_dots + i*n_k*n_q;
        int8_t *p_out_h = p_out + i*dim_head*n_q;
        ret |= luna_mat_trans_i8o8(p_v_h, p_v_h, dim_head, n_q);
        ret |= luna_mat_mul_i8i8o8(p_dots_h, p_v_h, p_out_h, n_q, n_k, dim_head, (q_output_v+q_output_softmax) - q_output_bmm1);
        ret |= luna_mat_trans_i8o8(p_out_h, p_out_h, n_q, dim_head);
    }

    // Step 8: Add relative position embeddings to output values
    shape[0] = headers, shape[1] = n_q, shape[2] = n_k;
    axis[0] = 1, axis[1] = 0, axis[2] = 2;
    luna_trans_axis_i8o8(p_dots, p_dots_emb_T, shape, axis, 3);
    
    luna_bmm_rel_value_int8(p_dots_emb_T, p_weight_emb_v, p_emb_v, p_out_emb, n_q, headers, n_k, dim_head, q_x_bmm3, q_y_bmm3, q_o_bmm3, max_rel);
    luna_split_mat_trans_i8o8(p_out_emb, p_out_emb_T, n_q, headers*dim_head);
    
    luna_add_int8(p_out, p_out_emb_T, p_out, 0, n_q*headers*dim_head, q_x_add2, q_y_add2, q_o_add2);

    // Step 9: Final projection
    ret |= luna_split_mat_mul_bias_i8i8i32o8(p_weight_out, p_out, p_bias_out, p_out2, dim_out, headers*dim_head, n_q, q_o_add2+q_weight_o-q_output);
    ret |= luna_split_mat_trans_i8o8(p_out2, p_output, dim_out, n_q);

    return ret;
}

/**
 * @brief Main multi-head attention operation for quantized integers
 * @param X Input tensor
 * @param W_q Query weight tensor
 * @param Bias_q Query bias tensor
 * @param W_k Key weight tensor
 * @param Bias_k Key bias tensor
 * @param W_v Value weight tensor
 * @param Bias_v Value bias tensor
 * @param W_o Output weight tensor
 * @param Bias_o Output bias tensor
 * @param emb_pos_qk Relative position embedding for keys
 * @param emb_pos_qkv Relative position embedding for values
 * @param Y Output tensor
 * @param workspace Workspace buffer
 * @param attrs Attention attributes
 * @return Operation status
 */
int32_t multiheadattention_luna(tTensor *X, tTensor *W_q, tTensor *Bias_q, tTensor *W_k, tTensor *Bias_k, 
                                tTensor *W_v, tTensor *Bias_v, tTensor *W_o, tTensor *Bias_o, tTensor *emb_pos_qk,
                                tTensor *emb_pos_qkv, tTensor *Y, tTensor *workspace, MultiheadAttentionAttrs *attrs) 
{
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    // Extract tensor pointers
    int8_t *p_input       = (int8_t *)X->dptr_;
    int8_t *p_weight_q    = (int8_t *)W_q->dptr_;
    int32_t*p_bias_q      = (int32_t *)Bias_q->dptr_;
    int8_t *p_weight_k    = (int8_t *)W_k->dptr_;
    int32_t*p_bias_k      = (int32_t *)Bias_k->dptr_;
    int8_t *p_weight_v    = (int8_t *)W_v->dptr_;
    int32_t*p_bias_v      = (int32_t *)Bias_v->dptr_;
    int8_t *p_weight_o    = (int8_t *)W_o->dptr_;
    int32_t*p_bias_o      = (int32_t *)Bias_o->dptr_;
    int8_t *p_weight_pos_qk   = (int8_t *)emb_pos_qk->dptr_;
    int8_t *p_weight_pos_qkv  = (int8_t *)emb_pos_qkv->dptr_;
    int8_t *p_output      = (int8_t *)Y->dptr_;
    int8_t *p_temp        = (int8_t *)workspace->dptr_;

    // Extract dimensions
    uint32_t num_head     = attrs->headers;
    uint32_t head_dim     = attrs->head_dim;
    uint32_t seq_len      = X->shape_.dims_[0];
    uint32_t embd_dim     = W_q->shape_.dims_[1];
    uint32_t hid_size     = W_q->shape_.dims_[0];

    uint32_t dim_in       = embd_dim;
    uint32_t dim_out      = embd_dim;
    uint32_t headers      = num_head;
    uint32_t dim_head     = head_dim;
    uint32_t n            = seq_len;
    uint32_t n_q          = n;
    uint32_t n_k          = n;
    uint32_t max_rel      = emb_pos_qk->shape_.dims_[0] / 2;
    
    // Extract scales
    int32_t scale         = attrs->iqmul_scalar;
    int32_t q_input       = X->scale_;
    int32_t q_weight_q    = W_q->scale_;
    int32_t q_weight_k    = W_k->scale_;
    int32_t q_wegiht_v    = W_v->scale_;
    int32_t q_output_q    = attrs->scale_iqmul_x;
    int32_t q_output_k    = attrs->scale_bmm0_y;
    int32_t q_output_v    = attrs->scale_bmm1_y;

    int32_t q_output_bmm0 = attrs->scale_bmm0_o;
    int32_t q_iqmul_scalar = attrs->scale_iqmul_y;
    int32_t q_iqmul_output = attrs->scale_iqmul_o;
    int32_t q_output_softmax = 7;
    int32_t q_output_bmm1 = attrs->scale_bmm1_o;
    int32_t q_weight_o    = W_o->scale_;
    int32_t q_output      = Y->scale_;

    // Copy relative position embeddings to temp buffer
    int8_t* p_weight_emb_k = p_temp; p_temp += (2 * max_rel + 1) * head_dim;
    opi_psram_cpy_out(p_weight_emb_k, p_weight_pos_qk, (2 * max_rel + 1) * head_dim);
    int8_t* p_weight_emb_v = p_temp; p_temp += (2 * max_rel + 1) * head_dim;
    opi_psram_cpy_out(p_weight_emb_v, p_weight_pos_qkv, (2 * max_rel + 1) * head_dim);

    // Extract additional scales
    int32_t q_x_bmm2 = emb_pos_qk->scale_;
    int32_t q_y_bmm2 = attrs->scale_bmm2_y;
    int32_t q_o_bmm2 = attrs->scale_bmm2_o;

    int32_t q_x_bmm3 = 7;
    int32_t q_y_bmm3 = attrs->scale_bmm3_y;
    int32_t q_o_bmm3 = attrs->scale_bmm3_o;

    int32_t q_x_add1 = attrs->scale_bmm0_o;
    int32_t q_y_add1 = attrs->scale_bmm2_o;
    int32_t q_o_add1 = attrs->scale_iqadd1_o;

    int32_t q_x_add2 = attrs->scale_bmm1_o;;
    int32_t q_y_add2 = attrs->scale_bmm3_o;
    int32_t q_o_add2 = attrs->scale_iqadd2_o;

    // Execute main attention computation
#if (defined(WIN32) || defined(linux))
    ret = luna_self_attention_int_trans(p_input, p_weight_q, p_bias_q, p_weight_k, p_bias_k, p_weight_v, p_bias_v,
                                        p_weight_o, p_bias_o, p_output, p_temp, dim_in, dim_out, headers, dim_head, n,
                                        scale, q_input, q_weight_q, q_weight_k, q_wegiht_v, q_output_q, q_output_k,
                                        q_output_v, q_output_bmm0, q_iqmul_scalar, q_iqmul_output, q_output_softmax,
                                        q_output_bmm1, q_weight_o, q_output, p_weight_emb_k, p_weight_emb_v,
                                        q_x_bmm2, q_y_bmm2, q_o_bmm2, q_x_bmm3, q_y_bmm3, q_o_bmm3, q_x_add1,
                                        q_y_add1, q_o_add1, q_x_add2, q_y_add2, q_o_add2, max_rel);
#else
    #include "lunaext_attention.h"
    ret = nlang_self_attention_int_trans(p_input, p_weight_q, p_bias_q, p_weight_k, p_bias_k, p_weight_v, p_bias_v,
                                         p_weight_o, p_bias_o, p_output, p_temp, dim_in, dim_out, headers, dim_head, n,
                                         scale, q_input, q_weight_q, q_weight_k, q_wegiht_v, q_output_q, q_output_k,
                                         q_output_v, q_output_bmm0, q_iqmul_scalar, q_iqmul_output, q_output_softmax,
                                         q_output_bmm1, q_weight_o, q_output, p_weight_emb_k, p_weight_emb_v,
                                         q_x_bmm2, q_y_bmm2, q_o_bmm2, q_x_bmm3, q_y_bmm3, q_o_bmm3, q_x_add1,
                                         q_y_add1, q_o_add1, q_x_add2, q_y_add2, q_o_add2, max_rel);
#endif
    return ret;
}

#endif //_MULTIHEADATTENTION_LUNA_H_