#ifndef _SPARIFYFFNINT_LUNA_H_
#define _SPARIFYFFNINT_LUNA_H_

#include <math.h>

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
 * @brief Extract 4-bit value from 8-bit data
 * @param bit8 8-bit input value
 * @param odd Flag to select lower (0) or upper (1) 4 bits
 * @return Extracted 4-bit value with sign extension
 */
static inline int8_t luna_extract_bit4(int8_t bit8, int8_t odd)
{
    if (odd) {
        // Extract lower 4 bits with sign extension
        return (((int32_t)bit8)<<(24))>>(28);
    } else {
        // Extract upper 4 bits with sign extension
        return (((int32_t)bit8)<<(28))>>(28);
    }
}

/**
 * @brief Sparse FFN integer computation with dynamic selection
 * @param p_input Input tensor (1, D)
 * @param p_weight_m0 First weight matrix (8, Dh/8, Di)
 * @param p_bias_m0 First bias vector (8, Dh/8)
 * @param p_weight_m1 Second weight matrix (8, D0, Dh/8)
 * @param p_bias_m1 Second bias vector (8, D0)
 * @param p_weight_mask Mask weight matrix (8, D)
 * @param p_bias_mask Mask bias vector (8, 1)
 * @param p_output Output tensor (1, Do)
 * @param p_temp Temporary buffer
 * @param dim_in Input dimension
 * @param dim_hidden Hidden dimension
 * @param dim_out Output dimension
 * @param seq_len Sequence length (should be 1)
 * @param group_num Number of groups for masking
 * @param q_input_m0 Input scale for first layer
 * @param q_weight_m0 Weight scale for first layer
 * @param q_output_m0 Output scale for first layer
 * @param q_input_m1 Input scale for second layer
 * @param q_weight_m1 Weight scale for second layer
 * @param q_output_m1 Output scale for second layer
 * @param q_input_mask Input scale for mask layer
 * @param q_weight_mask Weight scale for mask layer
 * @param q_output_mask Output scale for mask layer
 * @param is_4bit_m0 Flag indicating if first layer uses 4-bit weights
 * @param is_4bit_m1 Flag indicating if second layer uses 4-bit weights
 * @return Operation result status
 */
static int32_t luna_ffn_int_trans_dec(int8_t *p_input,  // (1,D)
    int8_t *p_weight_m0, int32_t *p_bias_m0, // (8, Dh/8, Di) weight@psram + bias@share
    int8_t *p_weight_m1, int32_t *p_bias_m1, // (8, D0, Dh/8) weight@psram + bias@share
    int8_t *p_weight_mask, int32_t *p_bias_mask, //(8, D)
    int8_t *p_output, int8_t *p_temp,
    uint32_t dim_in, uint32_t dim_hidden, uint32_t dim_out, uint32_t seq_len, uint32_t group_num, 
    int32_t q_input_m0, int32_t q_weight_m0, int32_t q_output_m0, 
    int32_t q_input_m1, int32_t q_weight_m1, int32_t q_output_m1,
    int32_t q_input_mask, int32_t q_weight_mask, int32_t q_output_mask,
    int8_t is_4bit_m0, int8_t is_4bit_m1)
{
    /*
        ## Parameter Description
        1. seq_len = 1 
        2. batch = 1
        3. Only 8-bit operations supported

        ## Packing Process
        1. w0 parameters don't need processing, (Dh, Di) => (8, Dh/8, Di)
        2. w1 parameters need processing, (Do, Dh) => (Do, 8, Dh/8) => (8, D0, Dh/8)

        ## Computation Process
        1. mask = w_mask * input, (8, D)*(D, 1) => (8, 1)  
        2. max_pos = topk(mask, 1), (8, 1) => (1)
        3. w0_mask = w0[max_pos]
        4. w1_mask = w1[max_pos]
        5. hidden_mask = w0_mask * input (Dh/8, D)*(D, 1) => (Dh/8, 1)
        6. output = w1_mask * hidden_mask (Do, Dh/8)*(Dh/8, 1) => (Do, 1)  
    */  
    
    int ret = T_ERR_NO_IMPLEMENTED;
    uint32_t shift_mask, shift_m0, shift_m1, dim_hidden_mask;

    int8_t *p_w_m0_mask, *p_w_m1_mask;
    int32_t *p_bias_m0_mask,*p_bias_m1_mask;

    if(1 != seq_len)
        return ret;
    if(dim_hidden%group_num !=0 )
        return ret;

    dim_hidden_mask = dim_hidden/group_num;
    shift_mask = q_input_mask+q_weight_mask-q_output_mask;
    shift_m0 = q_input_m0+q_weight_m0-q_output_m0;
    shift_m1 = q_input_m1+q_weight_m1-q_output_m1;

    ret = T_SUCCESS;
    int8_t *p_mask = p_temp; p_temp += group_num*sizeof(int8_t);
    int8_t *p_hidden_mask = p_temp; p_temp += dim_hidden/group_num*sizeof(int8_t);
    int32_t *p_max_val_pos = (int32_t *)p_temp; p_temp += 2*sizeof(int32_t);
    
    if (group_num > 1) {
        ret |= luna_split_mat_mul_bias_i8i8i32o8(p_weight_mask, p_input, p_bias_mask, p_mask, group_num, dim_in, 1, shift_mask); 
        ret |= luna_max_i8o32(p_mask, p_max_val_pos, group_num);
    } else {
        p_max_val_pos[1] = 0;
    }

    if (is_4bit_m0) {
        p_w_m0_mask = p_weight_m0 + ((p_max_val_pos[1]*dim_hidden_mask*dim_in)>>1);
    } else {
        p_w_m0_mask = p_weight_m0 + p_max_val_pos[1]*dim_hidden_mask*dim_in;
    }
    p_bias_m0_mask = p_bias_m0 + p_max_val_pos[1]*dim_hidden_mask;
    if (is_4bit_m0) {
        p_w_m1_mask = p_weight_m1 + ((p_max_val_pos[1]*dim_out*dim_hidden_mask)>>1);
    } else {
       p_w_m1_mask = p_weight_m1 + p_max_val_pos[1]*dim_out*dim_hidden_mask; 
    }
    p_bias_m1_mask = p_bias_m1;

    if (is_4bit_m0) {
        ret |= luna_split_mat_mul_bias_i4i8i32o8(p_w_m0_mask, p_input, p_bias_m0_mask, p_hidden_mask, dim_hidden_mask, dim_in, 1, shift_m0); 
    } else {
        ret |= luna_split_mat_mul_bias_i8i8i32o8(p_w_m0_mask, p_input, p_bias_m0_mask, p_hidden_mask, dim_hidden_mask, dim_in, 1, shift_m0); 
    }

    ret |= luna_relu_i8o8(p_hidden_mask, p_hidden_mask, dim_hidden_mask, 0);
    if (is_4bit_m1) {
        ret |= luna_split_mat_mul_bias_i4i8i32o8(p_w_m1_mask, p_hidden_mask, p_bias_m1_mask, p_output, dim_out, dim_hidden_mask, 1, shift_m1); 
    } else {
        ret |= luna_split_mat_mul_bias_i8i8i32o8(p_w_m1_mask, p_hidden_mask, p_bias_m1_mask, p_output, dim_out, dim_hidden_mask, 1, shift_m1); 
    } 

    return ret;
}

/**
 * @brief Main sparse FFN integer operation implementation
 * @param X Input tensor
 * @param weight1 First weight tensor
 * @param bias1 First bias tensor
 * @param weight2 Second weight tensor
 * @param bias2 Second bias tensor
 * @param weight3 Mask weight tensor
 * @param bias3 Mask bias tensor
 * @param workspace Workspace buffer
 * @param Y Output tensor
 * @param attrs Sparse FFN attributes
 * @return Operation result status
 */
int32_t sparifyffnint_luna(tTensor *X, tTensor *weight1, tTensor *bias1, tTensor *weight2, tTensor *bias2, 
                          tTensor *weight3, tTensor *bias3, tTensor *workspace, tTensor *Y, SparifyFFNIntAttrs *attrs) 
{
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    int8_t *p_input       = (int8_t *)X->dptr_;
    int8_t *p_weight_m0   = (int8_t *)weight1->dptr_;
    int8_t *p_weight_m1   = (int8_t *)weight2->dptr_;
    int8_t *p_weight_mask = (int8_t *)weight3->dptr_;
    int32_t *p_bias_m0    = (int32_t *)bias1->dptr_;
    int32_t *p_bias_m1    = (int32_t *)bias2->dptr_;
    int32_t *p_bias_mask  = (int32_t *)bias3->dptr_;
    int8_t *p_output      = (int8_t *)Y->dptr_;
    int8_t *p_temp        = (int8_t *)workspace->dptr_;

    int32_t seq_len       = X->shape_.dims_[0] * X->shape_.dims_[1];
    int32_t dim_in        = X->shape_.dims_[2];
    int32_t dim_hidden    = bias1->shape_.dims_[0];
    int32_t dim_out       = bias2->shape_.dims_[0];
    int32_t group_num     = attrs->group_num;
    int32_t q_input_m0    = X->scale_;
    int32_t q_weight_m0   = weight1->scale_;
    int32_t q_output_m0   = attrs->fc1_out_scale;
    int32_t q_input_m1    = attrs->fc1_out_scale;
    int32_t q_weight_m1   = weight2->scale_;
    int32_t q_output_m1   = Y->scale_;
    int32_t q_input_mask  = X->scale_;
    int32_t q_weight_mask = weight3->scale_;
    int32_t q_output_mask = attrs->mask_out_scale;
    
#if (defined(WIN32) || defined(linux))
    ret = luna_ffn_int_trans_dec(p_input, p_weight_m0, p_bias_m0, p_weight_m1, p_bias_m1, p_weight_mask, p_bias_mask,
                                p_output, p_temp, dim_in, dim_hidden, dim_out, seq_len, group_num, q_input_m0, q_weight_m0, 
                                q_output_m0, q_input_m1, q_weight_m1, q_output_m1, q_input_mask, q_weight_mask, q_output_mask, 1, 1);
#else
    ret = nlang_ffn_int_trans_dec(p_input, p_weight_m0, p_bias_m0, p_weight_m1, p_bias_m1, p_weight_mask, p_bias_mask,
                                p_output, p_temp, dim_in, dim_hidden, dim_out, seq_len, group_num, q_input_m0, q_weight_m0, 
                                q_output_m0, q_input_m1, q_weight_m1, q_output_m1, q_input_mask, q_weight_mask, q_output_mask, 1, 1);
#endif
    return ret;
}

#endif  //_FFNINT_LUNA_H_