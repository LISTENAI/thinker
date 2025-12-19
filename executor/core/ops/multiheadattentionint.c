#undef __OP__
#define __OP__ MultiheadAttention
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_ARCS
#include "./arcs/multiheadattentionint.h"
#endif

/**
 * Forward pass implementation for Integer Quantized Multi-head Attention operator
 * Performs multi-head attention computation on input tensor
 * @param op: Operator structure containing attention attributes
 * @param tensors: Array of input/output tensors (input, weights, biases, embeddings, output, workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Validate tensor count
    CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_ + 1));
    
    // Get attention attributes
    MultiheadAttentionAttrs *attrs = (MultiheadAttentionAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
    // Get input, weights, biases, embeddings, output, and workspace tensors
    tTensor *input = tensors[0];
    tTensor *weight_q = tensors[1];
    tTensor *bias_q = tensors[2];
    tTensor *weight_k = tensors[3];
    tTensor *bias_k = tensors[4];
    tTensor *weight_v = tensors[5];
    tTensor *bias_v = tensors[6];
    tTensor *weight_p = tensors[7];
    tTensor *bias_p = tensors[8];
    tTensor *emb_keys = tensors[9];
    tTensor *emb_values = tensors[10];
    
    tTensor *output = tensors[op->num_input_];
    tTensor *workspace = NULL;
    
#if THINKER_USE_ARCS
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

    if (num_tensor == op->num_input_ + op->num_output_ + 1) {
        workspace = ((tTensor **)tensors)[op->num_input_ + op->num_output_];
    }
    
    // Call hardware-specific multi-head attention implementation
    ret = multiheadattention_luna(input, weight_q, bias_q, weight_k, bias_k, weight_v, bias_v,
                                 weight_p, bias_p, emb_keys, emb_values, output, workspace, attrs);

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","MultiheadAttentionInt", total_t);
#endif
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__