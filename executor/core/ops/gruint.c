#undef __OP__
#define __OP__ GRUInt
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_VENUS
#include "./venus/gruint.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/gruint.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/gruint.h"
#endif

static size_t aligned_tensor_bytes(tTensor* tensor) {
    size_t bytes = getTensorDataSize(tensor);
    return (bytes + 15U) & ~(size_t)15U;
}

/**
 * Forward pass implementation for Gated Recurrent Unit Integer operator
 * Performs GRU computation with integer quantization
 * @param op: Operator structure containing GRU attributes
 * @param tensors: Array of input/output tensors (input, i2h_weights, h2h_weights, i2h_bias, h2h_bias, output, hidden_output, optional workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list for weight data handling
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator* op, tTensor** tensors, int32_t num_tensor, tDMA_List* list) {
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || list == NULL ||
                        (op->num_input_ != 5 && op->num_input_ != 6) ||
                        op->num_output_ != 2) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    int32_t expected_tensor = op->num_input_ + op->num_output_ + 1;
    if (list->total_ > 0) expected_tensor++;
#if THINKER_PARAM_CHECK
    if (num_tensor != expected_tensor) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    GRUIntAttrs* attr = (GRUIntAttrs*)((int8_t*)op + op->attr_offset_);
    int32_t weight_idx = op->num_input_ == 6 ? 1 : 0;
    tTensor* input = tensors[0];
    tTensor* i2h_w = tensors[weight_idx + 1];
    tTensor* h2h_w = tensors[weight_idx + 2];
    tTensor* i2h_bias = tensors[weight_idx + 3];
    tTensor* h2h_bias = tensors[weight_idx + 4];

    tTensor* output = tensors[op->num_input_];
    tTensor* hidden_o = tensors[op->num_input_ + 1];
    tTensor* workspace = tensors[op->num_input_ + op->num_output_];

#if THINKER_PARAM_CHECK
    if (input == NULL || i2h_w == NULL || h2h_w == NULL ||
                        i2h_bias == NULL || h2h_bias == NULL || output == NULL ||
                        hidden_o == NULL || workspace == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if (attr->layout > 1 || attr->direction > 1 ||
                        attr->input_size == 0 || attr->hidden_size == 0 ||
                        input->shape_.ndim_ != 3) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    int32_t seq_dim = attr->layout == 0 ? 0 : 1;
    int32_t batch_dim = attr->layout == 0 ? 1 : 0;
    uint32_t batch = input->shape_.dims_[batch_dim];
#if THINKER_PARAM_CHECK
    if (input->shape_.dims_[seq_dim] == 0 || batch == 0 ||
                        input->shape_.dims_[2] != attr->input_size ||
                        output->shape_.ndim_ != 3 ||
                        output->shape_.dims_[seq_dim] != input->shape_.dims_[seq_dim] ||
                        output->shape_.dims_[batch_dim] != batch ||
                        output->shape_.dims_[2] != attr->hidden_size ||
                        hidden_o->shape_.ndim_ != 3 ||
                        hidden_o->shape_.dims_[0] != 1 ||
                        hidden_o->shape_.dims_[1] != batch ||
                        hidden_o->shape_.dims_[2] != attr->hidden_size) {
        return (T_ERR_INVALID_DATA);
    }
#endif
#ifdef THINKER_USE_VENUS
    #if THINKER_PARAM_CHECK
    if (i2h_w->shape_.ndim_ != 2 ||
                        i2h_w->shape_.dims_[0] != attr->input_size ||
                        i2h_w->shape_.dims_[1] != attr->hidden_size * 3U ||
                        h2h_w->shape_.ndim_ != 2 ||
                        h2h_w->shape_.dims_[0] != attr->hidden_size ||
                        h2h_w->shape_.dims_[1] != attr->hidden_size * 3U) {
        return (T_ERR_INVALID_DATA);
    }
    #endif
#else
    #if THINKER_PARAM_CHECK
    if (i2h_w->shape_.ndim_ != 2 ||
                        i2h_w->shape_.dims_[0] != attr->hidden_size * 3U ||
                        i2h_w->shape_.dims_[1] != attr->input_size ||
                        h2h_w->shape_.ndim_ != 2 ||
                        h2h_w->shape_.dims_[0] != attr->hidden_size * 3U ||
                        h2h_w->shape_.dims_[1] != attr->hidden_size) {
        return (T_ERR_INVALID_DATA);
    }
    #endif
#endif
#if THINKER_PARAM_CHECK
    if (i2h_bias->shape_.ndim_ != 1 ||
                        i2h_bias->shape_.dims_[0] != attr->hidden_size * 3U ||
                        h2h_bias->shape_.ndim_ != 1 ||
                        h2h_bias->shape_.dims_[0] != attr->hidden_size * 3U) {
        return (T_ERR_INVALID_DATA);
    }

    if (input->dtype_ != Int8 || i2h_w->dtype_ != Int8 ||
                        h2h_w->dtype_ != Int8 || i2h_bias->dtype_ != Int32 ||
                        h2h_bias->dtype_ != Int32 || output->dtype_ != Int8 ||
                        hidden_o->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (input->zero_ != 0 || i2h_w->zero_ != 0 ||
                        h2h_w->zero_ != 0 || i2h_bias->zero_ != 0 ||
                        h2h_bias->zero_ != 0 || output->zero_ != 0 ||
                        hidden_o->zero_ != 0 || !isfinite(input->scale_) ||
                        !isfinite(i2h_w->scale_) || !isfinite(h2h_w->scale_) ||
                        !isfinite(output->scale_) || !isfinite(hidden_o->scale_) ||
                        floorf(input->scale_) != input->scale_ ||
                        floorf(i2h_w->scale_) != i2h_w->scale_ ||
                        floorf(h2h_w->scale_) != h2h_w->scale_ ||
                        floorf(output->scale_) != output->scale_ ||
                        floorf(hidden_o->scale_) != hidden_o->scale_ ||
                        output->scale_ != hidden_o->scale_) {
        return (T_ERR_INVALID_DATA);
    }
#endif

    // Initialize dummy tensors for hidden state and mask
    tTensor hidden_i_inst;
    hidden_i_inst.shape_.ndim_ = 0;
    tTensor *hidden_in = &hidden_i_inst;
    if(weight_idx == 1) {
        hidden_in = tensors[1];
#if THINKER_PARAM_CHECK
        if (hidden_in == NULL || hidden_in->dtype_ != Int8 ||
                            hidden_in->zero_ != 0 || hidden_in->shape_.ndim_ != 3 ||
                            hidden_in->shape_.dims_[0] != 1 ||
                            hidden_in->shape_.dims_[1] != batch ||
                            hidden_in->shape_.dims_[2] != attr->hidden_size ||
                            hidden_in->scale_ != hidden_o->scale_ ||
                            hidden_in->dptr_ == 0) {
            return (T_ERR_INVALID_DATA);
        }
#endif
    }
    
    tTensor mask;
    mask.shape_.ndim_ = 0;
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
    if (list->total_ != 0)
        getWeightData(list, 0);
#endif
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

#ifdef THINKER_USE_VENUS
    // Venus hardware implementation
    if (list->total_ > 0) {
        tTensor *dma_temp   = ((tTensor**)tensors)[op->num_input_ + op->num_output_ + 1];
        size_t dma_bytes = aligned_tensor_bytes(i2h_w) + aligned_tensor_bytes(h2h_w) +
                           aligned_tensor_bytes(i2h_bias) + aligned_tensor_bytes(h2h_bias);
#if THINKER_RUNTIME_CHECK
        if (dma_temp == NULL || dma_temp->dptr_ == 0 ||
                              dma_temp->mem_.type_ != 2 ||
                              getTensorDataSize(dma_temp) < dma_bytes) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
        tTensor i2h_w_temp  = i2h_w[0];
        i2h_w_temp.dptr_    = (addr_type)(dma_temp->dptr_);
        i2h_w_temp.mem_.type_ = 2;

        tTensor h2h_w_temp  = h2h_w[0];
        h2h_w_temp.dptr_ = (addr_type)((int8_t *)i2h_w_temp.dptr_ + aligned_tensor_bytes(&i2h_w_temp));
        h2h_w_temp.mem_.type_ = 2;

        tTensor i2h_bias_temp = i2h_bias[0];
        i2h_bias_temp.dptr_ = (addr_type)((int8_t *)h2h_w_temp.dptr_ + aligned_tensor_bytes(&h2h_w_temp));
        i2h_bias_temp.mem_.type_ = 2;

        tTensor h2h_bias_temp = h2h_bias[0];
        h2h_bias_temp.dptr_ = (addr_type)((int8_t *)i2h_bias_temp.dptr_ + aligned_tensor_bytes(&i2h_bias_temp));
        h2h_bias_temp.mem_.type_ = 2;


        THINKER_RET_CHECK(gruint_luna(input, hidden_in, &i2h_w_temp, &h2h_w_temp,
                      &i2h_bias_temp, &h2h_bias_temp,
                      output, hidden_o, attr, workspace), "gruint_luna");
    } else {
        THINKER_RET_CHECK(gruint_luna(input, hidden_in, i2h_w, h2h_w, i2h_bias, h2h_bias,
                      output, hidden_o, attr, workspace), "gruint_luna");
    }
#elif defined(THINKER_USE_ARCS) || defined(THINKER_USE_VENUSA)
    // ARC/VENUSA hardware implementation
    if(list->total_ > 0) {
            tTensor *dma_temp = tensors[op->num_input_ + op->num_output_ + 1];
            size_t dma_bytes = aligned_tensor_bytes(i2h_w) + aligned_tensor_bytes(h2h_w) +
                               aligned_tensor_bytes(i2h_bias) + aligned_tensor_bytes(h2h_bias);
#if THINKER_RUNTIME_CHECK
            if (dma_temp == NULL || dma_temp->dptr_ == 0 ||
                                  dma_temp->mem_.type_ != 2 ||
                                  getTensorDataSize(dma_temp) < dma_bytes) {
                return (T_ERR_NO_WORKSPACE);
            }
#endif
            tTensor i2h_w_temp  = i2h_w[0];
            i2h_w_temp.dptr_    = (addr_type)(dma_temp->dptr_);
            i2h_w_temp.mem_.type_ = 2;
            tTensor h2h_w_temp  = h2h_w[0];
            h2h_w_temp.dptr_    = (addr_type)((int8_t *)i2h_w_temp.dptr_ + aligned_tensor_bytes(&i2h_w_temp));
            h2h_w_temp.mem_.type_ = 2;
            tTensor i2h_bias_temp = i2h_bias[0];
            i2h_bias_temp.dptr_ = (addr_type)((int8_t *)h2h_w_temp.dptr_ + aligned_tensor_bytes(&h2h_w_temp));
            i2h_bias_temp.mem_.type_ = 2;
            tTensor h2h_bias_temp     = h2h_bias[0];
            h2h_bias_temp.dptr_ = (addr_type)((int8_t *)i2h_bias_temp.dptr_ + aligned_tensor_bytes(&i2h_bias_temp));
            h2h_bias_temp.mem_.type_ = 2;

            THINKER_RET_CHECK(gruint_luna(input, hidden_in, &i2h_w_temp, &h2h_w_temp, &i2h_bias_temp, &h2h_bias_temp,
                          output, hidden_o, attr, workspace), "gruint_luna");
    }
    else {
        THINKER_RET_CHECK(gruint_luna(input, hidden_in, i2h_w, h2h_w, i2h_bias, h2h_bias,
                          output, hidden_o, attr, workspace), "gruint_luna");
    }
#endif

#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","GruInt", total_t);  
#endif
    
    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
