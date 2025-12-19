#ifndef _MAXPOOL_LUNA_H_
#define _MAXPOOL_LUNA_H_

#include <stdio.h>
#include <string.h>

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#include "luna/luna_cnn_tools.h"
#define API_LIB(api) luna_##api
#endif

#include "core/operator_attrs.h"
#include "c_api/thinker_define.h"

/**
 * @brief Initialize max pooling parameters
 * @param attrs Pooling attributes
 * @param conv_attrs Pointer to convolution structure
 * @param X Input tensor
 * @param Y Output tensor
 */
static void luna_maxpool_para_init(PoolAttrs* attrs, conv_struct_t *conv_attrs, tTensor *X, tTensor *Y) {
    memset(conv_attrs, 0, sizeof(conv_struct_t));

    conv_attrs->input_c = X->shape_.dims_[1];  // Number of input channels
    conv_attrs->input_h = X->shape_.dims_[2];  // Input height
    conv_attrs->input_w = X->shape_.dims_[3];  // Input width
    conv_attrs->output_c = Y->shape_.dims_[1]; // Number of output channels
    conv_attrs->output_h = Y->shape_.dims_[2]; // Output height
    conv_attrs->output_w = Y->shape_.dims_[3]; // Output width
    conv_attrs->weight_h = attrs->kernel[0];   // Kernel height
    conv_attrs->weight_w = attrs->kernel[1];   // Kernel width
    conv_attrs->stride_h = attrs->stride[0];   // Stride height
    conv_attrs->stride_w = attrs->stride[1];   // Stride width
    conv_attrs->padding_h_up = attrs->pad[0];  // Top padding
    conv_attrs->padding_h_down = attrs->pad[2]; // Bottom padding
    conv_attrs->padding_w_left = attrs->pad[1]; // Left padding
    conv_attrs->padding_w_right = attrs->pad[3]; // Right padding
    conv_attrs->dilation_h = 1;               // Dilation height
    conv_attrs->dilation_w = 1;               // Dilation width
    conv_attrs->data_mem_type = X->mem_.type_; // Memory type
    conv_attrs->ou_bits = Y->byte_ * 8;       // Output bits
    conv_attrs->weight_bits = 8;              // Weight bits
    conv_attrs->out_padding_h = 0;            // Output padding height
    conv_attrs->out_padding_w = 0;            // Output padding width
    conv_attrs->group = 1;                    // Number of groups
    conv_attrs->positive_shift_type = ShiftType_FloorX05; // Positive shift type
    conv_attrs->positive_shift_value = 0;     // Positive shift value
    conv_attrs->negative_shift_type = ShiftType_FloorX05; // Negative shift type
    conv_attrs->negative_shift_value = 0;     // Negative shift value
    conv_attrs->activation_type = NO_ACTIVE;   // Activation type
    conv_attrs->is_bias = 0;                  // Bias flag
}

/**
 * @brief Perform max pooling operation
 * @param X Input tensor
 * @param Y Output tensor
 * @param Temp Temporary buffer
 * @param attrs Pooling attributes
 * @return Execution status
 */
int32_t maxpool_luna(const tTensor* X, tTensor* Y, tTensor* Temp, PoolAttrs *attrs) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    // Check if input data type is Int8
    if (Int8 != X->dtype_) {
        return T_ERR_INVALID_DATATYPE;
    }

    // Initialize pooling parameters
    conv_struct_t pool_struct_;
    luna_cnn_static_para_t conv_static_para;
    luna_maxpool_para_init(attrs, &pool_struct_, (tTensor *)X, Y);

    // Get dimensions
    int32_t batch = X->shape_.dims_[0];
    int32_t in_c = pool_struct_.input_c;
    int32_t in_h = pool_struct_.input_h;
    int32_t in_w = pool_struct_.input_w;
    int32_t ou_c = pool_struct_.output_c;
    int32_t ou_h = pool_struct_.output_h;
    int32_t ou_w = pool_struct_.output_w;

    // Calculate space sizes
    int32_t in_space = in_h * in_w * X->byte_;
    int32_t out_space = ou_h * ou_w * Y->byte_;
    int32_t out_size = out_space * ou_c;

    // Calculate batch sizes
    int32_t in_batch_size = in_c * in_h * in_w;
    int32_t ou_batch_size = ou_c * ou_h * ou_w * (Y->dtype_ & 0xF);

    // Process data based on memory type
    if (Y->mem_.type_ == 2) {
        ret = luna_split_conv_para_pack(&pool_struct_, &conv_static_para, LUNA_MAX_POOLING);
        if (ret != T_SUCCESS) return ret;

        // Process each batch
        for (int32_t n = 0; n < batch; n++) {
            int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
            int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
            ret = API_LIB(max_pooling2d_i8o8)(p_in, p_out, &conv_static_para);
        }
    } else {
        // Calculate workspace size
        int32_t workspace_size = Temp ? Temp->shape_.dims_[0] : 0;
        int32_t max_ch_per_run = workspace_size / out_space;
        if (max_ch_per_run == 0) {
            return T_ERR_NO_WORKSPACE;
        }

        // Process each batch and channel
        for (int32_t n = 0; n < batch; n++) {
            int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
            int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;

            for (int32_t ch_start = 0; ch_start < ou_c;) {
                int32_t remain = ou_c - ch_start;
                int32_t ch_cur = remain < max_ch_per_run ? remain : max_ch_per_run;
                pool_struct_.input_c = ch_cur;
                pool_struct_.output_c = ch_cur;

                ret = luna_split_conv_para_pack(&pool_struct_, &conv_static_para, LUNA_MAX_POOLING);
                if (ret != T_SUCCESS) return ret;

                ret = API_LIB(max_pooling2d_i8o8)(p_in + ch_start * in_space, (int8_t *)Temp->dptr_, &conv_static_para);
                opi_psram_cpy_out(p_out + ch_start * out_space, (int8_t *)Temp->dptr_, ch_cur * out_space * Y->byte_);
                ch_start += ch_cur;
            }
        }
    }

    return ret;
}

#endif