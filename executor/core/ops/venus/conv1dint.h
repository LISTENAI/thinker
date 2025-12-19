#ifndef _CONV1DINT_VENUS_H_
#define _CONV1DINT_VENUS_H_

#include <math.h>
#include <stdint.h>
#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

#include "thinker_status.h"

/**
 * @brief Matrix multiplication function pointer type for Conv1D
 * @param src1 Pointer to first input matrix
 * @param src2 Pointer to second input matrix
 * @param dst Pointer to output matrix
 * @param row Number of rows in first matrix
 * @param col Number of columns in first matrix (and rows in second matrix)
 * @param col2 Number of columns in second matrix
 * @param shift Bit shift value for scaling
 * @return Operation status
 */
typedef int32_t (*CONV1D_MAT_MUL_LUNA_API)(int8_t *src1, int8_t *src2, void *dst,
                                        int32_t row, int32_t col, int32_t col2,
                                        int32_t shift);

/**
 * @brief Split matrix multiplication function pointer type for Conv1D
 * @param src1 Pointer to first input matrix
 * @param src2 Pointer to second input matrix
 * @param dst Pointer to output matrix
 * @param split_num Number of splits
 * @param row Number of rows in first matrix
 * @param col Number of columns in first matrix (and rows in second matrix)
 * @param col2 Number of columns in second matrix
 * @param shift Bit shift value for scaling
 * @return Operation status
 */
typedef int32_t (*CONV1D_SPLIT_MAT_MUL_LUNA_API)(int8_t *src1, int8_t *src2,
                                              void *dst, int32_t split_num,
                                              int32_t row, int32_t col,
                                              int32_t col2, int32_t shift);

/**
 * @brief Vector addition function pointer type for Conv1D
 * @param src1 Pointer to first input vector
 * @param src2 Pointer to second input vector
 * @param dst Pointer to output vector
 * @param size Number of elements to process
 * @param shift Bit shift value for scaling
 * @return Operation status
 */
typedef int32_t (*CONV1D_VEC_ADD_LUNA_API)(void *src1, void *src2, void *dst,
                                        int32_t size, int32_t shift);

/**
 * @brief Structure storing Conv1D operation APIs
 */
struct conv1d_luna_api_item {
    void *luna_api;
};

/**
 * @brief List of Conv1D operation APIs for different data types
 */
static struct conv1d_luna_api_item conv1d_luna_api_list[][3] = {
    {{API_LIB(mat_mul_q7_int8)}, {API_LIB(mat_mul_q7_int16)}, {API_LIB(mat_mul_q7_int32)}},
    {{API_LIB(split_mat_mul_q7_int8)},
     {API_LIB(split_mat_mul_q7_int16)},
     {API_LIB(split_mat_mul_q7_int32)}},
    {{API_LIB(add_q7_int8)}, {API_LIB(add_q7_int16)}, {API_LIB(add_q7_int32)}},
    {{API_LIB(add_q15_int8)}, {API_LIB(add_q15_int16)}, {API_LIB(add_q15_int32)}},
    {{API_LIB(add_q31_int8)}, {API_LIB(add_q31_int16)}, {API_LIB(add_q31_int32)}}
};

/**
 * @brief Calculate the ceiling of a division using bit shifting
 * @param x Numerator
 * @param shift Bit shift amount (equivalent to dividing by 2^shift)
 * @return Ceiling of x divided by 2^shift
 */
static int32_t luna_quant_ceil(int32_t x, int32_t shift) {
    if (x & ~(0xFFFFFFFF << shift)) {
        return (x >> shift) + 1;
    } else {
        return (x >> shift);
    }
}

/**
 * @brief Convert image data to column format for Conv1D
 * @param src Input image data pointer
 * @param dst Output column data pointer
 * @param channel Number of channels in the input
 * @param height Height of the input
 * @param kernel Kernel size
 * @param stride Stride size
 * @return Operation status
 */
static int32_t img2col(int8_t *src, int8_t *dst, int32_t channel, int32_t height, int32_t kernel, int32_t stride) {
    int32_t ret = API_LIB(mat_trans_q7)(src, src, channel, height);
    int32_t num = (height - kernel) / stride + 1;
    int32_t data_size = channel * kernel;
    int32_t step_size = channel * stride;
    for (int32_t i = 0; i < num; i++) {
        memcpy(dst + i * data_size, src + i * step_size, data_size);
    }
    return ret;
}

/**
 * @brief Calculate Conv1D with different data types
 * @param w_dtype Weight data type
 * @param y_dtype Output data type
 * @param input Input data pointer
 * @param weight Weight data pointer
 * @param bias Bias data pointer (optional)
 * @param output Output data pointer
 * @param conv_attrs Convolution attributes
 * @return Operation status
 */
static int32_t calc_conv_luna(int32_t w_dtype, int32_t y_dtype, int8_t *input,
                              int8_t *weight, int32_t *bias, void *output,
                              s_conv_struct *conv_attrs) {
    int32_t ret = 0;
    switch (w_dtype) {
        case Int4:
            switch (y_dtype) {
                case Int8:
                    ret = API_LIB(conv_intx_int8)((const int8_t *)input, (int8_t *)weight,
                                                  (int32_t *)bias, (int8_t *)output,
                                                  conv_attrs, 4);
                    break;
                case Int16:
                    ret = API_LIB(conv_intx_int16)((const int8_t *)input, (int8_t *)weight,
                                                   (int32_t *)bias, (int16_t *)output,
                                                   conv_attrs, 4);
                    break;
                case Int32:
                    ret = API_LIB(conv_intx_int32)((const int8_t *)input, (int8_t *)weight,
                                                   (int32_t *)bias, (int32_t *)output,
                                                   conv_attrs, 4);
                    break;
            }
            break;
        case Int8:
            switch (y_dtype) {
                case Int8:
                    ret = API_LIB(conv_q7_int8)((const int8_t *)input, (int8_t *)weight,
                                                (int32_t *)bias, (int8_t *)output, conv_attrs);
                    break;
                case Int16:
                    ret = API_LIB(conv_q7_int16)((const int8_t *)input, (int8_t *)weight,
                                                 (int32_t *)bias, (int16_t *)output, conv_attrs);
                    break;
                case Int32:
                    ret = API_LIB(conv_q7_int32)((const int8_t *)input, (int8_t *)weight,
                                                 (int32_t *)bias, (int32_t *)output, conv_attrs);
                    break;
            }
            break;
    }
    return ret;
}

/**
 * @brief Calculate depthwise Conv1D with different data types
 * @param w_dtype Weight data type
 * @param y_dtype Output data type
 * @param input Input data pointer
 * @param weight Weight data pointer
 * @param bias Bias data pointer (optional)
 * @param output Output data pointer
 * @param conv_attrs Convolution attributes
 * @return Operation status
 */
static int32_t calc_depthwise_luna(int32_t w_dtype, int32_t y_dtype,
                                   int8_t *input, int8_t *weight, int32_t *bias,
                                   void *output, s_conv_struct *conv_attrs) {
    int32_t ret = 0;
    switch (w_dtype) {
        case Int4:
            switch (y_dtype) {
                case Int8:
                    ret = API_LIB(depthwise_conv_intx_int8)((const int8_t *)input,
                                                            (int8_t *)weight, (int32_t *)bias,
                                                            (int8_t *)output, conv_attrs, 4);
                    break;
                case Int16:
                    ret = API_LIB(depthwise_conv_intx_int16)((const int8_t *)input,
                                                             (int8_t *)weight, (int32_t *)bias,
                                                             (int16_t *)output, conv_attrs, 4);
                    break;
                case Int32:
                    ret = API_LIB(depthwise_conv_intx_int32)((const int8_t *)input,
                                                             (int8_t *)weight, (int32_t *)bias,
                                                             (int32_t *)output, conv_attrs, 4);
                    break;
            }
            break;
        case Int8:
            switch (y_dtype) {
                case Int8:
                    ret = API_LIB(depthwise_conv_q7_int8)((const int8_t *)input,
                                                          (int8_t *)weight, (int32_t *)bias,
                                                          (int8_t *)output, conv_attrs);
                    break;
                case Int16:
                    ret = API_LIB(depthwise_conv_q7_int16)((const int8_t *)input,
                                                           (int8_t *)weight, (int32_t *)bias,
                                                           (int16_t *)output, conv_attrs);
                    break;
                case Int32:
                    ret = API_LIB(depthwise_conv_q7_int32)((const int8_t *)input,
                                                           (int8_t *)weight, (int32_t *)bias,
                                                           (int32_t *)output, conv_attrs);
                    break;
            }
            break;
    }
    return ret;
}

/**
 * @brief Calculate split Conv1D with different data types
 * @param w_dtype Weight data type
 * @param y_dtype Output data type
 * @param input Input data pointer
 * @param weight Weight data pointer
 * @param bias Bias data pointer (optional)
 * @param output Output data pointer
 * @param conv_attrs Convolution attributes
 * @return Operation status
 */
static int32_t calc_split_cnn_luna(int32_t w_dtype, int32_t y_dtype,
                                   int8_t *input, int8_t *weight, int32_t *bias,
                                   void *output, s_conv_struct *conv_attrs) {
    int32_t ret = 0;
    switch (w_dtype) {
        case Int8:
            switch (y_dtype) {
                case Int8:
                    ret = API_LIB(conv_split_q7_int8)((const int8_t *)input, (int8_t *)weight,
                                                      (int32_t *)bias, (int8_t *)output,
                                                      conv_attrs);
                    break;
                case Int16:
                    ret = API_LIB(conv_split_q7_int16)((const int8_t *)input,
                                                       (int8_t *)weight, (int32_t *)bias,
                                                       (int16_t *)output, conv_attrs);
                    break;
                case Int32:
                    ret = API_LIB(conv_split_q7_int32)((const int8_t *)input,
                                                       (int8_t *)weight, (int32_t *)bias,
                                                       (int32_t *)output, conv_attrs);
                    break;
            }
            break;
    }
    return ret;
}

/**
 * @brief Initialize Conv1D attributes
 * @param attrs Conv1D attributes
 * @param conv_attrs Convolution attributes
 * @param X Input tensor
 * @param W Weight tensor
 * @param Bias Bias tensor (optional)
 * @param Y Output tensor
 */
static void conv1dint_para_init(Conv1dIntAttrs *attrs,
                                s_conv_struct *conv_attrs, tTensor *X,
                                tTensor *W, tTensor *Bias, tTensor *Y) {
    memset(conv_attrs, 0, sizeof(s_conv_struct));
    if (NULL != Bias) {
        conv_attrs->is_bias = 1;
    }
    conv_attrs->input_c = X->shape_.dims_[1];
    conv_attrs->input_h = X->shape_.dims_[2];
    conv_attrs->input_w = 1;
    conv_attrs->output_c = Y->shape_.dims_[1];
    conv_attrs->output_h = Y->shape_.dims_[2];
    conv_attrs->output_w = 1;
    conv_attrs->weight_h = attrs->kernel;
    conv_attrs->weight_w = 1;
    conv_attrs->stride_h = attrs->stride;
    conv_attrs->stride_w = 1;
    conv_attrs->padding_h_up = attrs->pad[0];
    conv_attrs->padding_h_down = attrs->pad[1];
    conv_attrs->padding_w_left = 0;
    conv_attrs->padding_w_right = 0;
    conv_attrs->input_h_after_padding = conv_attrs->input_h +
                                        conv_attrs->padding_h_up +
                                        conv_attrs->padding_h_down;
    conv_attrs->input_w_after_padding = conv_attrs->input_w +
                                        conv_attrs->padding_w_left +
                                        conv_attrs->padding_w_right;
    if (1 == attrs->act_type) {
        conv_attrs->activation_type = RELU;
    } else if (2 == attrs->act_type) {
        conv_attrs->activation_type = PRELU;
    } else {
        conv_attrs->activation_type = NO_ACTIVE;
    }
    int32_t q_x = (int32_t)X->scale_;
    int32_t q_w = (int32_t)W->scale_;
    int32_t q_y = (int32_t)Y->scale_;
    conv_attrs->positive_shift_type = ShiftType_FloorX05;
    conv_attrs->positive_shift_value = q_x + q_w - q_y;
    conv_attrs->negative_shift_type = ShiftType_FloorX05;
    conv_attrs->negative_shift_value = conv_attrs->positive_shift_value;
    conv_attrs->batch_num = attrs->group;
}

/**
 * @brief Perform Conv1D operation with quantized integer tensors
 * @param X Input tensor
 * @param W Weight tensor
 * @param Bias Bias tensor (optional)
 * @param Y Output tensor
 * @param Temp Temporary workspace tensor
 * @param attrs Conv1D attributes
 * @return Operation status
 */
int32_t conv1dint_luna(tTensor *X, tTensor *W, tTensor *Bias, tTensor *Y,
                       tTensor *Temp, Conv1dIntAttrs *attrs) {
    int32_t ret = T_ERR_FAIL;
    uint64_t paddr_b = 0;
    int32_t has_bias = 0;
    int32_t bias_idx = 0;
    int32_t ou_idx = (Y->dtype_ & 0xF) >> 1;
    if (Bias != NULL) {
        bias_idx = (Bias->dtype_ & 0xF) >> 1;
        paddr_b = Bias->dptr_;
        has_bias = 1;
    }
    int32_t kernel = attrs->kernel;
    if (kernel < 6) {
        s_conv_struct conv_attrs;
        conv1dint_para_init(attrs, &conv_attrs, X, W, Bias, Y);
        // Additional implementation details for kernel < 6
    }
    return ret;
}

#endif  //_CONV1DINT_VENUS_H_