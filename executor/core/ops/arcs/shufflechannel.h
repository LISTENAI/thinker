#ifndef __SHUFFLECHANNEL_H__
#define __SHUFFLECHANNEL_H__

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_basic_math.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

#include "thinker_status.h"

/**
 * @brief Perform channel shuffle operation in NCHW format
 * @param X Input tensor (NCHW layout)
 * @param Y Output tensor (NCHW layout)
 * @param group Number of groups for shuffling
 */
static void channel_shuffle_nchw(const tTensor *X, tTensor *Y, int32_t group)
{
    int32_t batch_num = (int32_t)X->shape_.dims_[0];
    int32_t group_column = X->shape_.dims_[1] / group;
    int32_t height = X->shape_.dims_[2];
    int32_t width = X->shape_.dims_[3];
    int32_t stride = X->shape_.dims_[1] * height * width;
    int32_t length = height * width;
    int8_t *input_data = (int8_t *)X->dptr_;
    int8_t *output_data = (int8_t *)Y->dptr_;
    
    CHECK_EQ(group * group_column, X->shape_.dims_[1]);
    
    for (int32_t b = 0; b < batch_num; ++b) {
        int8_t *input_ptr = input_data + b * stride * X->byte_;
        int8_t *output_ptr = output_data + b * stride * X->byte_;
        for (int32_t i = 0; i < group; ++i) {
            for (int32_t j = 0; j < group_column; ++j) {
                int8_t *p_i = input_ptr + (i * group_column + j) * length * X->byte_;
                int8_t *p_o = output_ptr + (j * group + i) * length * X->byte_;
                API_LIB(memcpy_i8o8)(p_o, p_i, length * X->byte_);
            }
        }
    }
}

/**
 * @brief Perform channel shuffle operation in NHWC format
 * @param X Input tensor (NHWC layout)
 * @param Y Output tensor (NHWC layout)
 * @param group Number of groups for shuffling
 */
static void channel_shuffle_nhwc(const tTensor *X, tTensor *Y, int32_t group)
{
    int32_t batch_num = (int32_t)X->shape_.dims_[0];
    int32_t group_column = X->shape_.dims_[3] / group;
    int32_t height = X->shape_.dims_[1];
    int32_t width = X->shape_.dims_[2];
    int32_t stride = X->shape_.dims_[3] * height * width;
    int8_t *input_data = (int8_t *)X->dptr_;
    int8_t *output_data = (int8_t *)Y->dptr_;
    
    CHECK_EQ(group * group_column, X->shape_.dims_[3]);
    
    for (int32_t b = 0; b < batch_num; ++b) {
        int8_t *input_ptr = input_data + b * stride * X->byte_;
        int8_t *output_ptr = output_data + b * stride * X->byte_;
        for (int32_t h = 0; h < height; ++h) {
            for (int32_t w = 0; w < width; ++w) {
                int32_t offset = h * (width * group * group_column) + w * (group * group_column);
                int8_t *p_i = input_ptr + offset * X->byte_;
                int8_t *p_o = output_ptr + offset * X->byte_;
                for (int32_t i = 0; i < group; ++i) {
                    for (int32_t j = 0; j < group_column; ++j) {
                        int8_t *p_ii = p_i + (i * group_column + j) * X->byte_;
                        int8_t *p_oo = p_o + (j * group + i) * X->byte_;
                        API_LIB(memcpy_i8o8)(p_oo, p_ii, X->byte_);
                    }
                }
            }
        }
    }
}

/**
 * @brief Shuffle channel operation implementation
 * @param X Input tensor
 * @param Y Output tensor
 * @param attr Shuffle channel attributes
 * @return Operation result status
 */
int32_t shufflechannel_venus(tTensor *X, tTensor *Y, ShuffleChannelAttrs *attr) 
{
    CHECK_EQ(X->shape_.ndim_, 4);
    Y->shape_ = X->shape_;
    Y->dtype_ = X->dtype_;
    Y->scale_ = X->scale_;
    CHECK_GE(attr->num_group, 1);
    
    // Only support fast memory (type 2)
    if ((2 != X->mem_.type_) || (2 != Y->mem_.type_))
        return T_ERR_INVALID_PLATFROM;

    // Perform channel shuffle based on axis
    if (attr->axis == 1)  // NCHW format
    {
        channel_shuffle_nchw(X, Y, attr->num_group);
    } 
    else if (attr->axis == 3)  // NHWC format
    {
        channel_shuffle_nhwc(X, Y, attr->num_group);
    }
    else 
    {
        return T_ERR_INVALID_PARA;
    }

    return T_SUCCESS;
}

#endif