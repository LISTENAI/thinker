#ifndef _SUM_LUNA_H_
#define _SUM_LUNA_H_

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
 * @brief Quantized summation operation implementation
 * @param input Input tensor
 * @param Temp Temporary workspace tensor
 * @param output Output tensor
 * @param attrs Summation attributes
 * @return int32_t Operation status
 */
int32_t iqsum_luna(tTensor *input, tTensor *Temp, tTensor *output, iqSumAttrs *attrs) {
    int32_t axis = attrs->axis;

    // Adjust axis for negative values
    if (axis < 0) {
        axis += input->shape_.ndim_;
    }

#if THINKER_PARAM_CHECK
if (axis != input->shape_.ndim_ - 1) {
    return (T_ERR_INVALID_PARA);
}

    if (input->dtype_ != output->dtype_ ||
                        (input->dtype_ != Int8 && input->dtype_ != Int16 &&
                         input->dtype_ != Int32)) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (input->mem_.type_ != 2 || output->mem_.type_ != 2) {
        return (T_ERR_NO_SUPPORT_OP);
    }
#endif

    // Calculate length and quantization shift
    int32_t len = 1;
    for (int32_t i = 0; i < axis; ++i) {
        len *= input->shape_.dims_[i];
    }
    int32_t axis_size = input->shape_.dims_[axis];
    int32_t shift = input->scale_ - output->scale_;
#if THINKER_PARAM_CHECK
if (output->mem_.type_ != 2 || shift < 0 || shift > 63) {
    return (T_ERR_INVALID_PARA);
}
#endif

    // Perform summation based on input and output data types
    switch (input->dtype_) {
        case Int8:
            switch (output->dtype_) {
                case Int8:
                    for (int32_t i = 0; i < len; i++) {
                        THINKER_RET_CHECK(API_LIB(vector_sum_i8o8)((const int8_t *)input->dptr_ + i * axis_size,
                                                        (int8_t *)output->dptr_ + i,
                                                        axis_size, shift), "luna_vector_sum_i8o8");
                    }
                    break;
                case Int16:
                    for (int32_t i = 0; i < len; i++) {
                        THINKER_RET_CHECK(API_LIB(vector_sum_i8o16)((const int8_t *)input->dptr_ + i * axis_size,
                                                         (int16_t *)output->dptr_ + i,
                                                         axis_size, shift), "luna_vector_sum_i8o16");
                    }
                    break;
                case Int32:
                    for (int32_t i = 0; i < len; i++) {
                        THINKER_RET_CHECK(API_LIB(vector_sum_i8o32)((const int8_t *)input->dptr_ + i * axis_size,
                                                         (int32_t *)output->dptr_ + i,
                                                         axis_size, shift), "luna_vector_sum_i8o32");
                    }
                    break;
                default:
                    return T_ERR_INVALID_DATATYPE;
            }
            break;
        case Int16:
            switch (output->dtype_) {
                case Int8:
                    for (int32_t i = 0; i < len; i++) {
                        THINKER_RET_CHECK(API_LIB(vector_sum_i16o8)((const int16_t *)input->dptr_ + i * axis_size,
                                                        (int8_t *)output->dptr_ + i,
                                                        axis_size, shift), "luna_vector_sum_i16o8");
                    }
                    break;
                case Int16:
                    for (int32_t i = 0; i < len; i++) {
                        THINKER_RET_CHECK(API_LIB(vector_sum_i16o16)((const int16_t *)input->dptr_ + i * axis_size,
                                                         (int16_t *)output->dptr_ + i,
                                                         axis_size, shift), "luna_vector_sum_i16o16");
                    }
                    break;
                case Int32:
                    for (int32_t i = 0; i < len; i++) {
                        THINKER_RET_CHECK(API_LIB(vector_sum_i16o32)((const int16_t *)input->dptr_ + i * axis_size,
                                                         (int32_t *)output->dptr_ + i,
                                                         axis_size, shift), "luna_vector_sum_i16o32");
                    }
                    break;
                default:
                    return T_ERR_INVALID_DATATYPE;
            }
            break;
        case Int32:
            switch (output->dtype_) {
                case Int8:
                    for (int32_t i = 0; i < len; i++) {
                        THINKER_RET_CHECK(API_LIB(vector_sum_i32o8)((const int32_t *)input->dptr_ + i * axis_size,
                                                        (int8_t *)output->dptr_ + i,
                                                        axis_size, shift), "luna_vector_sum_i32o8");
                    }
                    break;
                case Int16:
                    for (int32_t i = 0; i < len; i++) {
                        THINKER_RET_CHECK(API_LIB(vector_sum_i32o16)((const int32_t *)input->dptr_ + i * axis_size,
                                                         (int16_t *)output->dptr_ + i,
                                                         axis_size, shift), "luna_vector_sum_i32o16");
                    }
                    break;
                case Int32:
                    for (int32_t i = 0; i < len; i++) {
                        THINKER_RET_CHECK(API_LIB(vector_sum_i32o32)((const int32_t *)input->dptr_ + i * axis_size,
                                                         (int32_t *)output->dptr_ + i,
                                                         axis_size, shift), "luna_vector_sum_i32o32");
                    }
                    break;
                case Int64:
                    for (int32_t i = 0; i < len; i++) {
                        THINKER_RET_CHECK(API_LIB(vector_sum_i32o64)((const int32_t *)input->dptr_ + i * axis_size,
                                                         (int64_t *)output->dptr_ + i,
                                                         axis_size, shift), "luna_vector_sum_i32o64");
                    }
                    break;
                default:
                    return T_ERR_INVALID_DATATYPE;
            }
            break;
        default:
            return T_ERR_INVALID_DATATYPE;
    }

    return T_SUCCESS;
}

#endif  // _SUM_LUNA_H_
