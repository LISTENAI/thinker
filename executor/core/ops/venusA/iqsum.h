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
    int32_t ret = T_SUCCESS;
    int32_t axis = attrs->axis;
    size_t size = getTensorSize(input);

    // Adjust axis for negative values
    if (axis < 0) {
        axis += input->shape_.ndim_;
    }

    // Ensure summation is performed along the last dimension
    if (axis != input->shape_.ndim_ - 1) {
        return T_ERR_INVALID_PARA;
    }

    // Calculate length and quantization shift
    int32_t len = size / input->shape_.dims_[axis];
    int32_t shift = input->scale_ - output->scale_;

    // Perform summation based on input and output data types
    switch (input->dtype_) {
        case Int8:
            switch (output->dtype_) {
                case Int8:
                    for (int32_t i = 0; i < len; i++) {
                        ret |= API_LIB(vector_sum_i8o8)((const int8_t *)input->dptr_, 
                                                        (int8_t *)output->dptr_, 
                                                        input->shape_.dims_[axis], shift);
                    }
                    break;
                case Int16:
                    for (int32_t i = 0; i < len; i++) {
                        ret |= API_LIB(vector_sum_i8o16)((const int8_t *)input->dptr_, 
                                                         (int16_t *)output->dptr_, 
                                                         input->shape_.dims_[axis], shift);
                    }
                    break;
                case Int32:
                    for (int32_t i = 0; i < len; i++) {
                        ret |= API_LIB(vector_sum_i8o32)((const int8_t *)input->dptr_, 
                                                         (int32_t *)output->dptr_, 
                                                         input->shape_.dims_[axis], shift);
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
                        ret |= API_LIB(vector_sum_i16o8)((const int16_t *)input->dptr_, 
                                                        (int8_t *)output->dptr_, 
                                                        input->shape_.dims_[axis], shift);
                    }
                    break;
                case Int16:
                    for (int32_t i = 0; i < len; i++) {
                        ret |= API_LIB(vector_sum_i16o16)((const int16_t *)input->dptr_, 
                                                         (int16_t *)output->dptr_, 
                                                         input->shape_.dims_[axis], shift);
                    }
                    break;
                case Int32:
                    for (int32_t i = 0; i < len; i++) {
                        ret |= API_LIB(vector_sum_i16o32)((const int16_t *)input->dptr_, 
                                                         (int32_t *)output->dptr_, 
                                                         input->shape_.dims_[axis], shift);
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
                        ret |= API_LIB(vector_sum_i32o8)((const int32_t *)input->dptr_, 
                                                        (int8_t *)output->dptr_, 
                                                        input->shape_.dims_[axis], shift);
                    }
                    break;
                case Int16:
                    for (int32_t i = 0; i < len; i++) {
                        ret |= API_LIB(vector_sum_i32o16)((const int32_t *)input->dptr_, 
                                                         (int16_t *)output->dptr_, 
                                                         input->shape_.dims_[axis], shift);
                    }
                    break;
                case Int32:
                    for (int32_t i = 0; i < len; i++) {
                        ret |= API_LIB(vector_sum_i32o32)((const int32_t *)input->dptr_, 
                                                         (int32_t *)output->dptr_, 
                                                         input->shape_.dims_[axis], shift);
                    }
                    break;
                case Int64:
                    for (int32_t i = 0; i < len; i++) {
                        ret |= API_LIB(vector_sum_i32o64)((const int32_t *)input->dptr_, 
                                                         (int64_t *)output->dptr_, 
                                                         input->shape_.dims_[axis], shift);
                    }
                    break;
                default:
                    return T_ERR_INVALID_DATATYPE;
            }
            break;
        default:
            return T_ERR_INVALID_DATATYPE;
    }

    return ret;
}

#endif  // _SUM_LUNA_H_