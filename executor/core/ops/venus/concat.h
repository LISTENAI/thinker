#ifndef _CONCAT_VENUS_H_
#define _CONCAT_VENUS_H_

#include <string.h>
#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

#include "thinker_status.h"

/**
 * @brief Scale and quantize data from input to output
 * @param src Input data pointer
 * @param dst Output data pointer
 * @param size Number of elements to process
 * @param scale Scale factor for quantization
 */
static void scale_quant(int8_t *src, int8_t *dst, int32_t size, int8_t scale) {
    API_LIB(scale_q7_int8)(src, (int)(1 << scale), dst, size, 0);
}

/**
 * @brief Scale and dequantize data from input to output
 * @param src Input data pointer
 * @param dst Output data pointer
 * @param size Number of elements to process
 * @param scale Scale factor for dequantization
 */
static void scale_dequant8bit(int8_t *src, int8_t *dst, int32_t size, int8_t scale) {
    API_LIB(scale_q7_int8)(src, 1, dst, size, scale);
}

/**
 * @brief Concatenate multiple tensors along a specified axis
 * @param tensors Array of input tensors to concatenate
 * @param axis Axis along which to concatenate
 * @param input_num Number of input tensors
 * @param workspace Workspace buffer for temporary data
 * @param output Output tensor after concatenation
 * @return Operation status
 */
int32_t concat_luna(tTensor **tensors, int32_t axis, int32_t input_num,
                    tTensor *workspace, tTensor *output) {
    int32_t leading = 1, mid = output->shape_.dims_[axis], trailing = 1;
    int8_t *data_temp = (workspace == NULL) ? NULL : (int8_t *)workspace->dptr_;

    switch (tensors[0]->dtype_) {
        case Int8: {
            for (int32_t i = 0; i < axis; ++i) {
                leading *= output->shape_.dims_[i];
            }
            for (int32_t i = axis + 1; i < output->shape_.ndim_; ++i) {
                trailing *= output->shape_.dims_[i];
            }

            if (leading == 1) {
                int8_t *output_ptr = (int8_t *)(output->dptr_);
                for (int32_t i = 0; i < input_num; ++i) {
                    int8_t *src = (int8_t *)tensors[i]->dptr_;
                    int32_t input_scale = tensors[i]->scale_;
                    int32_t hw_curr = tensors[i]->shape_.dims_[axis] * trailing * output->byte_;

                    if (0 == hw_curr)
                        continue;

                    if (input_scale != output->scale_) {
                        if (2 != tensors[i]->mem_.type_) {
                            memcpy(data_temp, src, hw_curr);
                            src = data_temp;
                        }
                        if (2 != output->mem_.type_) {
                            output_ptr = data_temp + hw_curr * (2 == tensors[i]->mem_.type_);
                        }

                        if (input_scale < output->scale_) {
                            int32_t sub_scale = output->scale_ - input_scale;
                            scale_quant(src, output_ptr, hw_curr, sub_scale);
                        } else {
                            int32_t sub_scale = input_scale - output->scale_;
                            scale_dequant8bit(src, output_ptr, hw_curr, sub_scale);
                        }

                        if (2 != output->mem_.type_) {
                            memcpy((int8_t *)output->dptr_, output_ptr, hw_curr);
                        }
                    } else {
                        if ((2 == tensors[i]->mem_.type_) && (2 == output->mem_.type_))
                            API_LIB(memcpy)(output_ptr, src, hw_curr);
                        else
                            memcpy(output_ptr, src, hw_curr);
                    }

                    output_ptr += hw_curr;
                }
            } else {
                int32_t hw = mid * trailing * output->byte_;
                for (int32_t l = 0; l < leading; l++) {
                    int8_t *output_ptr = (int8_t *)(output->dptr_) + l * hw;
                    for (int32_t i = 0; i < input_num; ++i) {
                        int32_t hw_curr = tensors[i]->shape_.dims_[axis] * trailing * output->byte_;
                        int8_t *indptr_curr = (int8_t *)(tensors[i]->dptr_) + l * hw_curr;
                        int32_t input_scale = tensors[i]->scale_;

                        if (0 == hw_curr)
                            continue;

                        if (input_scale != output->scale_) {
                            if (2 != tensors[i]->mem_.type_) {
                                memcpy(data_temp, indptr_curr, hw_curr);
                                indptr_curr = data_temp;
                            }
                            if (2 != output->mem_.type_) {
                                output_ptr = data_temp + hw_curr * (2 == tensors[i]->mem_.type_);
                            }

                            if (input_scale < output->scale_) {
                                int32_t sub_scale = output->scale_ - input_scale;
                                scale_quant(indptr_curr, output_ptr, hw_curr, sub_scale);
                            } else {
                                int32_t sub_scale = input_scale - output->scale_;
                                scale_dequant8bit(indptr_curr, output_ptr, hw_curr, sub_scale);
                            }

                            if (2 != output->mem_.type_) {
                                memcpy((int8_t *)output->dptr_ + l * hw + i * hw_curr, output_ptr, hw_curr);
                            }
                        } else {
                            if ((2 == tensors[i]->mem_.type_) && (2 == output->mem_.type_))
                                API_LIB(memcpy)(output_ptr, indptr_curr, hw_curr);
                            else
                                memcpy(output_ptr, indptr_curr, hw_curr);
                        }
                        output_ptr += hw_curr;
                    }
                }
            }
            break;
        }

        case Int16: {
            for (int32_t i = 0; i < axis; ++i) {
                leading *= output->shape_.dims_[i];
            }
            for (int32_t i = axis + 1; i < output->shape_.ndim_; ++i) {
                trailing *= output->shape_.dims_[i];
            }

            if (leading == 1) {
                int16_t *output_ptr = (int16_t *)(output->dptr_);
                for (int32_t i = 0; i < input_num; ++i) {
                    int32_t hw_curr = tensors[i]->shape_.dims_[axis] * trailing;
                    if (0 == hw_curr)
                        continue;
                    memcpy(output_ptr, (int16_t *)(tensors[i]->dptr_), hw_curr * output->byte_);
                    output_ptr += hw_curr;
                }
            } else {
                int32_t hw = mid * trailing;
                for (int32_t l = 0; l < leading; l++) {
                    int16_t *output_ptr = (int16_t *)(output->dptr_) + l * hw;
                    for (int32_t i = 0; i < input_num; ++i) {
                        int32_t hw_curr = tensors[i]->shape_.dims_[axis] * trailing;
                        int16_t *indptr_curr = (int16_t *)(tensors[i]->dptr_) + l * hw_curr;
                        if (0 == hw_curr)
                            continue;
                        memcpy(output_ptr, indptr_curr, hw_curr * output->byte_);
                        output_ptr += hw_curr;
                    }
                }
            }
            break;
        }

        case Int32: {
            for (int32_t i = 0; i < axis; ++i) {
                leading *= output->shape_.dims_[i];
            }
            for (int32_t i = axis + 1; i < output->shape_.ndim_; ++i) {
                trailing *= output->shape_.dims_[i];
            }

            if (leading == 1) {
                int32_t *output_ptr = (int32_t *)(output->dptr_);
                for (int32_t i = 0; i < input_num; ++i) {
                    int32_t hw_curr = tensors[i]->shape_.dims_[axis] * trailing;
                    if (0 == hw_curr)
                        continue;
                    memcpy(output_ptr, (int32_t *)(tensors[i]->dptr_), hw_curr * output->byte_);
                    output_ptr += hw_curr;
                }
            } else {
                int32_t hw = mid * trailing;
                for (int32_t l = 0; l < leading; l++) {
                    int32_t *output_ptr = (int32_t *)(output->dptr_) + l * hw;
                    for (int32_t i = 0; i < input_num; ++i) {
                        int32_t hw_curr = tensors[i]->shape_.dims_[axis] * trailing;
                        int32_t *indptr_curr = (int32_t *)(tensors[i]->dptr_) + l * hw_curr;
                        if (0 == hw_curr)
                            continue;
                        memcpy(output_ptr, indptr_curr, hw_curr * output->byte_);
                        output_ptr += hw_curr;
                    }
                }
            }
            break;
        }

        default:
            break;
    }

    return 0;
}

#endif  //_CONCAT_VENUS_H_