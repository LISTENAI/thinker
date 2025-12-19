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
 * @brief Integer quantized sum operation
 * @param input Input tensor
 * @param Temp Temporary tensor (if needed)
 * @param output Output tensor
 * @param attrs Sum operation attributes
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

    // Only support summation along the last dimension
    if (axis != (input->shape_.ndim_ - 1)) {
        return T_ERR_INVALID_PARA;
    }

    // Calculate dimension lengths
    int32_t len = size / input->shape_.dims_[axis];
    int32_t shift = input->scale_ - output->scale_;

    // Perform summation and scaling
    for (int32_t i = 0; i < len; ++i) {
        ret |= API_LIB(vector_sum_q7_int32)((const q7_t *)input->dptr_, 
                                            (int32_t *)Temp->dptr_, 
                                            input->shape_.dims_[axis], 
                                            shift);
    }

    // Scale back to int8
    ret |= API_LIB(scale_q31_int8)((const q31_t *)Temp->dptr_, 1, 
                                   (int8_t *)output->dptr_, 
                                   size, 0);

    return ret;
}

#endif