#ifndef _SUM_LUNA_H_
#define _SUM_LUNA_H_

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "thinker_status.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief Integer Quantized Sum operation
 * @param input Input tensor
 * @param Temp Temporary workspace tensor
 * @param output Output tensor
 * @param attrs Sum attributes
 * @return int32_t Operation status
 */
int32_t iqsum_luna(tTensor *input, tTensor *Temp, tTensor *output, iqSumAttrs *attrs) 
{
    int32_t ret = T_SUCCESS;
    int32_t axis = attrs->axis;
    size_t size = getTensorSize(input);

    if (axis < 0)
        axis += input->shape_.ndim_;

    if (axis != (input->shape_.ndim_ - 1)) {
        return T_ERR_INVALID_PARA;
    }

    int32_t len = size / input->shape_.dims_[axis];
    int32_t shift = input->scale_ - output->scale_;

    for (int32_t i = 0; i < len; ++i) {
        ret |= API_LIB(vector_sum_i8o32)((const int8_t *)input->dptr_, 
                                       (int32_t *)Temp->dptr_, 
                                       input->shape_.dims_[axis], shift);
    }

    ret |= API_LIB(scale_i32i32o8)((const int32_t *)Temp->dptr_, 
                                   1, 
                                   (int8_t *)output->dptr_, 
                                   size, 0);

    return ret;
}

#endif