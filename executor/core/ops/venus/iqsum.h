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
    int32_t axis = attrs->axis;
    size_t size = getTensorSize(input);

    // Adjust axis for negative values
    if (axis < 0) {
        axis += input->shape_.ndim_;
    }

    // Only support summation along the last dimension
    #if THINKER_PARAM_CHECK
    if (axis != (input->shape_.ndim_ - 1)) {
        return (T_ERR_INVALID_PARA);
    }
    if (input->dtype_ != Int8 || output->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }
    #endif

    // Calculate dimension lengths
    int32_t len = size / input->shape_.dims_[axis];
    int32_t shift = input->scale_ - output->scale_;
    #if THINKER_PARAM_CHECK
    if (shift < 0 || shift > 63) {
        return (T_ERR_INVALID_PARA);
    }
    #endif
    #if THINKER_RUNTIME_CHECK
    if (Temp == NULL || Temp->dptr_ == 0 ||
                          getTensorDataSize(Temp) < (size_t)len * sizeof(int32_t)) {
        return (T_ERR_NO_WORKSPACE);
    }
    #endif

    // Perform summation and scaling
    for (int32_t i = 0; i < len; ++i) {
        THINKER_RET_CHECK(API_LIB(vector_sum_q7_int32)((const q7_t *)input->dptr_ + i * input->shape_.dims_[axis],
                                            (int32_t *)Temp->dptr_ + i,
                                            input->shape_.dims_[axis], 
                                            shift), "luna_vector_sum_q7_int32");
    }

    // Scale back to int8
    THINKER_RET_CHECK(API_LIB(scale_q31_int8)((const q31_t *)Temp->dptr_, 1, 
                                   (int8_t *)output->dptr_, 
                                   len, 0), "luna_scale_q31_int8");

    return T_SUCCESS;
}

#endif
