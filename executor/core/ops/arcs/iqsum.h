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
int32_t iqsum_luna(tTensor *input, tTensor *Temp, tTensor *output, iqSumAttrs *attrs) {
    int32_t axis = attrs->axis;
    size_t size = getTensorSize(input);

    if (axis < 0)
        axis += input->shape_.ndim_;

    #if THINKER_PARAM_CHECK
    if (axis != (input->shape_.ndim_ - 1)) {
        return (T_ERR_INVALID_PARA);
    }

    if (input->dtype_ != Int8 || output->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (input->mem_.type_ != 2) {
        return (T_ERR_NO_SUPPORT_OP);
    }
#endif

    int32_t axis_size = input->shape_.dims_[axis];
    int32_t len = size / axis_size;
    int32_t shift = input->scale_ - output->scale_;
    #if THINKER_PARAM_CHECK
    if (shift < 0 || shift > 63) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    #if THINKER_RUNTIME_CHECK
    if (Temp == NULL || Temp->dptr_ == 0 ||
                          Temp->mem_.type_ != 2 || output->mem_.type_ != 2 ||
                           getTensorDataSize(Temp) < (size_t)len * sizeof(int32_t)) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif

    for (int32_t i = 0; i < len; ++i) {
        THINKER_RET_CHECK(API_LIB(vector_sum_i8o32)((const int8_t *)input->dptr_ + i * axis_size,
                                       (int32_t *)Temp->dptr_ + i,
                                       axis_size, shift), "luna_vector_sum_i8o32");
    }

    THINKER_RET_CHECK(API_LIB(scale_i32i32o8)((const int32_t *)Temp->dptr_, 
                                   1, 
                                   (int8_t *)output->dptr_, 
                                   len, 0), "luna_scale_i32i32o8");

    return T_SUCCESS;
}

#endif
