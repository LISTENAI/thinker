#ifndef __PRELU_H__
#define __PRELU_H__

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
 * @brief Calculate PReLU operation
 * @param X Input tensor
 * @param Y Output tensor
 * @param size Number of elements to process
 * @param slope PReLU slope parameter
 * @param post_shift Post scaling shift value
 * @return Execution status
 */
static int32_t calc_prelu(tTensor *X, tTensor *Y, uint32_t size, int32_t slope, int32_t post_shift) {
    int32_t ret = T_ERR_NO_IMPLEMENTED;

    // Determine data types and call corresponding PReLU function
    switch (X->dtype_) {
        case Int8:
            switch (Y->dtype_) {
                case Int8:
                    ret = API_LIB(prelu_i8o8)((int8_t *)X->dptr_, slope, (int8_t *)Y->dptr_, size, post_shift);
                    break;
                case Int16:
                    ret = API_LIB(prelu_i8o16)((int8_t *)X->dptr_, slope, (int16_t *)Y->dptr_, size, post_shift);
                    break;
                case Int32:
                    ret = API_LIB(prelu_i8o32)((int8_t *)X->dptr_, slope, (int32_t *)Y->dptr_, size, post_shift);
                    break;
                default:
                    return T_ERR_INVALID_DATATYPE;
            }
            break;
        case Int16:
            switch (Y->dtype_) {
                case Int8:
                    ret = API_LIB(prelu_i16o8)((int16_t *)X->dptr_, slope, (int8_t *)Y->dptr_, size, post_shift);
                    break;
                case Int16:
                    ret = API_LIB(prelu_i16o16)((int16_t *)X->dptr_, slope, (int16_t *)Y->dptr_, size, post_shift);
                    break;
                case Int32:
                    ret = API_LIB(prelu_i16o32)((int16_t *)X->dptr_, slope, (int32_t *)Y->dptr_, size, post_shift);
                    break;
                default:
                    return T_ERR_INVALID_DATATYPE;
            }
            break;
        case Int32:
            switch (Y->dtype_) {
                case Int8:
                    ret = API_LIB(prelu_i32o8)((int32_t *)X->dptr_, slope, (int8_t *)Y->dptr_, size, post_shift);
                    break;
                case Int16:
                    ret = API_LIB(prelu_i32o16)((int32_t *)X->dptr_, slope, (int16_t *)Y->dptr_, size, post_shift);
                    break;
                case Int32:
                    ret = API_LIB(prelu_i32o32)((int32_t *)X->dptr_, slope, (int32_t *)Y->dptr_, size, post_shift);
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

/**
 * @brief Perform PReLU operation
 * @param X Input tensor
 * @param Y Output tensor
 * @param attrs PReLU attributes containing slope and post scaling shift
 * @return Execution status
 */
tStatus prelu_luna(tTensor *X, tTensor *Y, PreluAttrs *attrs) {
    tStatus status = T_ERR_FAIL;

    // Check if input and output are in PSRAM
    if (X->mem_.type_ != 2 || Y->mem_.type_ != 2) {
        return T_ERR_NO_IMPLEMENTED;
    }

    // Get PReLU parameters
    int32_t slope = attrs->slope;
    int32_t post_shift = attrs->post_shift;
    uint32_t size = getTensorSize(X);

    // Execute PReLU calculation
    status = (tStatus)calc_prelu(X, Y, size, slope, post_shift);

    return T_SUCCESS;
}

#endif