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
 * @brief PReLU API function pointer type
 */
typedef void *luna_prelu_api_item;
typedef int32_t (*luna_prelu_api)(const void *, int32_t, void *, uint32_t, int32_t);

/**
 * @brief PReLU API function table for different data types
 */
static luna_prelu_api_item luna_prelu_api_items[3][3] = {
    {
        API_LIB(prelu_q7_int8),  // Int8 input -> Int8 output
        API_LIB(prelu_q7_int16), // Int8 input -> Int16 output
        API_LIB(prelu_q7_int32)  // Int8 input -> Int32 output
    },
    {
        API_LIB(prelu_q15_int8),  // Int16 input -> Int8 output
        API_LIB(prelu_q15_int16), // Int16 input -> Int16 output
        API_LIB(prelu_q15_int32)  // Int16 input -> Int32 output
    },
    {
        API_LIB(prelu_q31_int8),  // Int32 input -> Int8 output
        API_LIB(prelu_q31_int16), // Int32 input -> Int16 output
        API_LIB(prelu_q31_int32)  // Int32 input -> Int32 output
    }
};

/**
 * @brief Calculate PReLU activation
 * @param X Input tensor
 * @param Y Output tensor
 * @param size Size of input data
 * @param slope PReLU slope parameter
 * @param post_shift Post-shift value
 * @return int32_t Operation status
 */
static int32_t calc_prelu(tTensor *X, tTensor *Y, uint32_t size, int32_t slope, int32_t post_shift) {
    int32_t in_idx = (X->dtype_ & 0xF) >> 1;  // Get input data type index
    int32_t out_idx = (Y->dtype_ & 0xF) >> 1; // Get output data type index
    luna_prelu_api luna_prelu = (luna_prelu_api)(luna_prelu_api_items[in_idx][out_idx]); // Select appropriate API

    return luna_prelu((const void *)X->dptr_, slope, (void *)Y->dptr_, size, post_shift);
}

/**
 * @brief Main PReLU function
 * @param X Input tensor
 * @param Y Output tensor
 * @param attrs PReLU attributes
 * @return tStatus Operation status
 */
tStatus prelu_luna(tTensor *X, tTensor *Y, PreluAttrs *attrs) {
    int32_t slope = attrs->slope;
    int32_t post_shift = attrs->post_shift;
    uint32_t size = getTensorSize(X);
    return calc_prelu(X, Y, size, slope, post_shift);
}

#endif