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

typedef void *luna_prelu_api_item;
// Function pointer type for PReLU operation
typedef int32_t (*luna_prelu_api)(const void *, uint32_t, void *, uint32_t size, uint32_t post_shift);

// Lookup table for different data type combinations
static luna_prelu_api_item luna_prelu_api_items[][2] = {
    {API_LIB(prelu_i8o8), API_LIB(prelu_i8o32)},
    {API_LIB(prelu_i32o8), API_LIB(prelu_i32o32)}
};

/**
 * @brief Performs PReLU computation based on input and output tensor types.
 *
 * @param X Input tensor
 * @param Y Output tensor
 * @param size Number of elements to process
 * @param slope Slope value for negative inputs
 * @param post_shift Right shift amount after computation
 * @return Status of the operation
 */
static int32_t calc_prelu(tTensor *X, tTensor *Y, uint32_t size, int32_t slope, int32_t post_shift) {
    int32_t in_idx = (X->dtype_ & 0xF) >> 1;       // Extract input data type index
    int32_t out_idx = (Y->dtype_ & 0xF) >> 1;      // Extract output data type index
    luna_prelu_api luna_prelu = (luna_prelu_api)(luna_prelu_api_items[in_idx][out_idx]); // Get appropriate function
    return luna_prelu((const void *)X->dptr_, slope, (void *)Y->dptr_, size, post_shift); // Execute PReLU
}

/**
 * @brief Main entry point for PReLU execution using Luna backend.
 *
 * @param X Input tensor
 * @param Y Output tensor
 * @param attrs Attributes including slope and post_shift
 * @return T_SUCCESS if successful
 */
tStatus prelu_luna(tTensor *X, tTensor *Y, PreluAttrs *attrs) {
    int32_t slope = attrs->slope;
    int32_t post_shift = attrs->post_shift;
    uint32_t size = getTensorSize(X);

    return calc_prelu(X, Y, size, slope, post_shift); // Perform PReLU calculation
}

#endif // __PRELU_H__