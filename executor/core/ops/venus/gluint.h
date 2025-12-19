#ifndef _GLU_LUNA_H_
#define _GLU_LUNA_H_

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/type_switch.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "thinker_status.h"
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_basic_math.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

/**
 * @brief Perform Gated Linear Unit (GLU) operation with quantized inputs and outputs
 * @param X Pointer to input tensor (contains concatenated input and gate)
 * @param Y Pointer to output tensor
 * @param workspace Pointer to workspace tensor for intermediate results
 * @param attrs Pointer to GluIntAttrs containing operation attributes
 * @return int32_t Return status (T_ERR_NO_IMPLEMENTED if not implemented, T_ERR_INVALID_PARA for invalid parameters, T_SUCCESS if successful)
 */
int32_t gluint_luna(tTensor *X, tTensor *Y, tTensor *workspace, GluIntAttrs *attrs) {
    // Check if axis is valid (negative axis not supported)
    if (attrs->axis >= 0) {
        return T_ERR_INVALID_PARA;
    }

    // Get output tensor size
    uint32_t size = getTensorSize(Y);

    // Pointers to input and output data
    int8_t *srcA = (int8_t *)X->dptr_;
    int8_t *srcB = srcA + size;
    int8_t *dst = (int8_t *)Y->dptr_ + size;
    int32_t *tmp = (int32_t *)workspace->dptr_;

    // Quantization parameters
    int32_t x_q = (int32_t)X->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    int32_t delt_q = 27 - x_q;  // Quantization delta for sigmoid input

    // Perform GLU operation:
    // 1. Scale gate (srcB) to int32
    // 2. Apply quantization delta
    // 3. Compute sigmoid
    // 4. Multiply with input (srcA) and scale to output
    int32_t ret = API_LIB(scale_i8i8o32)(srcB, 1, tmp, size, 0);
    ret |= API_LIB(scale_i32i32o32)(tmp, (1 << delt_q), tmp, size, 0);
    ret |= API_LIB(sigmoid_i32o8)(tmp, dst, size);
    ret |= API_LIB(mul_i8i8o8)(srcA, dst, dst, size, (7 + x_q - y_q));

    return ret;
}

#endif  // _GLU_LUNA_H_