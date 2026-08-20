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
    int32_t axis = attrs->axis < 0 ? X->shape_.ndim_ + attrs->axis : attrs->axis;
    #if THINKER_PARAM_CHECK
    if (axis != X->shape_.ndim_ - 1) {
        return (T_ERR_INVALID_PARA);
    }
    if (X->dtype_ != Int8 || Y->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }
    if (X->mem_.type_ != 2 || Y->mem_.type_ != 2) {
        return (T_ERR_INVALID_PLATFROM);
    }
    #endif
    #if THINKER_RUNTIME_CHECK
    if (workspace == NULL || workspace->dptr_ == 0 ||
                          workspace->mem_.type_ != 2 ||
                          workspace->dtype_ != Int8 || workspace->byte_ != 1) {
        return (T_ERR_NO_WORKSPACE);
    }
    #endif

    // Get output tensor size
    uint32_t size = getTensorSize(Y);
    #if THINKER_PARAM_CHECK
    if (size == 0 || getTensorSize(X) != (size_t)size * 2 ||
                        size != X->shape_.dims_[axis] / 2) {
        return (T_ERR_INVALID_DATA);
    }
    #endif
    #if THINKER_RUNTIME_CHECK
    if (getTensorDataSize(workspace) < (size_t)size * 3) {
        return (T_ERR_NO_WORKSPACE);
    }
    #endif

    // Pointers to input and output data
    int8_t *srcA = (int8_t *)X->dptr_;
    int8_t *srcB = srcA + size;
    int8_t *dst = (int8_t *)Y->dptr_;
    int16_t *tmp = (int16_t *)workspace->dptr_;
    int8_t *gate = (int8_t *)(tmp + size);

    // Quantization parameters
    int32_t x_q = (int32_t)X->scale_;
    int32_t y_q = (int32_t)Y->scale_;
    #if THINKER_PARAM_CHECK
    if (!isfinite(X->scale_) || !isfinite(Y->scale_) ||
                        floorf(X->scale_) != X->scale_ ||
                        floorf(Y->scale_) != Y->scale_) {
        return (T_ERR_INVALID_PARA);
    }
    #endif
    int32_t delt_q = 11 - x_q;  // Venus sigmoid expects Q11 int16 input.
    int32_t mul_shift = 7 + x_q - y_q;
    #if THINKER_PARAM_CHECK
    if (delt_q < -63 || delt_q > 6 ||
                        mul_shift < 0 || mul_shift > 63) {
        return (T_ERR_INVALID_PARA);
    }
    #endif
    int32_t scale = delt_q > 0 ? (1 << delt_q) : 1;
    int32_t shift = delt_q < 0 ? -delt_q : 0;

    // Perform GLU operation:
    // Scale the gate to Q11, evaluate sigmoid in Q7, then apply it to A.
    THINKER_RET_CHECK(API_LIB(scale_q7_int16)(srcB, scale, tmp, size, shift), "luna_scale_q7_int16");
    THINKER_RET_CHECK(API_LIB(sigmoid_int8)(tmp, gate, size), "luna_sigmoid_int8");
    THINKER_RET_CHECK(API_LIB(mul_q7_int8)(srcA, gate, dst, size, mul_shift), "luna_mul_q7_int8");

    return T_SUCCESS;
}

#endif  // _GLU_LUNA_H_
