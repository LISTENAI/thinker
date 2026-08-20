#undef __OP__
#define __OP__ Dequant
#include <stdio.h>
#include <string.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"


/**
 * Forward pass implementation for Dequantization operator
 * Converts quantized integer tensors to floating-point tensors
 * @param op: Operator structure
 * @param tensors: Array of input/output tensors (input, output, optional workspace)
 * @param num_tensor: Total number of tensors
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    #if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if (op->num_input_ != 1 || op->num_output_ != 1 ||
                        num_tensor != op->num_input_ + op->num_output_) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
    // Get input and output tensors
    tTensor *X = ((tTensor **)tensors)[0];
    tTensor *Y = ((tTensor **)tensors)[op->num_input_];
#if THINKER_PARAM_CHECK
    if (X == NULL || Y == NULL || X->dptr_ == 0 || Y->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
    // Validate input data type
    if ((X->dtype_ != Int8) && (X->dtype_ != Uint8) && (X->dtype_ != Int32))
        return T_ERR_INVALID_DATATYPE;
#if THINKER_PARAM_CHECK
    if (Y->dtype_ != Float32 || X->zero_ != 0 || Y->zero_ != 0) {
        return (T_ERR_INVALID_PARA);
    }
#endif
#if THINKER_RUNTIME_CHECK
    if (!equalShape(&X->shape_, &Y->shape_) ||
                          X->dptr_ == Y->dptr_) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    size_t size = getTensorSize(X);
#if THINKER_RUNTIME_CHECK
    if (size == 0 || size > INT32_MAX ||
                          getTensorSize(Y) != size ||
                          getTensorDataSize(Y) < size * sizeof(float)) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    if (X->scale_ < 0.0f || X->scale_ > 30.0f ||
        X->scale_ != (float)(int32_t)X->scale_)
        return T_ERR_INVALID_PARA;
    DequantAttrs *attrs = (DequantAttrs *)((int8_t *)op + op->attr_offset_);
    if (attrs->scale_o != (uint8_t)(int32_t)X->scale_)
        return T_ERR_INVALID_PARA;
    int8_t scale = (int8_t)X->scale_;
    float *output = (float *)Y->dptr_;
    
    // Process based on input data type
    if (X->dtype_ == Int8) {
        int8_t *input = (int8_t *)X->dptr_;
        dequant8bit(input, output, size, scale);
    } 
    else if (X->dtype_ == Uint8) {
        uint8_t *input = (uint8_t *)X->dptr_;
        dequantU8bit(input, output, size, scale);
    } 
    else if (X->dtype_ == Int32) {
        int32_t *input = (int32_t *)X->dptr_;
        dequant32bit(input, output, size, scale);
    } 
    else {
        return T_ERR_INVALID_PARA;
    }
    
    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
