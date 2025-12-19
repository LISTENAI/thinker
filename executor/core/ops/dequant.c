#undef __OP__
#define __OP__ Dequant
#include <stdio.h>
#include <string.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_register.h"


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
    // Validate tensor count
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
    
    // Get input and output tensors
    tTensor *X = ((tTensor **)tensors)[0];
    tTensor *Y = ((tTensor **)tensors)[op->num_input_];
    
    // Get workspace tensor if present
    tTensor *workspace = NULL;
    if (num_tensor > op->num_input_ + op->num_output_) {
        workspace = tensors[num_tensor - 1];
    }
    
    int32_t ret = T_ERR_NO_IMPLEMENTED;
    
    // Validate input data type
    if ((X->dtype_ != Int8) && (X->dtype_ != Uint8) && (X->dtype_ != Int32))
        return T_ERR_INVALID_DATATYPE;
    
    // Calculate tensor size and get output pointer
    size_t size = getTensorSize(X);
    int8_t scale = X->scale_;
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
    else if (X->dptr_ == Int32) {  // Note: This seems to be a bug - should check X->dtype_
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