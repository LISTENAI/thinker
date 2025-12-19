// Quantization operator implementation

#undef __OP__
#define __OP__ Quant
#include <stdio.h>
#include <string.h>
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

/**
 * @brief Execute the quantization operation
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));  // Validate tensor count
    
    QuantAttrs *attr = (QuantAttrs *)((int8_t *)op + op->attr_offset_);
    tTensor *X = tensors[0];  // Input tensor
    tTensor *Y = tensors[op->num_input_];  // Output tensor

    // Validate data type
    int32_t data_bits = attr->data_bits;
    if (data_bits != 8 && data_bits != 16 && data_bits != 32) {
        return T_ERR_INVALID_DATATYPE;
    }

    if (X->dtype_ != Float32) {
        return T_ERR_INVALID_DATATYPE;
    }

    size_t size = getTensorSize(X);
    float *input = (float *)X->dptr_;  // Input data pointer
    int8_t *output = (int8_t *)Y->dptr_;  // Output data pointer
    int8_t scale = Y->scale_;  // Quantization scale

    quant(input, output, size, scale);  // Perform quantization

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__