#undef __OP__
#define __OP__ Cast
#include "thinker_status.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "core/comm/type_switch.h"

// Forward pass implementation for Cast operator
int32_t X(Forward)(tOperator *op, tTensor **tensors, int num_tensor, tDMA_List*list) {
    // Validate input tensor count
    CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));
    
    // Get cast attributes
    CastAttrs *attr = (CastAttrs *)((char *)op + op->attr_offset_);
    
    // Extract input and output tensors
    tTensor *X = tensors[0];        // Input tensor
    tTensor *Y = tensors[op->num_input_];  // Output tensor
    
    // Get total number of elements
    int32_t size = getTensorSize(X);
    
    // Copy data element by element with type conversion
    for (int i = 0; i < size; i++) {
        DATA_TYPE_SWITCH_ALL(X->dtype_, IType, {
            const IType *input = (IType *)(X->dptr_);  // Cast input pointer to source type
            DATA_TYPE_SWITCH_ALL(Y->dtype_, OType, {
                OType *output = (OType *)(Y->dptr_);   // Cast output pointer to target type
                output[i] = input[i];                  // Copy element
            });
        });
    }
    
    return T_SUCCESS;  // Return success status
}

#include "core/operator_template.h"
#undef __OP__