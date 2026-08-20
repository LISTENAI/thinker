#undef __OP__
#define __OP__ Cast
#include "thinker_status.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "core/comm/type_switch.h"

// Forward pass implementation for Cast operator
int32_t X(Forward)(tOperator *op, tTensor **tensors, int num_tensor, tDMA_List*list) {
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if (op->num_input_ != 1 || op->num_output_ != 1 ||
                        num_tensor != op->num_input_ + op->num_output_) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
    // Get cast attributes
    CastAttrs *attr = (CastAttrs *)((char *)op + op->attr_offset_);
    
    // Extract input and output tensors
    tTensor *X = tensors[0];        // Input tensor
    tTensor *Y = tensors[op->num_input_];  // Output tensor

#if THINKER_PARAM_CHECK
    if (X == NULL || Y == NULL || attr->to != Y->dtype_) {
        return (T_ERR_INVALID_PARA);
    }

    if ((X->dtype_ != Float32 && X->dtype_ != Int8 && X->dtype_ != Int16 &&
         X->dtype_ != Int32 && X->dtype_ != Int64 && X->dtype_ != Uint8 &&
         X->dtype_ != Uint16 && X->dtype_ != Uint32 && X->dtype_ != Uint64) ||
        (Y->dtype_ != Float32 && Y->dtype_ != Int8 && Y->dtype_ != Int16 &&
         Y->dtype_ != Int32 && Y->dtype_ != Int64 && Y->dtype_ != Uint8 &&
         Y->dtype_ != Uint16 && Y->dtype_ != Uint32 && Y->dtype_ != Uint64)) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (!equalShape(&X->shape_, &Y->shape_) || X->dptr_ == 0 ||
                          Y->dptr_ == 0 || X->dptr_ == Y->dptr_) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
    // Get total number of elements
    int32_t size = getTensorSize(X);
#if THINKER_RUNTIME_CHECK
    if (size <= 0 || getTensorSize(Y) != (size_t)size) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
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
