#undef __OP__
#define __OP__ Clip
#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/type_switch.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

/**
 * Clip operation macro - clips values between min and max bounds
 * @param x: Input array
 * @param y: Output array  
 * @param size: Array size
 * @param max: Maximum value
 * @param min: Minimum value
 */
#define CLIP(x, y, size, max, min)     \
  for (int32_t i = 0; i < size; i++) { \
    if (x[i] < min) {                  \
      y[i] = min;                      \
    } else if (x[i] > max) {           \
      y[i] = max;                      \
    } else {                           \
      y[i] = x[i];                     \
    }                                  \
  }

/**
 * Forward pass implementation for Clip operator
 * @param op: Operator structure
 * @param tensors: Input/output tensor array
 * @param num_tensor: Number of tensors
 * @param list: DMA list (unused)
 * @return: Status code
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Get input and output tensors
    tTensor *X = ((tTensor **)tensors)[0];
    tTensor *Y = ((tTensor **)tensors)[op->num_input_];
    
    // Get clip attributes
    ClipAttrs *attrs = (ClipAttrs *)((int8_t *)op + op->attr_offset_);
    
    // Get clip bounds (default values from attributes)
    float max = attrs->max;
    float min = attrs->min;
    
    // If additional input tensors are provided for min/max values
    if (op->num_input_ > 1) {
        tTensor *XMin = tensors[1];
        tTensor *XMax = tensors[2];
        
        // Switch based on data type to read min/max values
        DATA_TYPE_SWITCH_ALL(XMin->dtype_, Type, {
            max = *(Type *)XMax->dptr_;
            min = *(Type *)XMin->dptr_;
        });
    }

    // Calculate total number of elements
    int32_t size = getTensorSize(X);
    int16_t dtype = X->dtype_;
    
    // Process based on data type
    switch (dtype) {
        case Int8: {
            int8_t *input = (int8_t *)X->dptr_;
            int8_t *output = (int8_t *)Y->dptr_;
            CLIP(input, output, size, max, min);
        } break;
        
        case Int16: {
            int16_t *input = (int16_t *)X->dptr_;
            int16_t *output = (int16_t *)Y->dptr_;
            CLIP(input, output, size, max, min);
        } break;
        
        case Int32: {
            int32_t *input = (int32_t *)X->dptr_;
            int32_t *output = (int32_t *)Y->dptr_;
            CLIP(input, output, size, max, min);
        } break;
        
        case Float32: {
            float *input = (float *)X->dptr_;
            float *output = (float *)Y->dptr_;
            CLIP(input, output, size, max, min);
        } break;
        
        default:
            THINKER_LOG_FATAL("Unsupported data type for Clip operation!");
            return T_ERR_INVALID_DATATYPE;
    }
    
    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__