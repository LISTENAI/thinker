#undef __OP__
#define __OP__ Clip
#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
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

static int32_t clip_bound_to_int(float bound, int32_t min, int32_t max) {
    if (bound <= (float)min) return min;
    if (bound >= (float)max) return max;
    return (int32_t)bound;
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
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if ((op->num_input_ != 1 && op->num_input_ != 3) ||
                        op->num_output_ != 1 ||
                        num_tensor < op->num_input_ + op->num_output_) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    // Get input and output tensors
    tTensor *X = ((tTensor **)tensors)[0];
    tTensor *Y = ((tTensor **)tensors)[op->num_input_];
    
    // Get clip attributes
    ClipAttrs *attrs = (ClipAttrs *)((int8_t *)op + op->attr_offset_);
#if THINKER_PARAM_CHECK
    if (op->num_input_ == 1 &&
                        (attrs->min != attrs->min || attrs->max != attrs->max ||
                         attrs->min > attrs->max)) {
        return (T_ERR_INVALID_PARA);
    }

    if (X == NULL || Y == NULL ||
                        (X->dtype_ != Int8 && X->dtype_ != Int16 &&
                         X->dtype_ != Int32 && X->dtype_ != Float32) ||
                        Y->dtype_ != X->dtype_) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (Y->scale_ != X->scale_) {
        return (T_ERR_INVALID_PARA);
    }

    if (!equalShape(&X->shape_, &Y->shape_) ||
                          X->dptr_ == 0 || Y->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    tTensor *XMin = NULL;
    tTensor *XMax = NULL;
    if (op->num_input_ == 3) {
        XMin = tensors[1];
        XMax = tensors[2];
#if THINKER_PARAM_CHECK
        if (XMin == NULL || XMax == NULL ||
                            XMin->dtype_ != X->dtype_ ||
                            XMax->dtype_ != X->dtype_) {
            return (T_ERR_INVALID_DATATYPE);
        }
#endif
#if THINKER_RUNTIME_CHECK
        if (XMin->dptr_ == 0 || XMax->dptr_ == 0 ||
                              getTensorSize(XMin) != 1 ||
                              getTensorSize(XMax) != 1) {
            return (T_ERR_INVALID_PARA);
        }
#endif
    }

    // Calculate total number of elements
    size_t tensor_size = getTensorSize(X);
#if THINKER_RUNTIME_CHECK
    if (tensor_size == 0 || tensor_size > INT32_MAX ||
                          getTensorSize(Y) != tensor_size) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    int32_t size = (int32_t)tensor_size;
    int16_t dtype = X->dtype_;
    
    // Process based on data type
    switch (dtype) {
        case Int8: {
            int8_t *input = (int8_t *)X->dptr_;
            int8_t *output = (int8_t *)Y->dptr_;
            int8_t min = XMin ? *(int8_t *)XMin->dptr_ :
                (int8_t)clip_bound_to_int(attrs->min, INT8_MIN, INT8_MAX);
            int8_t max = XMax ? *(int8_t *)XMax->dptr_ :
                (int8_t)clip_bound_to_int(attrs->max, INT8_MIN, INT8_MAX);
#if THINKER_PARAM_CHECK
            if (!XMin && min > max) {
                return (T_ERR_INVALID_PARA);
            }
#endif
#if THINKER_RUNTIME_CHECK
            if (XMin && min > max) {
                return (T_ERR_INVALID_PARA);
            }
#endif
            CLIP(input, output, size, max, min);
        } break;
        
        case Int16: {
            int16_t *input = (int16_t *)X->dptr_;
            int16_t *output = (int16_t *)Y->dptr_;
            int16_t min = XMin ? *(int16_t *)XMin->dptr_ :
                (int16_t)clip_bound_to_int(attrs->min, INT16_MIN, INT16_MAX);
            int16_t max = XMax ? *(int16_t *)XMax->dptr_ :
                (int16_t)clip_bound_to_int(attrs->max, INT16_MIN, INT16_MAX);
#if THINKER_PARAM_CHECK
            if (!XMin && min > max) {
                return (T_ERR_INVALID_PARA);
            }
#endif
#if THINKER_RUNTIME_CHECK
            if (XMin && min > max) {
                return (T_ERR_INVALID_PARA);
            }
#endif
            CLIP(input, output, size, max, min);
        } break;
        
        case Int32: {
            int32_t *input = (int32_t *)X->dptr_;
            int32_t *output = (int32_t *)Y->dptr_;
            int32_t min = XMin ? *(int32_t *)XMin->dptr_ :
                clip_bound_to_int(attrs->min, INT32_MIN, INT32_MAX);
            int32_t max = XMax ? *(int32_t *)XMax->dptr_ :
                clip_bound_to_int(attrs->max, INT32_MIN, INT32_MAX);
#if THINKER_PARAM_CHECK
            if (!XMin && min > max) {
                return (T_ERR_INVALID_PARA);
            }
#endif
#if THINKER_RUNTIME_CHECK
            if (XMin && min > max) {
                return (T_ERR_INVALID_PARA);
            }
#endif
            CLIP(input, output, size, max, min);
        } break;
        
        case Float32: {
            float *input = (float *)X->dptr_;
            float *output = (float *)Y->dptr_;
            float min = XMin ? *(float *)XMin->dptr_ : attrs->min;
            float max = XMax ? *(float *)XMax->dptr_ : attrs->max;
#if THINKER_PARAM_CHECK
            if (!XMin && (min != min || max != max || min > max)) {
                return (T_ERR_INVALID_PARA);
            }
#endif
#if THINKER_RUNTIME_CHECK
            if (XMin &&
                                  (min != min || max != max || min > max)) {
                return (T_ERR_INVALID_PARA);
            }
#endif
            CLIP(input, output, size, max, min);
        } break;
        
        default:
#if THINKER_PARAM_CHECK
            if (1) {
                return (T_ERR_INVALID_DATATYPE);
            }
#endif
            break;
    }
    
    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
