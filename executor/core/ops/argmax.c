#undef __OP__
#define __OP__ ArgMax
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

// Include platform-specific implementations
#ifdef THINKER_USE_VENUS
#include "./venus/argmax.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/argmax.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/argmax.h"
#endif

// Forward pass implementation for ArgMax operator
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if (op->num_input_ != 1 || op->num_output_ != 1 || num_tensor < 2) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
    // Get operator attributes
    ArgMaxAttrs *attrs = (ArgMaxAttrs *)((int8_t *)op + op->attr_offset_);
    tTensor *X = tensors[0];
    tTensor *Y = tensors[op->num_input_];

#if THINKER_PARAM_CHECK
    if (X == NULL || Y == NULL || X->shape_.ndim_ == 0 ||
                        Y->shape_.ndim_ != X->shape_.ndim_ ||
                        Y->dtype_ != Int32 || (attrs->axis != -1 &&
                         attrs->axis != (int32_t)X->shape_.ndim_ - 1)) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    int32_t leading = 1;
    for (int32_t i = 0; i < (int32_t)X->shape_.ndim_ - 1; ++i) {
        leading *= X->shape_.dims_[i];
    }
#if THINKER_RUNTIME_CHECK
    if (X->shape_.dims_[X->shape_.ndim_ - 1] <= 0 ||
                          (X->shape_.ndim_ > 1 && X->shape_.dims_[0] != 1) ||
                          Y->shape_.dims_[0] != 2 || X->dptr_ == 0 ||
                          Y->dptr_ == 0 || X->dptr_ == Y->dptr_) {
        return (T_ERR_INVALID_PARA);
    }
    for (int32_t i = 1; i < (int32_t)X->shape_.ndim_ - 1; ++i) {
        if (X->shape_.dims_[i] <= 0 ||
                              Y->shape_.dims_[i] != X->shape_.dims_[i]) {
            return (T_ERR_INVALID_PARA);
        }
    }

    if ((X->shape_.ndim_ > 1 &&
                           Y->shape_.dims_[Y->shape_.ndim_ - 1] != 1) ||
                          getTensorSize(Y) != (size_t)leading * 2) {
        return (T_ERR_INVALID_PARA);
    }
#endif
       
    // Check if any platform is enabled
    #if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
        #if THINKER_PROFILE
        uint64_t start_t = tick_count();  // Record start time for profiling
        #endif
        
#if THINKER_RUNTIME_CHECK
        if (num_tensor <= op->num_input_ + op->num_output_) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
        tTensor *workspace = tensors[op->num_input_ + op->num_output_];
#if THINKER_RUNTIME_CHECK
        if (workspace == NULL || workspace->dptr_ == 0 ||
                              workspace->mem_.type_ != 2 ||
                              workspace->shape_.ndim_ != 1 ||
                              workspace->shape_.dims_[0] < 8 ||
                              workspace->dptr_ == X->dptr_ ||
                              workspace->dptr_ == Y->dptr_) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
        THINKER_RET_CHECK(argmax_luna(X, Y, workspace, attrs), "argmax_luna");
        
        #if THINKER_PROFILE
        uint64_t finish_t = tick_count();  // Record end time for profiling
        uint32_t total_t = (uint32_t)(finish_t - start_t);
        printf("%8s | %u | (","ArgMax", total_t);  
        #endif
    #else
        return T_ERR_NO_SUPPORT_OP;
    #endif

    return T_SUCCESS;  // Return result code
}

#include "core/operator_template.h"
#undef __OP__
