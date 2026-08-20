#undef __OP__
#define __OP__ AvgPool2dInt
#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

// Include platform-specific implementations
#ifdef THINKER_USE_VENUS
#include "./venus/avgpool2dint.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/avgpool2dint.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/avgpool2dint.h"
#endif

// Forward pass implementation for Average Pooling 2D Integer operator
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL) {
        return (T_ERR_INVALID_PARA);
    }
    if (op->num_input_ != 1 || op->num_output_ != 1 ||
                        num_tensor < 2) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
    // Get operator attributes
    PoolAttrs *attrs = (PoolAttrs *)((int8_t *)op + op->attr_offset_);
    
    // Get input and output tensors
    tTensor *X = ((tTensor **)tensors)[0];  // Input tensor
    tTensor *Y = ((tTensor **)tensors)[op->num_input_];  // Output tensor

    #if THINKER_PARAM_CHECK
    if (X == NULL || Y == NULL || X->shape_.ndim_ != 4 ||
                        Y->shape_.ndim_ != 4 || X->dtype_ != Int8 ||
                        Y->dtype_ != Int8 || attrs->layout != 0 || attrs->ceil != 0 ||
                        attrs->kernel[0] == 0 || attrs->kernel[1] == 0 ||
                        attrs->stride[0] == 0 || attrs->stride[1] == 0) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    int32_t expected_h = (X->shape_.dims_[2] + attrs->pad[0] + attrs->pad[2] -
                          attrs->kernel[0]) / attrs->stride[0] + 1;
    int32_t expected_w = (X->shape_.dims_[3] + attrs->pad[1] + attrs->pad[3] -
                          attrs->kernel[1]) / attrs->stride[1] + 1;
#if THINKER_RUNTIME_CHECK
    if (X->shape_.dims_[0] != 1 || Y->shape_.dims_[0] != 1 ||
                          X->shape_.dims_[1] <= 0 || X->shape_.dims_[2] <= 0 ||
                          X->shape_.dims_[3] <= 0 ||
                          X->shape_.dims_[2] + attrs->pad[0] + attrs->pad[2] <
                              attrs->kernel[0] ||
                          X->shape_.dims_[3] + attrs->pad[1] + attrs->pad[3] <
                              attrs->kernel[1] ||
                          expected_h <= 0 || expected_w <= 0 ||
                          Y->shape_.dims_[1] != X->shape_.dims_[1] ||
                          Y->shape_.dims_[2] != expected_h ||
                          Y->shape_.dims_[3] != expected_w || X->dptr_ == 0 ||
                          Y->dptr_ == 0 || X->dptr_ == Y->dptr_) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
    // Check if any platform is enabled
    #if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
        #if THINKER_PROFILE
        uint64_t start_t = tick_count();  // Record start time for profiling
        #endif
        
        // Check if workspace tensor is provided
#if THINKER_RUNTIME_CHECK
        if (num_tensor <= op->num_input_ + op->num_output_) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
        tTensor *workspace = tensors[op->num_input_ + op->num_output_];
#if THINKER_RUNTIME_CHECK
        if (workspace == NULL || workspace->dptr_ == 0 ||
                              workspace->mem_.type_ != 2 ||
                              workspace->shape_.ndim_ != 1) {
            return (T_ERR_NO_WORKSPACE);
        }
#endif
        THINKER_RET_CHECK(avgpool2dint_luna(X, Y, workspace, attrs), "avgpool2dint_luna");
        
        #if THINKER_PROFILE
        uint64_t finish_t = tick_count();  // Record end time for profiling
        uint32_t total_t = (uint32_t)(finish_t - start_t);
        printf("%8s | %u | (", "MeanPool2dInt", total_t); 
        #endif
    #else
        return T_ERR_NO_SUPPORT_OP;
    #endif

    return T_SUCCESS;  // Return result code
}

#include "core/operator_template.h"
#undef __OP__
