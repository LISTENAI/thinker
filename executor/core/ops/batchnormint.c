#undef __OP__
#define __OP__ BatchNorm2dInt
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

// Include platform-specific implementations
#ifdef THINKER_USE_VENUS
#include "./venus/batchnormint.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/batchnormint.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/batchnormint.h"
#endif

// Forward pass implementation for Batch Normalization Integer operator
int32_t X(Forward)(tOperator* op, tTensor** tensors, int32_t num_tensor, tDMA_List* list) {
    // Validate input tensor count
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
    
    int32_t ret = T_ERR_NO_IMPLEMENTED;  // Default error return
    
    // Extract input tensors
    tTensor* X = ((tTensor**)tensors)[0];     // Input tensor
    tTensor* W = ((tTensor**)tensors)[1];     // Weight tensor
    tTensor* Bias = ((tTensor**)tensors)[2];  // Bias tensor
    tTensor* Y = ((tTensor**)tensors)[op->num_input_];  // Output tensor
    
    // Check if any platform is enabled
    #if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
        #if THINKER_PROFILE
        uint64_t start_t = tick_count();  // Record start time for profiling
        #endif
        
        // Get workspace tensor and call platform-specific implementation
        tTensor* workspace = ((tTensor**)tensors)[num_tensor - 1];
        ret = batchnormint_luna(X, W, Bias, Y, workspace);
        
        #if THINKER_PROFILE
        uint64_t finish_t = tick_count();  // Record end time for profiling
        uint32_t total_t = (uint32_t)(finish_t - start_t);
        printf("%8s | %u | (","batchNormInt", total_t);  
        #endif
    #endif

    return ret;  // Return result code
}

#include "core/operator_template.h"
#undef __OP__