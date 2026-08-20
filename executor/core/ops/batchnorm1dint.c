#undef __OP__
#define __OP__ BatchNorm1dInt
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_VENUS
#include "./venus/batchnorm1dint.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/batchnorm1dint.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/batchnorm1dint.h"
#endif

// Forward pass implementation for Batch Normalization Integer operator
int32_t X(Forward)(tOperator* op, tTensor** tensors, int32_t num_tensor, tDMA_List* list) {
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if (op->num_input_ != 3 || op->num_output_ != 1 ||
                        num_tensor < op->num_input_ + op->num_output_) {
        return (T_ERR_INVALID_PARA);
    }
#endif
   
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
#if THINKER_RUNTIME_CHECK
    if (num_tensor <= op->num_input_ + op->num_output_) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif
    tTensor *Workspace = tensors[op->num_input_ + op->num_output_];
    THINKER_RET_CHECK(batchnorm1dint_luna(X, W, Bias, Y, Workspace), "batchnormint_luna");

    #if THINKER_PROFILE
    uint64_t finish_t = tick_count();  // Record end time for profiling
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","BatchNorm1dInt", total_t);  
    #endif
#else
    return T_ERR_NO_SUPPORT_OP;
#endif

    return T_SUCCESS;  // Return result code
}

#include "core/operator_template.h"
#undef __OP__
