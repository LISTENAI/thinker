#undef __OP__
#define __OP__ BmmInt
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

// Include platform-specific implementations
#ifdef THINKER_USE_VENUS
#include "./venus/bmmint.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/bmmint.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/bmmint.h"
#endif

// Forward pass implementation for Batch Matrix Multiplication Integer operator
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if (op->num_input_ != 2 || op->num_output_ != 1 ||
                        num_tensor < op->num_input_ + op->num_output_) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
    // Extract input tensors
    tTensor *X = tensors[0];                    // First input tensor
    tTensor *Y = tensors[1];                    // Second input tensor
    tTensor *O = tensors[op->num_input_];       // Output tensor
    
    // Get workspace tensor if available
    tTensor *Workspace = NULL;
    if (num_tensor > op->num_input_ + op->num_output_)
        Workspace = tensors[op->num_input_ + op->num_output_];
    
    // Check if any platform is enabled
    #if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
        #if THINKER_PROFILE
        uint64_t start_t = tick_count();  // Record start time for profiling
        #endif
        
        // Call platform-specific BMM integer implementation
        THINKER_RET_CHECK(bmmint_luna(X, Y, O, Workspace), "bmmint_luna");
        
        #if THINKER_PROFILE
        uint64_t finish_t = tick_count();  // Record end time for profiling
        uint32_t total_t = (uint32_t)(finish_t - start_t);
        printf("%8s | %u | (","BmmInt", total_t);  
        #endif
    #else
        return T_ERR_NO_SUPPORT_OP;
    #endif

    return T_SUCCESS;  // Return result code
}

#include "core/operator_template.h"
#undef __OP__
