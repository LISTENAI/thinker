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
    // Validate input tensor count
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
    
    int32_t ret = T_ERR_NO_IMPLEMENTED;  // Default error return
    
    // Get operator attributes
    iqBinaryAttrs *attrs = (iqBinaryAttrs *)((int8_t *)op + op->attr_offset_);
    
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
        ret = bmmint_luna(X, Y, O, Workspace);
        
        #if THINKER_PROFILE
        uint64_t finish_t = tick_count();  // Record end time for profiling
        uint32_t total_t = (uint32_t)(finish_t - start_t);
        printf("%8s | %u | (","BmmInt", total_t);  
        #endif
    #endif

    return ret;  // Return result code
}

#include "core/operator_template.h"
#undef __OP__