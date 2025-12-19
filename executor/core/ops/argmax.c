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
    // Validate input tensor count
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
    
    // Get operator attributes
    ArgMaxAttrs *attrs = (ArgMaxAttrs *)((int8_t *)op + op->attr_offset_);
    
    int32_t ret = T_ERR_NO_IMPLEMENTED;  // Default error return
    
    // Check if any platform is enabled
    #if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
        #if THINKER_PROFILE
        uint64_t start_t = tick_count();  // Record start time for profiling
        #endif
        
        // Call platform-specific ArgMax implementation
        ret = argmax_luna(tensors[0], tensors[op->num_input_], tensors[num_tensor - 1], attrs);
        
        #if THINKER_PROFILE
        uint64_t finish_t = tick_count();  // Record end time for profiling
        uint32_t total_t = (uint32_t)(finish_t - start_t);
        printf("%8s | %u | (","ArgMax", total_t);  
        #endif
    #endif

    return ret;  // Return result code
}

#include "core/operator_template.h"
#undef __OP__