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
    // Validate input tensor count
    CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
    
    // Get operator attributes
    PoolAttrs *attrs = (PoolAttrs *)((int8_t *)op + op->attr_offset_);
    
    // Get input and output tensors
    tTensor *X = ((tTensor **)tensors)[0];  // Input tensor
    tTensor *Y = ((tTensor **)tensors)[op->num_input_];  // Output tensor
    
    int32_t ret = T_ERR_NO_IMPLEMENTED;  // Default error return
    
    // Check if any platform is enabled
    #if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
        #if THINKER_PROFILE
        uint64_t start_t = tick_count();  // Record start time for profiling
        #endif
        
        // Check if workspace tensor is provided
        if (num_tensor > ((op->num_input_ + op->num_output_))) {
            tTensor *workspace = ((tTensor **)tensors)[num_tensor - 1];  // Workspace tensor
            ret = avgpool2dint_luna(X, Y, workspace, attrs);  // Call platform-specific implementation
        }
        
        #if THINKER_PROFILE
        uint64_t finish_t = tick_count();  // Record end time for profiling
        uint32_t total_t = (uint32_t)(finish_t - start_t);
        printf("%8s | %u | (", "MeanPool2dInt", total_t); 
        #endif
    #endif

    return ret;  // Return result code
}

#include "core/operator_template.h"
#undef __OP__