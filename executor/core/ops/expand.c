#undef __OP__
#define __OP__ Expand
#include <string.h>
#include "thinker_status.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "core/comm/utils.h"
#include "core/comm/type_switch.h"

#ifdef THINKER_USE_ARCS
#include "arcs/luna/opi_psram_cpy.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/luna/luna_misc_math.h"
#endif

/**
 * Forward pass implementation for Expand operator
 * Expands input tensor to match target shape by repeating elements
 * @param op: Operator structure containing expansion attributes
 * @param tensors: Array of input/output tensors (input, output, optional workspace)
 * @param num_tensor: Total number of tensors (must be 3)
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    // Validate tensor count
    CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));
    if (num_tensor != 3) 
        return T_ERR_INVALID_PARA;

    // Get input and output tensors
    tTensor *X = (tTensor *)tensors[0];
    tTensor *Y = (tTensor *)tensors[op->num_input_];

    // Get shape information
    int32_t xdim = X->shape_.ndim_;
    int32_t ydim = Y->shape_.ndim_;
    const uint32_t *tShape = X->shape_.dims_;
    const uint32_t *yshape = Y->shape_.dims_;

    // Calculate leading dimension multiplier
    int32_t bl = ydim - xdim;
    int32_t leading = 1;
    for (int32_t i = 0; i < bl; ++i) {
        leading *= yshape[i];
    }

    // Calculate expanded size and shape
    int32_t size = 1;
    uint32_t expandshape[7];
    for (int32_t i = bl; i < ydim; ++i) {
        size *= yshape[i];
        expandshape[i - bl] = yshape[i];
    }

    // Process data based on data type
    DATA_TYPE_SWITCH_ALL(X->dtype_, Type, {
        const Type *input = (Type *)X->dptr_;
        Type *output = (Type *)Y->dptr_;
        int32_t ndim = xdim;
        int32_t input_accumu[7];
        int32_t output_accumu[7];
        
        // Calculate accumulation factors for indexing
        input_accumu[ndim - 1] = output_accumu[ndim - 1] = 1;
        for (int32_t i = ndim - 1; i > 0; i--) {
            input_accumu[i - 1] = input_accumu[i] * tShape[i];
            output_accumu[i - 1] = output_accumu[i] * expandshape[i];
        }
        
        // Copy data element by element
        for (int32_t i = 0; i < size; ++i) {
            int32_t inputIdx = 0;
            int32_t i_ = i;
            for (int32_t j = 0; j < ndim; ++j) {
                int32_t outIdx = i_ / output_accumu[j];
                inputIdx += (outIdx % tShape[j]) * input_accumu[j];
                i_ %= output_accumu[j];
            }
            output[i] = input[inputIdx];
        }
        
        // Copy expanded regions
        for (int32_t i = 1; i < leading; ++i)
#if THINKER_USE_VENUS
            memcpy(output + i * size * sizeof(Type), output, size * sizeof(Type));
#elif THINKER_USE_ARCS || THINKER_USE_VENUSA
            opi_psram_cpy_out(output + i * size * sizeof(Type), output, size * sizeof(Type));
#endif
    });

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__