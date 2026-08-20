// Requant operator implementation

#undef __OP__
#define __OP__ Requant
#include <math.h>
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/requant.h"  // Venus backend implementation
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/requant.h"   // Arcs backend implementation
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/requant.h" // VenusA backend implementation
#endif

/**
 * @brief Execute the Requant operation
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || op->num_input_ != 1 ||
                        op->num_output_ != 1 || num_tensor != 2) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    RequantAttrs *attrs = (RequantAttrs *)((int8_t *)op + op->attr_offset_);
    tTensor *X = tensors[0];
    tTensor *Y = tensors[1];
#if THINKER_PARAM_CHECK
    if (X == NULL || Y == NULL || X->dptr_ == 0 || Y->dptr_ == 0 ||
                        !equalShape(&X->shape_, &Y->shape_)) {
        return (T_ERR_INVALID_DATA);
    }

    if (attrs->data_bits != 8 || attrs->o_bits != Y->byte_ * 8 ||
                        X->byte_ != 1 || attrs->quant_type != 1 ||
                        !isfinite(X->scale_) || !isfinite(Y->scale_) ||
                        floorf(X->scale_) != X->scale_ || floorf(Y->scale_) != Y->scale_) {
        return (T_ERR_INVALID_PARA);
    }
#endif

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
    THINKER_RET_CHECK(requant_luna(X, Y), "requant_luna");
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
