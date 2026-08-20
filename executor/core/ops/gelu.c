// GELU operator implementation

#undef __OP__
#define __OP__ QGelu
#include "core/operator_register.h"
#include "thinker_status.h"
#include "core/comm/utils.h"

#ifdef THINKER_USE_VENUSA
#include "./venusA/gelu.h"  // VenusA backend implementation
#endif

/**
 * @brief Execute the GELU operation
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    (void)list;
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || op->num_input_ != 1 ||
                        op->num_output_ != 1 || num_tensor != 3) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    tTensor *X = tensors[0];
    tTensor *Y = tensors[1];
    tTensor *workspace = tensors[2];
#if THINKER_PARAM_CHECK
    if (X == NULL || Y == NULL || workspace == NULL) {
        return (T_ERR_INVALID_PARA);
    }

    if (X->shape_.ndim_ > 7 || !equalShape(&X->shape_, &Y->shape_) ||
                        X->layout_ != Y->layout_) {
        return (T_ERR_INVALID_DATA);
    }

    if ((X->dtype_ != Int8 && X->dtype_ != Int16 &&
                         X->dtype_ != Int32) || Y->dtype_ != Int8) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (X->zero_ != 0 || Y->zero_ != 0) {
        return (T_ERR_INVALID_PARA);
    }

    if (getTensorSize(X) == 0 || getTensorSize(X) > UINT32_MAX ||
                        getTensorSize(Y) != getTensorSize(X)) {
        return (T_ERR_INVALID_DATA);
    }

    if (X->dptr_ == 0 || Y->dptr_ == 0 || workspace->dptr_ == 0) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    
#if THINKER_PROFILE
    uint64_t start_t = tick_count();
#endif

#ifdef THINKER_USE_VENUSA
    THINKER_RET_CHECK(gelu_luna(X, Y, workspace), "gelu_luna");
#else
    return T_ERR_NO_IMPLEMENTED;
#endif
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","gelu", total_t);  
#endif
    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
