// SoftmaxInt operator implementation

#undef __OP__
#define __OP__ SoftmaxInt
#include <math.h>
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/softmaxint.h"  // Venus backend implementation
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/softmaxint.h"   // Arcs backend implementation
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/softmaxint.h" // VenusA backend implementation
#endif

/**
 * @brief Execute the SoftmaxInt operation
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || op->num_input_ != 1 ||
                        op->num_output_ != 1 || num_tensor < 2 || num_tensor > 3) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    SoftmaxIntAttrs *attr = (SoftmaxIntAttrs *)((int8_t *)op + op->attr_offset_);
    tTensor *data = tensors[0];
    tTensor *out = tensors[1];
#if THINKER_PARAM_CHECK
    if (data == NULL || out == NULL || data->dptr_ == 0 || out->dptr_ == 0 ||
                        data->shape_.ndim_ == 0 || !equalShape(&data->shape_, &out->shape_) ||
                        data->zero_ != 0 || out->zero_ != 0 ||
                        !isfinite(data->scale_) || !isfinite(out->scale_) ||
                        floorf(data->scale_) != data->scale_ || floorf(out->scale_) != out->scale_) {
        return (T_ERR_INVALID_DATA);
    }
#endif

    tTensor *workspace = NULL;
    if (num_tensor > op->num_input_ + op->num_output_) {
        workspace = tensors[op->num_input_ + op->num_output_];
    }
#if THINKER_RUNTIME_CHECK
    if (workspace == NULL || workspace->dptr_ == 0 ||
                          workspace->shape_.ndim_ != 1 || workspace->dtype_ != Int8 ||
                          workspace->byte_ != 1 || workspace->mem_.type_ != 2) {
        return (T_ERR_NO_WORKSPACE);
    }
#endif

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();  // Start profiling
#endif
    THINKER_RET_CHECK(softmaxint_luna(data, out, workspace, attr), "softmaxint_luna");
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","SoftmaxInt", total_t);  // Print profiling results
#endif
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
