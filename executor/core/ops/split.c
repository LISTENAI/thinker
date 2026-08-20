// Split operator implementation

#undef __OP__
#define __OP__ Split
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"
#include "core/comm/utils.h"

#ifdef THINKER_USE_VENUS
#include "./venus/split.h"  // Venus backend implementation
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/split.h"   // Arcs backend implementation
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/split.h" // VenusA backend implementation
#endif

/**
 * @brief Execute the Split operation
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    (void)list;
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || op->num_input_ != 1 ||
                        op->num_output_ <= 0 || op->num_output_ > 8 ||
                        num_tensor != op->num_input_ + op->num_output_) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    SplitAttrs *attr = (SplitAttrs *)((int8_t *)op + op->attr_offset_);
    tTensor *input = tensors[0];
#if THINKER_PARAM_CHECK
    if (input == NULL || input->dptr_ == 0 ||
                        input->shape_.ndim_ == 0 || input->shape_.ndim_ > 7 ||
                        input->dtype_ == Int4 ||
                        input->byte_ == 0 || attr->dims != op->num_output_) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    int32_t axis = attr->axis;
    if (axis < 0) axis += input->shape_.ndim_;
#if THINKER_PARAM_CHECK
    if (axis < 0 || axis >= input->shape_.ndim_) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    attr->axis = axis;
    uint64_t total = 0;
    for (int32_t n = 0; n < attr->dims; ++n) {
        tTensor *output = tensors[n + 1];
#if THINKER_PARAM_CHECK
        if (output == NULL || output->dptr_ == 0 ||
                            attr->split[n] <= 0 ||
                            output->shape_.ndim_ != input->shape_.ndim_ ||
                            output->layout_ != input->layout_) {
            return (T_ERR_INVALID_PARA);
        }

        if (output->dtype_ != input->dtype_ ||
                            output->byte_ != input->byte_) {
            return (T_ERR_INVALID_DATATYPE);
        }
#endif
        for (int32_t i = 0; i < input->shape_.ndim_; ++i) {
            uint32_t expected = i == axis ? (uint32_t)attr->split[n]
                                          : input->shape_.dims_[i];
#if THINKER_PARAM_CHECK
            if (output->shape_.dims_[i] != expected) {
                return (T_ERR_INVALID_DATA);
            }
#endif
        }
        total += (uint32_t)attr->split[n];
    }
#if THINKER_PARAM_CHECK
    if (total != input->shape_.dims_[axis]) {
        return (T_ERR_INVALID_DATA);
    }
#endif

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
    THINKER_RET_CHECK(split_venus(input, tensors, attr), "split_venus");  // Execute split operation
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
