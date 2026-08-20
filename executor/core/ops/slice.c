// Slice operator implementation

#undef __OP__
#define __OP__ Slice
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"
#include "core/comm/utils.h"

static int64_t slice_parameter_value(const tTensor *tensor) {
    return tensor->dtype_ == Int32 ? ((const int32_t *)tensor->dptr_)[0]
                                   : ((const int64_t *)tensor->dptr_)[0];
}

#ifdef THINKER_USE_VENUS
#include "./venus/slice.h"  // Venus backend implementation
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/slice.h"   // Arcs backend implementation
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/slice.h" // VenusA backend implementation
#endif

/**
 * @brief Execute the Slice operation
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    (void)list;
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || op->num_input_ < 3 ||
                        op->num_input_ > 5 || op->num_output_ != 1 ||
                        num_tensor != op->num_input_ + 1) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    tTensor *input = tensors[0];
    tTensor *output = tensors[op->num_input_];
#if THINKER_PARAM_CHECK
    if (input == NULL || output == NULL ||
                        input->shape_.ndim_ == 0 || input->shape_.ndim_ > 7 ||
                        output->shape_.ndim_ != input->shape_.ndim_) {
        return (T_ERR_INVALID_PARA);
    }

    for (int32_t i = 1; i < op->num_input_; ++i) {
        tTensor *parameter = tensors[i];
        if (parameter == NULL || parameter->dptr_ == 0 ||
                            (parameter->dtype_ != Int32 &&
                             parameter->dtype_ != Int64) ||
                            parameter->shape_.ndim_ != 1 ||
                            parameter->shape_.dims_[0] != 1) {
            return (T_ERR_INVALID_PARA);
        }
    }

    if (input->dtype_ != output->dtype_ ||
                        input->byte_ != output->byte_) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (input->dtype_ == Int4 || input->byte_ == 0 ||
                        input->layout_ != output->layout_) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    // Get slice parameters
    int64_t start64 = slice_parameter_value(tensors[1]);
    int64_t end64 = slice_parameter_value(tensors[2]);
    int64_t axis64 = op->num_input_ >= 4 ? slice_parameter_value(tensors[3]) : 0;
#if THINKER_PARAM_CHECK
    if (axis64 < INT32_MIN || axis64 > INT32_MAX) {
        return (T_ERR_INVALID_DATA);
    }
#endif
    int32_t axis = (int32_t)axis64;
    int32_t step = 1;
    if (5 == op->num_input_) {
        int64_t step64 = slice_parameter_value(tensors[4]);
#if THINKER_PARAM_CHECK
        if (step64 < INT32_MIN || step64 > INT32_MAX) {
            return (T_ERR_INVALID_DATA);
        }
#endif
        step = (int32_t)step64;
    }
#if THINKER_PARAM_CHECK
    if (step != 1) {
        return (T_ERR_NO_IMPLEMENTED);
    }
#endif

    int32_t rank = input->shape_.ndim_;
#if THINKER_PARAM_CHECK
    if (axis < -rank || axis >= rank) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    if (axis < 0) axis += rank;
    int32_t axis_size = input->shape_.dims_[axis];
    int32_t real_start;
    if (start64 < 0) {
        real_start = start64 < -(int64_t)axis_size
                         ? 0
                         : (int32_t)(start64 + axis_size);
    } else {
        real_start = start64 > axis_size ? axis_size : (int32_t)start64;
    }
    uint32_t slice_size = output->shape_.dims_[axis];
#if THINKER_PARAM_CHECK
    if (slice_size > (uint32_t)(axis_size - real_start)) {
        return (T_ERR_INVALID_DATA);
    }
#endif
    int32_t real_end = real_start + (int32_t)slice_size;
    if (end64 >= INT32_MIN && end64 <= INT32_MAX) {
        int32_t expected_end;
        if (end64 < 0) {
            expected_end = end64 < -(int64_t)axis_size
                               ? 0
                               : (int32_t)(end64 + axis_size);
        } else {
            expected_end = end64 > axis_size ? axis_size : (int32_t)end64;
        }
        int32_t expected_size = expected_end > real_start
                                    ? expected_end - real_start
                                    : 0;
#if THINKER_PARAM_CHECK
        if (slice_size != (uint32_t)expected_size) {
            return (T_ERR_INVALID_DATA);
        }
#endif
    }

#if THINKER_PARAM_CHECK
    for (int32_t i = 0; i < rank; ++i) {
        uint32_t expected = i == axis ? slice_size
                                      : input->shape_.dims_[i];
        if (output->shape_.dims_[i] != expected) {
            return (T_ERR_INVALID_DATA);
        }
    }

    if (getTensorDataSize(output) > 0 &&
                        (input->dptr_ == 0 || output->dptr_ == 0)) {
        return (T_ERR_INVALID_PARA);
    }
#endif

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
#if THINKER_PROFILE
    uint64_t start_t = tick_count();  // Start profiling
#endif
    THINKER_RET_CHECK(slice_luna(input, real_start, real_end, axis, step, output), "slice_luna");  // Execute slice operation
#if THINKER_PROFILE
    uint64_t finish_t = tick_count();
    uint32_t total_t = (uint32_t)(finish_t - start_t);
    printf("%8s | %u | (","Slice", total_t);  // Print profiling results
#endif
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
