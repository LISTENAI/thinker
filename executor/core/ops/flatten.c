#undef __OP__
#define __OP__ Flatten
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"
#include "core/comm/utils.h"

#ifdef THINKER_USE_VENUS
#include "./venus/flatten.h"
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/flatten.h"
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/flatten.h"
#endif

/**
 * Forward pass implementation for Flatten operator
 * Reshapes input tensor to 2D format (batch_size, total_elements)
 * @param op: Operator structure containing flatten attributes
 * @param tensors: Array of input/output tensors (input, output)
 * @param num_tensor: Total number of tensors (must equal input + output count)
 * @param list: DMA list (unused)
 * @return: Status code indicating success or failure
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    (void)list;
#if THINKER_PARAM_CHECK
    if (op == NULL || tensors == NULL || op->num_input_ != 1 ||
                        op->num_output_ != 1 || num_tensor != 2) {
        return (T_ERR_INVALID_PARA);
    }
#endif

    tTensor *input = tensors[0];
    tTensor *output = tensors[1];
#if THINKER_PARAM_CHECK
    if (input == NULL || output == NULL ||
                        (getTensorDataSize(input) > 0 &&
                         (input->dptr_ == 0 || output->dptr_ == 0))) {
        return (T_ERR_INVALID_PARA);
    }

    if (input->shape_.ndim_ > 7 || output->shape_.ndim_ != 2) {
        return (T_ERR_INVALID_PARA);
    }

    if (input->dtype_ != output->dtype_ ||
                        input->byte_ != output->byte_) {
        return (T_ERR_INVALID_DATATYPE);
    }

    if (input->layout_ != output->layout_ ||
                        getTensorDataSize(input) != getTensorDataSize(output)) {
        return (T_ERR_INVALID_DATA);
    }
#endif
    
    FlattenAttrs *attr = (FlattenAttrs *)((int8_t *)op + op->attr_offset_);
    int32_t axis = attr->axis;
    int32_t rank = input->shape_.ndim_;
    if (axis < 0) axis += rank;
#if THINKER_PARAM_CHECK
    if (axis < 0 || axis > rank) {
        return (T_ERR_INVALID_PARA);
    }
#endif
    size_t first = 1;
    size_t second = 1;
    for (int32_t i = 0; i < axis; ++i) first *= input->shape_.dims_[i];
    for (int32_t i = axis; i < rank; ++i) second *= input->shape_.dims_[i];
#if THINKER_PARAM_CHECK
    if (output->shape_.dims_[0] != first ||
                        output->shape_.dims_[1] != second) {
        return (T_ERR_INVALID_DATA);
    }
#endif
    
#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
    // Call hardware-specific flatten implementation
    THINKER_RET_CHECK(flatten_luna(input, output, attr), "flatten_luna");
#endif

    return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
