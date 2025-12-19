// Tile operator implementation

#undef __OP__
#define __OP__ Tile
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/tile.h"  // Venus backend implementation
#endif

#ifdef THINKER_USE_ARCS
#include "./arcs/tile.h"   // Arcs backend implementation
#endif

#ifdef THINKER_USE_VENUSA
#include "./venusA/tile.h" // VenusA backend implementation
#endif

/**
 * @brief Execute the Tile operation
 * @param op Pointer to the operator
 * @param tensors Array of input and output tensors
 * @param num_tensor Number of tensors
 * @param list DMA list (unused in this implementation)
 * @return int32_t Execution status
 */
int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) {
    CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));  // Validate tensor count

    if (num_tensor != 3) {
        return T_ERR_INVALID_PARA;  // Invalid number of tensors
    }

    int32_t ret = T_ERR_NO_IMPLEMENTED;

#if THINKER_USE_VENUS || THINKER_USE_ARCS || THINKER_USE_VENUSA
    ret = tile_luna(tensors[0], tensors[1], tensors[op->num_input_]);  // Execute tile operation
#endif

    return ret;
}

#include "core/operator_template.h"
#undef __OP__