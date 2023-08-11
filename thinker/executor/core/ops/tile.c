#undef __OP__
#define __OP__ Tile
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/tile.h"
#endif

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                   tDMA_List *list) {
  CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));
  int32_t ret = T_ERR_NO_IMPLEMENTED;
  if (num_tensor != 3) return T_ERR_INVALID_PARA;

#ifdef THINKER_USE_VENUS
  ret = tile_luna(tensors[0],tensors[1],tensors[op->num_input_]);
#endif
  if (ret != T_SUCCESS) {
    return ret;
  }
  return ret;
}

#include "core/operator_template.h"
#undef __OP__