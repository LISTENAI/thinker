#undef __OP__
#define __OP__ Flatten
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/flatten.h"
#endif

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                   tDMA_List *list) {
  CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));
  int32_t ret = T_ERR_NO_IMPLEMENTED;
  FlattenAttrs *attr = (FlattenAttrs *)((int8_t *)op + op->attr_offset_);

#ifdef THINKER_USE_VENUS
  ret = flatten_luna(tensors[0], tensors[op->num_input_], attr);
#endif
  if (ret != T_SUCCESS) {
    return ret;
  }
  return 0;
}

#include "core/operator_template.h"
#undef __OP__