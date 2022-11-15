#undef __OP__
#define __OP__ iqSub
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/iqsub.h"
#endif

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                   tDMA_List *list) {
  CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_ + 1));
  iqBinaryAttrs *attrs = (iqBinaryAttrs *)((int8_t *)op + op->attr_offset_);
  int32_t ret = T_ERR_NO_IMPLEMENTED;
  tTensor *temp = tensors[op->num_input_ + op->num_output_];
#ifdef THINKER_USE_VENUS
  ret = iqsub_luna(tensors[0], tensors[1], temp, tensors[op->num_input_]);
#endif
  if (ret != T_SUCCESS) {
    return ret;
  }
  return ret;
}

#include "core/operator_template.h"
#undef __OP__