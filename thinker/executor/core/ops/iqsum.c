#undef __OP__
#define __OP__ iqSum
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"
#ifdef THINKER_USE_VENUS
#include "./venus/iqsum.h"
#endif

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                   tDMA_List *list) {
  CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_ + 1));
  int32_t ret = T_ERR_NO_IMPLEMENTED;
  iqSumAttrs *attrs = (iqSumAttrs *)((int8_t *)op + op->attr_offset_);
  tTensor *Temp = tensors[op->num_input_ + op->num_output_];
#ifdef THINKER_USE_VENUS
  ret = iqsum_luna(tensors[0], Temp, tensors[op->num_input_], attrs);
#endif
  if (ret != T_SUCCESS) {
    return ret;
  }
  return ret;
}

#include "core/operator_template.h"
#undef __OP__