#undef __OP__
#define __OP__ MaxPool
#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/maxpool.h"
#endif

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                   tDMA_List *list) {
  CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
  PoolAttrs *attrs = (PoolAttrs *)((int8_t *)op + op->attr_offset_);
  tTensor *X = ((tTensor **)tensors)[0];
  tTensor *Y = ((tTensor **)tensors)[op->num_input_];
  tTensor *Temp = ((tTensor **)tensors)[op->num_input_ + 1];
  int32_t ret = T_SUCCESS;

#ifdef THINKER_USE_VENUS
  ret = maxpool_luna(X, Y, Temp, attrs);
#endif

  return ret;
}

#include "core/operator_template.h"
#undef __OP__
