#undef __OP__
#define __OP__ AvgPool2dInt
#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/avgpool2dint.h"
#endif

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                   tDMA_List *list) {
  CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
  PoolAttrs *attrs = (PoolAttrs *)((int8_t *)op + op->attr_offset_);
  tTensor *X = ((tTensor **)tensors)[0];
  tTensor *Y = ((tTensor **)tensors)[op->num_input_];

  int32_t ret = T_ERR_NO_IMPLEMENTED;

#ifdef THINKER_USE_VENUS
  if (num_tensor > ((op->num_input_ + op->num_output_))) {
    tTensor *workspace = ((tTensor **)tensors)[num_tensor - 1];
    ret = avgpool2dint_luna(X, Y, workspace, attrs);
  }
#endif

  return ret;
}

#include "core/operator_template.h"
#undef __OP__
