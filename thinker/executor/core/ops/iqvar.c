#undef __OP__
#define __OP__ iqVar
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_VENUS
#include "./venus/iqvar.h"
#endif

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                   tDMA_List *list) {
  CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
  iqvarAttrs *attrs = (iqvarAttrs *)((int8_t *)op + op->attr_offset_);
  int32_t ret = T_ERR_NO_IMPLEMENTED;

  tTensor *X = ((tTensor **)tensors)[0];
  tTensor *workspace = ((tTensor **)tensors)[num_tensor - 1];
  tTensor *Y = ((tTensor **)tensors)[op->num_input_];

#ifdef THINKER_USE_VENUS
  ret = iqvar(X, Y, workspace, attrs);
#endif
  if (ret != T_SUCCESS) {
    return ret;
  }
  return 0;
}

#include "core/operator_template.h"
#undef __OP__