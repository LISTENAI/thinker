#undef __OP__
#define __OP__ BatchNorm2dInt
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_VENUS
#include "./venus/batchnormint.h"
#endif

int32_t X(Forward)(tOperator* op, tTensor** tensors, int32_t num_tensor,
                   tDMA_List* list) {
  CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
  int32_t ret = T_ERR_NO_IMPLEMENTED;

  tTensor* X = ((tTensor**)tensors)[0];
  tTensor* W = ((tTensor**)tensors)[1];
  tTensor* Bias = ((tTensor**)tensors)[2];
  tTensor* Y = ((tTensor**)tensors)[op->num_input_];

#ifdef THINKER_USE_VENUS
  tTensor* workspace = ((tTensor**)tensors)[num_tensor - 1];
  ret = batchnormint_luna(X, W, Bias, Y, workspace);
#endif
  if (ret != T_SUCCESS) {
    return ret;
  }
  return 0;
}

#include "core/operator_template.h"
#undef __OP__
