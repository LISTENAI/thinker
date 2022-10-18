#undef __OP__
#define __OP__ Quant
#include <stdio.h>
#include <string.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_VENUS
#include "./venus/quantize_linear.h"
#endif

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                   tDMA_List *list) {
  CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
  QuantAttrs *attr = (QuantAttrs *)((int8_t *)op + op->attr_offset_);
  tTensor *X = ((tTensor **)tensors)[0];
  tTensor *Y = ((tTensor **)tensors)[op->num_input_];
  tTensor *workspace = NULL;
  if (num_tensor > op->num_input_ + op->num_output_) {
    workspace = tensors[num_tensor - 1];
  }
  int32_t ret = T_ERR_NO_IMPLEMENTED;

#ifdef THINKER_USE_VENUS
  ret = quantize_linear_luna(X, Y, workspace, attr);
#endif
  if (ret != T_SUCCESS) {
    return ret;
  }
  return ret;
}

#include "core/operator_template.h"
#undef __OP__
