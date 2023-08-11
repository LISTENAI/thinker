#undef __OP__
#define __OP__ iqMul
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/iqmul.h"
#endif

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                   tDMA_List *list) {
	CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
  iqBinaryAttrs *attrs = (iqBinaryAttrs *)((int8_t *)op + op->attr_offset_);
  int32_t ret = T_ERR_NO_IMPLEMENTED;
  tTensor *workspace = ((tTensor **)tensors)[num_tensor - 1];
#ifdef THINKER_USE_VENUS
  ret = iqmul_luna(tensors[0], tensors[1], tensors[op->num_input_], workspace, attrs);
#endif

  if (ret != T_SUCCESS) {
    return ret;
  }
  return ret;
}

#include "core/operator_template.h"
#undef __OP__