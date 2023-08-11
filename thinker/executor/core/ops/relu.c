#undef __OP__
#define __OP__ Relu
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/relu.h"
#endif

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                   tDMA_List *list) {
  CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
  int32_t ret = T_ERR_NO_IMPLEMENTED;

  tTensor *Workspace = NULL;
  if (num_tensor > op->num_input_ + op->num_output_)
  {
	  Workspace = tensors[op->num_input_ + op->num_output_];
  }
#ifdef THINKER_USE_VENUS
  ret = relu_luna(tensors[0], tensors[op->num_input_], Workspace);
#endif
  if (ret != T_SUCCESS) {
    return ret;
  }
  return ret;
}

#include "core/operator_template.h"
#undef __OP__
