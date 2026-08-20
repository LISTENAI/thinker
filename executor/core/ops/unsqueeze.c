#undef __OP__
#define __OP__ Unsqueeze
#include "core/operator_register.h"
#include "thinker_status.h"

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) 
{
  CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));
  if ((op->num_input_ != 1 && op->num_input_ != 2) || op->num_output_ != 1)
    return T_ERR_INVALID_PARA;

  tTensor *X  = tensors[0];
  tTensor *Y  = tensors[op->num_input_];

  if (X == NULL || Y == NULL || X->dptr_ == 0 || X->dptr_ != Y->dptr_)
    return T_ERR_INVALID_DATA;

  return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
