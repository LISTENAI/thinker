#undef __OP__
#define __OP__ Unsqueeze
#include "core/operator_register.h"
#include "thinker_status.h"

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list) 
{
  CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));
  tTensor *X  = tensors[0];
  tTensor *Y  = tensors[op->num_input_];

  if (num_tensor != 2) 
    return T_ERR_INVALID_PARA;

  if (X->dptr_ != Y->dptr_)
    return T_ERR_INVALID_DATA;

  return T_SUCCESS;
}

#include "core/operator_template.h"
#undef __OP__
