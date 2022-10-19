#undef __OP__
#define __OP__ Slice
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/slice.h"
#endif

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                   tDMA_List *list) {
  CHECK_EQ(num_tensor, (op->num_input_ + op->num_output_));
  int32_t start = (int32_t)(((int64_t *)(tensors[1]->dptr_))[0]);
  int32_t end = (int32_t)(((int64_t *)(tensors[2]->dptr_))[0]);
  int32_t axis = (int32_t)(((int64_t *)(tensors[3]->dptr_))[0]);
  int32_t step = 1;
  if (5 == op->num_input_) {
    step = (int32_t)(((int64_t *)(tensors[4]->dptr_))[0]);
  }

  int32_t ret = T_ERR_NO_IMPLEMENTED;
#ifdef THINKER_USE_VENUS
  ret = slice_luna(tensors[0], start, end, axis, step, tensors[op->num_input_]);
#endif
  tTensor *X1 = tensors[0];
  tTensor *Y = tensors[op->num_input_];
  if (ret != T_SUCCESS) {
    return ret;
  }
  return ret;
}

#include "core/operator_template.h"
#undef __OP__
