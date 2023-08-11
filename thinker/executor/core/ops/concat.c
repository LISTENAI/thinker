#undef __OP__
#define __OP__ iqCat
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/concat.h"
#endif

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                   tDMA_List *list) {
  CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
  iqCatAttrs *attr = (iqCatAttrs *)((int8_t *)op + op->attr_offset_);
  int32_t axis = attr->axis;
  if (axis < 0) {
    axis += tensors[0]->shape_.ndim_;
  }
  int32_t ret = T_ERR_NO_IMPLEMENTED;

  tTensor *workspace = NULL;
  if (num_tensor == op->num_input_ + op->num_output_ + 1){
    workspace = ((tTensor**)tensors)[op->num_input_ + op->num_output_];
  }

#ifdef THINKER_USE_VENUS
ret = concat_luna(tensors, axis, op->num_input_, workspace, tensors[op->num_input_]);

#endif
  if (ret != T_SUCCESS) {
    return ret;
  }
  return ret;
}

#include "core/operator_template.h"
#undef __OP__
