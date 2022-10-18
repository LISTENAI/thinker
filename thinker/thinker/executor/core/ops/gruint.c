#undef __OP__
#define __OP__ GRUInt
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_VENUS
#include "./venus/gruint.h"
#endif

int32_t X(Forward)(tOperator* op, tTensor** tensors, int32_t num_tensor,
                   tDMA_List* list) {
  CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
  GRUIntAttrs* attr = (GRUIntAttrs*)((int8_t*)op + op->attr_offset_);
  int32_t ret = T_ERR_NO_IMPLEMENTED;

  tTensor* input = tensors[0];
  tTensor* i2h_w = tensors[1];
  tTensor* h2h_w = tensors[2];
  tTensor* i2h_bias = tensors[3];
  tTensor* h2h_bias = tensors[4];

  tTensor* output = tensors[op->num_input_];
  tTensor* hidden_o = tensors[op->num_input_ + 1];

  tTensor* workspace = NULL;
  if (num_tensor > op->num_input_ + op->num_output_) {
    workspace = tensors[op->num_input_ + op->num_output_];
  }

  tTensor hidden_i_inst;

  hidden_i_inst.shape_.ndim_ = 0;
  // hidden_i_inst.dptr_ = NULL;
  tTensor mask;
  // mask.dptr_ = NULL;
  mask.shape_.ndim_ = 0;

#ifdef THINKER_USE_VENUS
  ret = gruint_luna(input, &hidden_i_inst, i2h_w, h2h_w, i2h_bias, h2h_bias,
                    &mask, output, hidden_o, attr, workspace);
#endif

  if (ret != T_SUCCESS) {
    return ret;
  }
  return 0;
}

#include "core/operator_template.h"
#undef __OP__
