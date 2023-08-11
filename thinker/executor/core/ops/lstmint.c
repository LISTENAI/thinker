#undef __OP__
#define __OP__ LSTMInt
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

#ifdef THINKER_USE_VENUS
#include "./venus/lstmint.h"
#endif

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                   tDMA_List *list) {
  CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
  LstmIntAttrs *attr = (LstmIntAttrs *)((int8_t *)op + op->attr_offset_);
  int32_t ret = T_ERR_NO_IMPLEMENTED;

#ifdef THINKER_USE_VENUS
  getWeightData(list, 0);
#endif
  int32_t w_idx = 1;
  tTensor *t_seq = NULL;
  tTensor *t_hidden_in = NULL;
  tTensor *t_cell_in = NULL;
  if (op->num_input_ == 6)  //include seq_len
  {
    w_idx = 2;
    t_seq = tensors[1];
  }else if(op->num_input_ == 8)
  {
    w_idx = 4;
    t_seq = tensors[1];
    t_hidden_in = tensors[2];
    t_cell_in = tensors[3];
  }
  tTensor *input = tensors[0];
  tTensor i2h_w_tmp;
  tTensor h2h_w_tmp;
  tTensor i2h_bias_tmp;
  tTensor h2h_bias_tmp;
  tTensor *i2h_w = &i2h_w_tmp;        // = tensors[1];
  tTensor *h2h_w = &h2h_w_tmp;        // = tensors[2];
  tTensor *i2h_bias = &i2h_bias_tmp;  // = tensors[3];
  tTensor *h2h_bias = &h2h_bias_tmp;  // = tensors[4];
  memcpy(&i2h_w_tmp, tensors[w_idx], sizeof(tTensor));
  memcpy(&h2h_w_tmp, tensors[w_idx + 1], sizeof(tTensor));
  memcpy(&i2h_bias_tmp, tensors[w_idx + 2], sizeof(tTensor));
  memcpy(&h2h_bias_tmp, tensors[w_idx + 3], sizeof(tTensor));

  tTensor *output = tensors[op->num_input_];
  tTensor *hidden_o = tensors[op->num_input_ + 1];
  hidden_o->dtype_ = Int8;
  tTensor *hidden_c = tensors[op->num_input_ + 2];
  hidden_c->dtype_ = Int16;
  tTensor *dma_temp = NULL;

  tTensor *workspace = (void *)NULL;
  if (num_tensor == op->num_input_ + op->num_output_ + 2) {
    int32_t iw_size = getShapeSize(&(i2h_w->shape_));
    int32_t hw_size = getShapeSize(&(h2h_w->shape_));
    int32_t ib_size = getShapeSize(&(i2h_bias->shape_));
    int32_t hb_size = getShapeSize(&(h2h_bias->shape_));
    workspace = tensors[op->num_input_ + op->num_output_];
    dma_temp = tensors[op->num_input_ + op->num_output_ + 1];
    i2h_w->dptr_ = (addr_type)dma_temp->dptr_;
    h2h_w->dptr_ = (addr_type)((int8_t *)i2h_w->dptr_ + ALIGN16(iw_size));
    i2h_bias->dptr_ = (addr_type)((int8_t *)h2h_w->dptr_ + ALIGN16(hw_size));
    h2h_bias->dptr_ =
        (addr_type)((int32_t *)i2h_bias->dptr_ + ALIGN16(ib_size));
  }

  // hidden_c_inst.dptr_ = (void*)NULL;

#ifdef THINKER_USE_VENUS
  ret = lstmint_luna(input, t_hidden_in, t_cell_in, i2h_w, h2h_w,
                     i2h_bias, h2h_bias, t_seq, output, hidden_o, hidden_c,
                     attr, workspace);
#endif

  if (ret != T_SUCCESS) {
    return ret;
  }
  return 0;
}

#include "core/operator_template.h"
#undef __OP__
