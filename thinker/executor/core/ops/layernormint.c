#undef __OP__
#define __OP__ LayerNormInt
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#ifdef THINKER_USE_VENUS
#include "./venus/layernormint.h"
#endif

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                   tDMA_List *list) {
  int32_t ret = T_ERR_NO_IMPLEMENTED;
  CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
  LayerNormIntAttrs *attrs =
      (LayerNormIntAttrs *)((int8_t *)op + op->attr_offset_);

  tTensor *X = ((tTensor **)tensors)[0];
  tTensor *Y = ((tTensor **)tensors)[op->num_input_];

#ifdef THINKER_USE_VENUS
  getWeightData(list, 0);
#endif
  tTensor weight_tmp;
  memcpy(&weight_tmp, tensors[1], sizeof(tTensor));
  tTensor *weight = &weight_tmp;  // tensors[1];
  tTensor *bias = NULL;
  tTensor *workspace = NULL;

  tTensor *dma_buffer = NULL;
  if (num_tensor == op->num_input_ + op->num_output_ + 1) {
    dma_buffer = ((tTensor **)tensors)[op->num_input_ + op->num_output_];
    weight->dptr_ = (addr_type)dma_buffer->dptr_;
  } else if (num_tensor == op->num_input_ + op->num_output_ + 2) {
    workspace = ((tTensor **)tensors)[op->num_input_ + op->num_output_];
    dma_buffer = ((tTensor **)tensors)[op->num_input_ + op->num_output_ + 1];
    weight->dptr_ = (addr_type)dma_buffer->dptr_;
  }

  if (3 == op->num_input_) {
    bias = ((tTensor **)tensors)[op->num_input_ - 1];
    bias->scale_ = X->scale_ + weight->scale_;
    int32_t size = getShapeSize(&(weight->shape_));
    bias->dptr_ = (addr_type)((int8_t *)dma_buffer->dptr_ +
                              ALIGN16(size * weight->byte_));  // ALIGN16(size)
  }

#ifdef THINKER_USE_VENUS
  ret = layernormalint_venus(X, weight, bias, Y, workspace, attrs);
#endif
  if (ret != T_SUCCESS) {
    return ret;
  }
  return ret;
}

#include "core/operator_template.h"
#undef __OP__
