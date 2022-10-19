#undef __OP__
#define __OP__ convtranspose2dint

#include <math.h>

#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/deconv2dint.h"
#endif

static int32_t xx = 0;

typedef struct X(Attrs) {
  uint8_t dilation[3];
  uint16_t kernel[3];
  uint8_t pad[6];
  uint8_t output_padding[6];
  uint8_t stride[3];
  int16_t group;
  int16_t layout;
  uint8_t quant_type;
  uint8_t act_type;
  int8_t max;
  int8_t min;
  uint8_t optimize_Strategy;
  uint8_t algorithm;
} X(Attrs);

int32_t X(Forward)(tOperator* op, tTensor** tensors, int32_t num_tensor,
                   tDMA_List* list) {
  CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
  CHECK_GE(op->num_input_, 8);
  CHECK_LE(op->num_input_, 9);
  ConvTranspose2dIntAttrs* attrs =
      (ConvTranspose2dIntAttrs*)((int8_t*)op + op->attr_offset_);
  tTensor* X = ((tTensor**)tensors)[0];
  tTensor* x_scale = ((tTensor**)tensors)[1];
  tTensor* x_zero = ((tTensor**)tensors)[2];

  X->scale_ = *(int32_t*)(x_scale->dptr_);

#ifdef THINKER_USE_VENUS
  getWeightData(list, 0);
#endif

  tTensor* W = ((tTensor**)tensors)[3];
  tTensor* w_scale = ((tTensor**)tensors)[4];
  tTensor* w_zero = ((tTensor**)tensors)[5];

  W->scale_ = *(int32_t*)(w_scale->dptr_);

  tTensor* Y = ((tTensor**)tensors)[op->num_input_];
  tTensor* y_scale = ((tTensor**)tensors)[6];
  tTensor* y_zero = ((tTensor**)tensors)[7];

  Y->scale_ = *(int32_t*)(y_scale->dptr_);

  tTensor* Temp = NULL;
  tTensor* dma_temp = NULL;
  tTensor Weight_temp = W[0];

  if (num_tensor == op->num_input_ + op->num_output_ + 1) {
    Temp = ((tTensor**)tensors)[op->num_input_ + op->num_output_];
    Weight_temp.dptr_ = (addr_type)dma_temp->dptr_;
  } else if (num_tensor == op->num_input_ + op->num_output_ + 2) {
    Temp = ((tTensor**)tensors)[op->num_input_ + op->num_output_];
    dma_temp = ((tTensor**)tensors)[op->num_input_ + op->num_output_ + 1];
    Weight_temp.dptr_ = (addr_type)dma_temp->dptr_;
  }

  int32_t ret = T_ERR_NO_IMPLEMENTED;
  if (9 == op->num_input_) {
    tTensor* Bias = ((tTensor**)tensors)[op->num_input_ - 1];
    tTensor Bias_temp = Bias[0];
    Bias_temp.scale_ = X->scale_ * W->scale_;
    int32_t size = getShapeSize(&(W->shape_));
    Bias_temp.dptr_ =
        (uint64_t)((int8_t*)Weight_temp.dptr_ + ALIGN16(size) + 64);

#ifdef THINKER_USE_VENUS
    ret = deconv2dint_venus(X, &Weight_temp, &Bias_temp, Y, Temp, attrs);
#endif
  }

  else {
#ifdef THINKER_USE_VENUS
    ret = deconv2dint_venus(X, &Weight_temp, NULL, Y, Temp, attrs);
#endif
  }
  if (ret != T_SUCCESS) {
    return ret;
  }
  return ret;
}

#include "core/operator_template.h"
#undef __OP__
