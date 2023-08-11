#undef __OP__
#define __OP__ ConvTranspose2dInt

#include <math.h>

#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"

#ifdef THINKER_USE_VENUS
#include "./venus/deconv2dint.h"
#endif

static int32_t xx = 0;


int32_t X(Forward)(tOperator* op, tTensor** tensors, int32_t num_tensor,
                   tDMA_List* list) {
  CHECK_GE(num_tensor, (op->num_input_ + op->num_output_));
  CHECK_GE(op->num_input_, 2);
  CHECK_LE(op->num_input_, 3);
  ConvTranspose2dIntAttrs* attrs = (ConvTranspose2dIntAttrs*)((int8_t*)op + op->attr_offset_);
  tTensor* X = ((tTensor**)tensors)[0];

#ifdef THINKER_USE_VENUS
  getWeightData(list, 0);
#endif

  tTensor* W = ((tTensor**)tensors)[1];
  tTensor* Y = ((tTensor**)tensors)[op->num_input_];

  tTensor* Temp = NULL;
  tTensor* dma_temp = NULL;
  tTensor Weight_temp = W[0];

  if (num_tensor == op->num_input_ + op->num_output_ + 1) {
    dma_temp = ((tTensor**)tensors)[op->num_input_ + op->num_output_];
    Weight_temp.dptr_ = (addr_type)dma_temp->dptr_;
  } else if (num_tensor == op->num_input_ + op->num_output_ + 2) {
    Temp = ((tTensor**)tensors)[op->num_input_ + op->num_output_];
    dma_temp = ((tTensor**)tensors)[op->num_input_ + op->num_output_ + 1];
    Weight_temp.dptr_ = (addr_type)dma_temp->dptr_;
  }

  int32_t ret = T_ERR_NO_IMPLEMENTED;
#ifdef THINKER_USE_VENUS
  if (3 == op->num_input_) {
    tTensor* Bias = ((tTensor**)tensors)[op->num_input_ - 1];
    tTensor Bias_temp = Bias[0];
    Bias_temp.scale_ = X->scale_ + W->scale_;
    int32_t size = getShapeSize(&(W->shape_));
    Bias_temp.dptr_ = (addr_type)((int8_t*)Weight_temp.dptr_ + ALIGN16(size));
    ret = deconv2dint_venus(X, &Weight_temp, &Bias_temp, Y, Temp, attrs);
  } else {
    ret = deconv2dint_venus(X, &Weight_temp, NULL, Y, Temp, attrs);
  }
#endif

  if (ret != T_SUCCESS) {
    return ret;
  }
  return ret;
}

#include "core/operator_template.h"
#undef __OP__
