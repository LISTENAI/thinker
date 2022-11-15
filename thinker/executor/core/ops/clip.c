#undef __OP__
#define __OP__ Clip
#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/type_switch.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"
#include "thinker_status.h"
#ifdef THINKER_USE_VENUS
#include "venus/clip.h"
#endif

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                   tDMA_List *list) {
  ClipAttrs *attrs = (ClipAttrs *)((int8_t *)op + op->attr_offset_);
  tTensor *X = ((tTensor **)tensors)[0];
  tTensor *Y = ((tTensor **)tensors)[op->num_input_];
  float max = attrs->max;
  float min = attrs->min;
  if (op->num_input_ > 1) {
    tTensor *XMin = tensors[1];
    tTensor *XMax = tensors[2];
    DATA_TYPE_SWITCH_ALL(XMin->dtype_, Type, {
      max = *(Type *)XMax->dptr_;
      min = *(Type *)XMin->dptr_;
    });
  }

  int32_t ret = T_ERR_NO_IMPLEMENTED;
#ifdef THINKER_USE_VENUS
  ret = clip_venus(X, Y, attrs, max, min);
#endif
  if (ret != T_SUCCESS) {
    return ret;
  }
  return ret;
}

#include "core/operator_template.h"
#undef __OP__
