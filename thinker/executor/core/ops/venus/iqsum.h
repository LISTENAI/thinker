#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "luna/luna_math.h"
#include "thinker_status.h"

typedef void *luna_sum_api_item;
typedef int32_t (*luna_sum_handle)(const void *, int32_t *, uint32_t, uint32_t);
static luna_sum_api_item luna_sum_items[3] = {
    luna_vector_sum_q7_int32,
    luna_vector_sum_q15_int32,
    luna_vector_sum_q31_int32,
};

int32_t iqsum_luna(tTensor *inputs, tTensor *Temp, tTensor *output,
                   SumAttrs *attrs) {
  int32_t axis = attrs->axis;

  return 0;
}