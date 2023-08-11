#ifndef _CAST_VENUS_H_
#define _CAST_VENUS_H_

#include "thinker_status.h"
#include "c_api/thinker_define.h"
#include "core/comm/utils.h"
#include "core/comm/type_switch.h"
#include "core/comm/thinker_log.h"
#include <string.h>

int32_t cast_luna(tTensor* X, tTensor* Y, CastAttrs *attrs) {

    // int8_t to = attrs->to;
    assert(Y->dtype_ == attrs->to);
    int32_t size = getTensorSize(X);
     
    for (int i = 0; i < size; i++) {
        DATA_TYPE_SWITCH_ALL(X->dtype_, IType, {
        const IType *input = (IType *)(X->dptr_);
        DATA_TYPE_SWITCH_ALL(Y->dtype_, OType, {
            OType *output = (OType *)(Y->dptr_);
            output[i] = input[i];
        });
        });
    }
    return 0;
}
#endif  //_CAST_VENUS_H_
