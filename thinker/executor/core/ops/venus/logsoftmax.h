#ifndef _LOGSOFTMAX_LUNA_H_
#define _LOGSOFTMAX_LUNA_H_

#include <math.h>
#include <stdio.h>
#include <string.h>
#include "core/comm/utils.h"
#include "core/comm/thinker_log.h"
#include "luna/luna_math.h"
#include "core/operator_attrs.h"
#include "hifi/NatureDSP_Signal_math.h"
#include "hifi/NatureDSP_Signal_vector.h"

int32_t logsoftmax_luna(tTensor *data, tTensor *out, LogSoftmaxAttrs *attrs)
{
    CHECK_EQ(data->dtype_, out->dtype_);
    if(attrs->axis<0)
        attrs->axis+=data->shape_.ndim_;
    CHECK_LT(attrs->axis, data->shape_.ndim_);
    int leading = 1, stride = 1;
    int i = 0;
    for( ; i < attrs->axis; ++i) {
        leading *= data->shape_.dims_[i];
    }
    for( ; i < data->shape_.ndim_; ++i) {
        stride *= data->shape_.dims_[i];
    }
    if(Float32 == data->dtype_) {
        float *src = (float *)(data->dptr_);
        float *dst = (float *)(out->dptr_);
        for(int l = 0; l < leading; ++l) {
            float *lsrc = src + l * stride;
            float *ldst = dst + l * stride;
            vec_softmaxf(ldst, lsrc, stride);
            vec_lognf(lsrc, ldst, stride);
            memcpy(ldst, lsrc, stride * sizeof(float));
        }
    } else {
        THINKER_LOG_FATAL("LogSoftmax support float data type only.");
    }
    return 0;
}
#endif
