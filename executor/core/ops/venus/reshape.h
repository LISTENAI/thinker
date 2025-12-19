#ifndef __REQUANT_H__
#define __REQUANT_H__

#include <math.h>
#include <stdio.h>
#include <string.h>
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"

/**
 * @brief Reshape tensor data without changing the underlying memory
 * @param X Input tensor
 * @param Y Output tensor
 * @return int32_t Operation status
 */
int32_t reshape_luna(tTensor *X, tTensor *Y) {
    int8_t *input = (int8_t *)X->dptr_;
    int8_t *output = (int8_t *)Y->dptr_;
    
    if (input != output) {
        size_t size = getTensorSize(X);
        memcpy(output, input, X->byte_ * size);
    }
    
    return 0;
}

#endif