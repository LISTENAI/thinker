#ifndef __REQUANT_H__
#define __REQUANT_H__

#include <string.h>

#include "core/comm/utils.h"
#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"

#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif

#include "thinker_status.h"

/**
 * @brief Reshape operation that copies data between tensors
 * @param X Input tensor
 * @param Y Output tensor
 * @return Operation result status
 */
int32_t reshape_luna(tTensor *X, tTensor *Y) {
  int8_t *input 	= (int8_t *)X->dptr_;
  int8_t *output 	= (int8_t *)Y->dptr_;

  // Only copy if input and output buffers are different
  if (input != output) {
    size_t size = getTensorSize(X) * X->byte_;
    if (size != 0) {
    	if (Y->mem_.type_ != 2) {
			// Output is in PSRAM - use PSRAM copy out
    		opi_psram_cpy_out(output, input, size);
    		return T_SUCCESS;
    	}
    	else if (X->mem_.type_ != 2)
    	{
			// Input is in PSRAM - use PSRAM memcpy
    		return API_LIB(psrammemcpy_i8o8)(output, input, size);
    	}
    	else
			// Both in fast memory - use regular memcpy
    		return API_LIB(memcpy_i8o8)(output, input, size);
    }
  }

  return T_SUCCESS;
}

#endif
