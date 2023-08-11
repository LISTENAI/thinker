#ifndef __RELU_H__
#define __RELU_H__

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "luna/luna_math.h"
#include "thinker_status.h"

static int32_t calc_relu_luna(int32_t X_dtype, int32_t Y_dtype, void *src,
                              void *dst, int32_t size, int32_t shift) {
  int32_t ret = -1;
  switch (X_dtype) {
    case Int8: {
      switch (Y_dtype) {
        case Int8:
          ret = luna_relu_q7_int8((const q7_t *)src, (q7_t *)dst, size, shift);
          break;
        case Int16:
          ret =
              luna_relu_q7_int16((const q7_t *)src, (q15_t *)dst, size, shift);
          break;
        case Int32:
          ret =
              luna_relu_q7_int32((const q7_t *)src, (q31_t *)dst, size, shift);
          break;
      }
    } break;
    case Int16: {
      switch (Y_dtype) {
        case Int8:
          ret =
              luna_relu_q15_int8((const q15_t *)src, (q7_t *)dst, size, shift);
          break;
        case Int16:
          ret = luna_relu_q15_int16((const q15_t *)src, (q15_t *)dst, size,
                                    shift);
          break;
        case Int32:
          ret = luna_relu_q15_int32((const q15_t *)src, (q31_t *)dst, size,
                                    shift);
          break;
      }
    } break;
    case Int32: {
      switch (Y_dtype) {
        case Int8:
          ret =
              luna_relu_q31_int8((const q31_t *)src, (q7_t *)dst, size, shift);
          break;
        case Int16:
          ret = luna_relu_q31_int16((const q31_t *)src, (q15_t *)dst, size,
                                    shift);
          break;
        case Int32:
          ret = luna_relu_q31_int32((const q31_t *)src, (q31_t *)dst, size,
                                    shift);
          break;
      }
    } break;
  }
  return ret;
}

tStatus relu_luna(tTensor *X, tTensor *Y, tTensor *Workspace) {
	int32_t ret = -1;
	int32_t shift = 0;

	void *src = (void *)X->dptr_;
	void *dst = (void *)Y->dptr_;
	void *tmp_buf = NULL;
	int32_t tmp_size = 0;

	if (NULL != Workspace)
	{
		tmp_buf = (void *)Workspace->dptr_;
		tmp_size = getTensorSize(Workspace);
	}

	size_t size = getTensorSize(X);

	if (2 != X->mem_.type_)  // X is psram
	{
		if (Int8 != X->dtype_ || Int8 != Y->dtype_)
		{
			return -1;
		}

		int32_t split_num = 1;
		int32_t split_size = size;
		while (split_size > tmp_size)
		{
			split_size = (size + split_num - 1) / split_num;
			split_num++;
		}
		int32_t final_s_size = size - split_size * (split_num - 1);
		for (int i = 0; i < split_num; i++)
		{
			int8_t *p_in = (int8_t *)src + i * split_size;
			int8_t *p_ou = (int8_t *)dst + i * split_size;
			if (i == (split_size - 1))
			{
				split_size = final_s_size;
			}
			memcpy(tmp_buf, p_in, split_size);
			ret = calc_relu_luna(Int8, Int8, tmp_buf, tmp_buf, split_size, shift);
			memcpy(p_ou, tmp_buf, split_size);
		}
	}
	else
	{
		ret = calc_relu_luna(X->dtype_, Y->dtype_, src, dst, size, shift);
	}

	return T_SUCCESS;
}
#endif
