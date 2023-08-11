#ifndef _ADD_LUNA_H_
#define _ADD_LUNA_H_

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "luna/luna_math.h"
#include "thinker_status.h"

int32_t iqadd_luna(tTensor *X1, tTensor *X2, tTensor *Temp, tTensor *Y) {
  int32_t ret = -1;

  int32_t x1_q = (int32_t)X1->scale_;
  int32_t x2_q = (int32_t)X2->scale_;
  int32_t y_q = (int32_t)Y->scale_;
  void *src1 = (void *)X1->dptr_;
  void *src2 = (void *)X2->dptr_;
  void *dst = (void *)Y->dptr_;
  int8_t *tmp_buf1 = NULL;
  int8_t *tmp_buf2 = NULL;
  size_t size = getTensorSize(X1);
  int32_t x1_is_psram = 0;
  int32_t x2_is_psram = 0;
  int32_t y_is_psram = 0;
  int32_t used_tmp_size = 0;

  if (Temp)
  {
    tmp_buf1 = (int8_t *)Temp->dptr_;
  }

  if ((1 == X1->mem_.type_ || 3 == X1->mem_.type_) &&
      (Temp))  // need copy psram to share
  {
    x1_is_psram = 1;
  }

  if ((1 == X2->mem_.type_ || 3 == X2->mem_.type_) &&
      (Temp))  // need copy psram to share
  {
    x2_is_psram = 1;
  }

  if ((1 == Y->mem_.type_ || 3 == Y->mem_.type_) &&
      (Temp))  // need copy psram to share
  {
    y_is_psram = 1;
  }

  if (equalShape(&X1->shape_, &X2->shape_) && (X1->dtype_ == X2->dtype_)) {
	  int32_t scale1 = 1;
	  int32_t scale2 = 1;
	  int32_t shift1 = 0;
	  int32_t shift2 = 0;

	  if (x1_q > y_q)
	  {
		  shift1 = x1_q - y_q;
	  }
	  else
	  {
		  scale1 = (1 << (y_q - x1_q));
	  }
	  if (x2_q > y_q)
	  {
		  shift2 = x2_q - y_q;
	  }
	  else
	  {
		  scale2 = (1 << (y_q - x2_q));
	  }
    switch (X1->dtype_) {
      case Int8: {
        if (x1_is_psram){
          src1 = (int8_t *)Temp->dptr_;
          memcpy(src1, (void *)X1->dptr_, size * sizeof(int8_t));
          if (x1_q != y_q){
            ret = luna_scale_q7_int8((const q7_t *)src1, (scale1), (int8_t *)src1, size, shift1);
          }
        }
        else{
            if (x1_q != y_q){
              src1 = (int8_t *)Temp->dptr_;
              ret = luna_scale_q7_int8((const q7_t *)X1->dptr_, (scale1), (int8_t *)src1, size, shift1);
            }
        }
        if (x2_is_psram){
          src2 = (int8_t *)Temp->dptr_ + (x1_is_psram) * size;
          memcpy(src2, (void *)X2->dptr_, size * sizeof(int8_t));
          if (x2_q != y_q){
            ret = luna_scale_q7_int8((const q7_t *)src2, (scale2), (int8_t *)src2, size, shift2);
          }
        }
        else{
            if (x2_q != y_q){
              src2 = (int8_t *)Temp->dptr_ + ((x1_is_psram) || (x1_q != y_q)) * size;
              ret = luna_scale_q7_int8((const q7_t *)X2->dptr_, (scale2), (int8_t *)src2, size, shift2);
            }          
        }

        if (y_is_psram){
          dst = (int8_t *)Temp->dptr_;
        }

        ret = luna_add_q7_int8((const q7_t *)src1, (q7_t *)src2, (int8_t *)dst, size, 0);

        if (y_is_psram){
          memcpy((void *)Y->dptr_, dst, size * sizeof(int8_t));
        }

        // if (Temp)
        // {
        //   tmp_buf2 = tmp_buf1 + size * sizeof(int8_t);
        //   if (x1_is_psram)
        //   {
        //     memcpy(tmp_buf1, src1, size * sizeof(int8_t));
        //     src1 = tmp_buf1;
        //     used_tmp_size += size * sizeof(int8_t);
        //   }
        //   if (x2_is_psram)
        //   {
        //     memcpy(tmp_buf2, src2, size * sizeof(int8_t));
        //     src2 = tmp_buf2;
        //     used_tmp_size += size * sizeof(int8_t);
        //   }
        //   if (y_is_psram)
        //   {
        //     dst = tmp_buf1;
        //   }
        //   ret = luna_scale_q7_int8((const q7_t *)src1, (scale1), (int8_t *)tmp_buf1, size,
        //                          shift1);
        //   ret = luna_scale_q7_int8((const q7_t *)src2, (scale2), (int8_t *)tmp_buf2, size,
        //                           shift2);        
        //   ret = luna_add_q7_int8((const q7_t *)tmp_buf1, (q7_t *)tmp_buf2, (int8_t *)dst,
        //                          size, 0);
        // }
        // else
        // {
        //   ret = luna_scale_q7_int8((const q7_t *)src1, (scale1), (int8_t *)dst, size,
        //                          shift1);
        //   ret = luna_scale_q7_int8((const q7_t *)src2, (scale2), (int8_t *)src2, size,
        //                           shift2);        
        //   ret = luna_add_q7_int8((const q7_t *)dst, (q7_t *)src2, (int8_t *)dst,
        //                         size, 0);
        // }
        // if (y_is_psram)
        // {
        //   memcpy((void *)Y->dptr_, dst, size * sizeof(int8_t));
        // }
      } break;
      case Int16: {
        ret = luna_scale_q15_int16((const q15_t *)src1, (scale1), (int16_t *)dst,
                                   size, shift1);
        ret = luna_scale_q15_int16((const q15_t *)src2, (scale2), (int16_t *)src2,
                                   size, shift2);
        ret = luna_add_q15_int16((const q15_t *)dst, (q15_t *)src2,
                                 (int16_t *)dst, size, 0);
      } break;
      case Int32: {
        ret = luna_scale_q31_int32((const q31_t *)src1, (scale1), (int32_t *)dst,
                                   size, shift1);
        ret = luna_scale_q31_int32((const q31_t *)src2, (scale2), (int32_t *)src2,
                                   size, shift2);
        ret = luna_add_q31_int32((const q31_t *)dst, (q31_t *)src2,
                                 (int32_t *)dst, size, 0);
      } break;
      default:
        break;
    }
  }

  return ret;
}

#endif
