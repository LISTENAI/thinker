#ifndef __RELU_H__
#define __RELU_H__

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#ifdef THINKER_USE_NNBLAS
#include "nnblas/nnblas_op.h"
#define API_LIB(api) nnblas_##api
#else
#include "luna/luna_math.h"
#define API_LIB(api) luna_##api
#endif
#include "thinker_status.h"


/**
 * @brief Perform ReLU operation based on input and output data types
 * @param X Input tensor
 * @param Y Output tensor
 * @param Workspace Workspace tensor for data temporarily storage
 * @return Execution status
 */
tStatus relu_luna(tTensor *X, tTensor *Y, tTensor *Workspace) 
{
	int32_t ret = T_ERR_NO_IMPLEMENTED;
  uint32_t size = getShapeSize(&(X->shape_));
  uint32_t workspace_size = Workspace ? getShapeSize(&(Workspace->shape_)) : 0;
  int32_t shift = Y->scale_ - X->scale_;
  bool srcInPSRAM = (X->mem_.type_ != 2);
  bool dstInPSRAM = (Y->mem_.type_ != 2);

  if (Int8 == X->dtype_ && Int8 == Y->dtype_) {
    int8_t *src   = (int8_t *)X->dptr_;
    int8_t *dst   = (int8_t *)Y->dptr_;

    // Helper function to handle data type conversions and memory management
    if (srcInPSRAM) { // X is psram
      ret = API_LIB(memcpy_i8o8)((int8_t *)Workspace->dptr_, src, size);
      if (size > workspace_size)
        return T_ERR_NO_WORKSPACE;
      src = (int8_t *)Workspace->dptr_;
    }
    if (dstInPSRAM) {
      if (size > workspace_size)
        return T_ERR_NO_WORKSPACE;
      dst = (int8_t *)Workspace->dptr_;
    }

    ret = API_LIB(relu_i8o8)(src, dst, size, shift);
    if (dstInPSRAM) {
    	opi_psram_cpy_out((int8_t *)Y->dptr_, dst, size);
    }
  }
  else if (Int8 == X->dtype_ && Int16 == Y->dtype_) {
    int8_t *src   = (int8_t *)X->dptr_;
    int16_t *dst  = (int16_t *)Y->dptr_;
    uint32_t size = getShapeSize(&(X->shape_));

    if (srcInPSRAM) { // X is psram
      if (size > workspace_size)
        return T_ERR_NO_WORKSPACE;
      ret = API_LIB(memcpy_i8o8)((int8_t *)Workspace->dptr_, src, size);

      src = (int8_t *)Workspace->dptr_;
    }
    if (dstInPSRAM) {
      if (size > workspace_size)
        return T_ERR_NO_WORKSPACE;
      dst = 2 == X->mem_.type_? (int16_t *)Workspace->dptr_ : (int16_t *)Workspace->dptr_ + size;
    }
    ret = API_LIB(relu_i8o16)(src, dst, size, shift);

    if (dstInPSRAM) {
    	opi_psram_cpy_out((int8_t *)Y->dptr_, dst, size * 2);
    }
  }
  else if (Int8 == X->dtype_ && Int32 == Y->dtype_) {
    int8_t *src   = (int8_t *)X->dptr_;
    int32_t *dst  = (int32_t *)Y->dptr_;
    uint32_t size = getShapeSize(&(X->shape_));

    if (srcInPSRAM) { // X is psram
      if (size > workspace_size)
        return T_ERR_NO_WORKSPACE;
      ret = API_LIB(memcpy_i8o8)((int8_t *)Workspace->dptr_, src, size);

      src = (int8_t *)Workspace->dptr_;
    }
    if (dstInPSRAM) {
      if (size > workspace_size)
        return T_ERR_NO_WORKSPACE;
      dst = 2 == X->mem_.type_? (int32_t *)Workspace->dptr_ : (int32_t *)Workspace->dptr_ + size;
    }
    ret = API_LIB(relu_i8o32)(src, dst, size, shift);

    if (dstInPSRAM) {
    	opi_psram_cpy_out((int8_t *)Y->dptr_, dst, size * 4);
    }
  }
  else if (Int16 == X->dtype_ && Int8 == Y->dtype_) {
    int16_t *src   = (int16_t *)X->dptr_;
    int8_t *dst  = (int8_t *)Y->dptr_;
    uint32_t size = getShapeSize(&(X->shape_));

    if (srcInPSRAM) { // X is psram
      if (size > workspace_size)
        return T_ERR_NO_WORKSPACE;
      ret = API_LIB(memcpy_i8o8)((int8_t *)Workspace->dptr_, (int8_t*)src, size*2);

      src = (int16_t *)Workspace->dptr_;
    }
    if (dstInPSRAM) {
      if (size > workspace_size)
        return T_ERR_NO_WORKSPACE;
      dst = (int8_t *)Workspace->dptr_;
    }
    ret = API_LIB(relu_i16o8)(src, dst, size, shift);

    if (dstInPSRAM) {
    	opi_psram_cpy_out((int8_t *)Y->dptr_, dst, size);
    }
  }
  else if (Int16 == X->dtype_ && Int16 == Y->dtype_) {
    int16_t *src   = (int16_t *)X->dptr_;
    int16_t *dst  = (int16_t *)Y->dptr_;
    uint32_t size = getShapeSize(&(X->shape_));

    if (srcInPSRAM) { // X is psram
      if (size > workspace_size)
        return T_ERR_NO_WORKSPACE;
      ret = API_LIB(memcpy_i8o8)((int8_t *)Workspace->dptr_, (int8_t*)src, size*2);

      src = (int16_t *)Workspace->dptr_;
    }
    if (dstInPSRAM) {
      if (size > workspace_size)
        return T_ERR_NO_WORKSPACE;
      dst = (int16_t *)Workspace->dptr_;
    }
    ret = API_LIB(relu_i16o16)(src, dst, size, shift);

    if (dstInPSRAM) {
    	opi_psram_cpy_out((int8_t *)Y->dptr_, dst, 2*size);
    }
  }
  else if (Int16 == X->dtype_ && Int32 == Y->dtype_) {
    int16_t *src   = (int16_t *)X->dptr_;
    int32_t *dst  = (int32_t *)Y->dptr_;
    uint32_t size = getShapeSize(&(X->shape_));

    if (srcInPSRAM) { // X is psram
      if (size > workspace_size)
        return T_ERR_NO_WORKSPACE;
      ret = API_LIB(memcpy_i8o8)((int8_t *)Workspace->dptr_, (int8_t*)src, size*2);

      src = (int16_t *)Workspace->dptr_;
    }
    if (dstInPSRAM) {
      if (size > workspace_size)
        return T_ERR_NO_WORKSPACE;
      dst = 2 == X->mem_.type_? (int32_t *)Workspace->dptr_ : (int32_t *)Workspace->dptr_ + size;
    }
    ret = API_LIB(relu_i16o32)(src, dst, size, shift);

    if (dstInPSRAM) {
    	opi_psram_cpy_out((int8_t *)Y->dptr_, dst, 4*size);
    }
  }
  else if (Int32 == X->dtype_ && Int8 == Y->dtype_) {
    int32_t *src  = (int32_t *)X->dptr_;
    int8_t *dst   = (int8_t *)Y->dptr_;
    uint32_t size = getShapeSize(&(X->shape_));

    if (srcInPSRAM) { // X is psram
      ret = API_LIB(memcpy_i8o8)((int8_t *)Workspace->dptr_, (int8_t *)src, size * 4);
      src = (int32_t *)Workspace->dptr_;
    }
    if (dstInPSRAM)
      dst = (int8_t *)Workspace->dptr_;

    ret = API_LIB(relu_i32o8)(src, dst, size, shift);

    if (dstInPSRAM) {
    	opi_psram_cpy_out((int8_t *)Y->dptr_, dst, size);
    }
  }
  else if (Int32 == X->dtype_ && Int16 == Y->dtype_) {
    int32_t *src   = (int32_t *)X->dptr_;
    int16_t *dst  = (int16_t *)Y->dptr_;
    uint32_t size = getShapeSize(&(X->shape_));

    if (srcInPSRAM) { // X is psram
      if (size > workspace_size)
        return T_ERR_NO_WORKSPACE;
      ret = API_LIB(memcpy_i8o8)((int8_t *)Workspace->dptr_, (int8_t*)src, size*4);

      src = (int32_t *)Workspace->dptr_;
    }
    if (dstInPSRAM) {
      if (size > workspace_size)
        return T_ERR_NO_WORKSPACE;
      dst = (int16_t *)Workspace->dptr_;
    }
    ret = API_LIB(relu_i32o16)(src, dst, size, shift);

    if (dstInPSRAM) {
    	opi_psram_cpy_out((int8_t *)Y->dptr_, dst, 2*size);
    }
  }
  else if (Int32 == X->dtype_ && Int32 == Y->dtype_) {
    int32_t *src  = (int32_t *)X->dptr_;
    int32_t *dst  = (int32_t *)Y->dptr_;
    uint32_t size = getShapeSize(&(X->shape_));

    if (srcInPSRAM) { // X is psram
      ret = API_LIB(memcpy_i8o8)((int8_t *)Workspace->dptr_, (int8_t *)src, size * 4);
      src = (int32_t *)Workspace->dptr_;
    }
    if (dstInPSRAM)
      dst = (int32_t *)Workspace->dptr_;

    ret = API_LIB(relu_i32o32)(src, dst, size, shift);

    if (dstInPSRAM) {
    	opi_psram_cpy_out((int8_t *)Y->dptr_, dst, size * 4);
    }
  }
  else {
    return T_ERR_INVALID_DATATYPE;
  }

	return (tStatus)ret;
}
#endif
