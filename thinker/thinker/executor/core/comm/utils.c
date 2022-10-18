#include "utils.h"

bool equalShape(tShape *src, tShape *dst) {
  if (src->ndim_ != dst->ndim_) {
    return false;
  }
  for (int32_t i = 0; i < src->ndim_; i++) {
    if (src->dims_[i] != dst->dims_[i]) {
      return false;
    }
  }
  return true;
}

size_t getShapeSize(tShape *shape) {
  uint64_t size = 1;
  for (int32_t i = 0; i < shape->ndim_; ++i) {
    size *= shape->dims_[i];
  }
  return size;
}

size_t getTensorSize(const tTensor *tensor) {
  uint64_t size = 1;
  if (4 == tensor->shape_.ndim_) {
    int32_t c = tensor->shape_.dims_[1];
    if (NHWC4 == tensor->layout_ || NC4HW4_T == tensor->layout_) {
      const size_t c_r4 = (tensor->shape_.dims_[1] + 3) / 4 * 4;
      size = tensor->shape_.dims_[0] * c_r4 * tensor->shape_.dims_[2] *
             tensor->shape_.dims_[3];
    } else {
      size = tensor->shape_.dims_[0] * c * tensor->shape_.dims_[2] *
             tensor->shape_.dims_[3];
    }
  } else {
    for (int32_t i = 0; i < tensor->shape_.ndim_; ++i) {
      size *= tensor->shape_.dims_[i];
    }
  }
  return size;
}

tShape calcStride(const tShape *shape) {
  tShape dst_shape;
  dst_shape.ndim_ = shape->ndim_;
  int32_t cumprod = 1;
  for (int32_t i = shape->ndim_ - 1; i >= 0; --i) {
    dst_shape.dims_[i] = (shape->dims_[i] > 1) ? cumprod : 0;
    cumprod *= shape->dims_[i];
  }
  return dst_shape;
}

void quant(float *src, int8_t *dst, int32_t size, int8_t scale) {
  float scalef = (float)(1 << scale);
  for (int32_t i = 0; i < size; ++i) {
    dst[i] = (int8_t)SATURATE_8BITS(floorf(scalef * src[i] + 0.5));
  }
}

void dequant8bit(int8_t *src, float *dst, int32_t size, int8_t scale) {
  float scale1 = 1.f / (1 << scale);
  for (int32_t i = size - 1; i >= 0; --i) {
    dst[i] = src[i] * scale1;
  }
}

void dequantU8bit(uint8_t *src, float *dst, int32_t size, int8_t scale) {
  float scale1 = 1.f / (1 << scale);
  for (int32_t i = size - 1; i >= 0; --i) {
    dst[i] = src[i] * scale1;
  }
}

void dequant32bit(int32_t *src, float *dst, int32_t size, int8_t scale) {
  float scale1 = 1.f / (1 << scale);
  for (int32_t i = size - 1; i >= 0; --i) {
    dst[i] = src[i] * scale1;
  }
}

#ifdef THINKER_USE_VENUS
#include "ops/venus/luna/opi_psram_cpy.h"

void lunaDmaInit(void) { dma_init(); }

void getWeightData(tDMA_List *dma_list, int32_t channel) {
  dma_wait_complete(0);
  if (dma_list->cout_ < dma_list->total_) {
    int32_t index = dma_list->cout_;
    tTensor *src = dma_list->dma_[index].src_tensors_;
    tTensor *dst = dma_list->dma_[index].dst_tensors_;
    int32_t size = dma_list->dma_[index].size_;
    dma_cpy_async(0, (void *)dst->dptr_, (void *)src->dptr_, size);
    dma_list->cout_++;
  }

  return;
}

#endif  // THINKER_USE_VENUS
