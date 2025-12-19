#include "utils.h"

// Check if two shapes are equal
bool equalShape(tShape *src, tShape *dst) {
    if (src->ndim_ != dst->ndim_) return false;
    for (int32_t i = 0; i < src->ndim_; i++) {
        if (src->dims_[i] != dst->dims_[i]) return false;
    }
    return true;
}

// Get total size of a shape
size_t getShapeSize(tShape *shape) {
    uint64_t size = 1;
    for (int32_t i = 0; i < shape->ndim_; ++i) {
        size *= shape->dims_[i];
    }
    return size;
}

// Get total size of a tensor considering layout
size_t getTensorSize(const tTensor *tensor) {
    uint64_t size = 1;
    if (4 == tensor->shape_.ndim_) {
        int32_t c = tensor->shape_.dims_[1];
        if (NHWC4 == tensor->layout_ || NC4HW4_T == tensor->layout_) {
            const size_t c_r4 = (tensor->shape_.dims_[1] + 3) / 4 * 4;
            size = tensor->shape_.dims_[0] * c_r4 * tensor->shape_.dims_[2] * tensor->shape_.dims_[3];
        } else {
            size = tensor->shape_.dims_[0] * c * tensor->shape_.dims_[2] * tensor->shape_.dims_[3];
        }
    } else {
        for (int32_t i = 0; i < tensor->shape_.ndim_; ++i) {
            size *= tensor->shape_.dims_[i];
        }
    }
    return size;
}

// Calculate strides for a shape
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

// Quantize float values to int8
void quant(float *src, int8_t *dst, int32_t size, int8_t scale) {
    float scalef = (float)(1 << scale);
    for (int32_t i = 0; i < size; ++i) {
        dst[i] = (int8_t)SATURATE_8BITS(floorf(scalef * src[i] + 0.5));
    }
}

// Dequantize int8 values to float
void dequant8bit(int8_t *src, float *dst, int32_t size, int8_t scale) {
    float scale1 = 1.f / (1 << scale);
    for (int32_t i = size - 1; i >= 0; --i) {
        dst[i] = src[i] * scale1;
    }
}

// Dequantize uint8 values to float
void dequantU8bit(uint8_t *src, float *dst, int32_t size, int8_t scale) {
    float scale1 = 1.f / (1 << scale);
    for (int32_t i = size - 1; i >= 0; --i) {
        dst[i] = src[i] * scale1;
    }
}

// Dequantize int32 values to float
void dequant32bit(int32_t *src, float *dst, int32_t size, int8_t scale) {
    float scale1 = 1.f / (1 << scale);
    for (int32_t i = size - 1; i >= 0; --i) {
        dst[i] = src[i] * scale1;
    }
}

// Convert 4-bit to 8-bit values with sign extension
void convert_4bitto8bit(int8_t *dst, int8_t *src, int32_t size) {
    for (int32_t i = 0; i < size / 2; i++) {
        int8_t high = (src[i] >> 4) & 0x0F;
        if ((high & 0x08) != 0) high |= 0xF0;
        dst[2 * i + 1] = high;

        int8_t low = src[i] & 0x0F;
        if ((low & 0x08) != 0) low |= 0xF0;
        dst[2 * i] = low;
    }
}

// Convert 4-bit to 32-bit values with sign extension
void convert_4bitto32bit(int32_t *dst, int8_t *src, int32_t size) {
    for (int32_t i = 0; i < size / 2; i++) {
        int8_t high = (src[i] >> 4) & 0x0F;
        if ((high & 0x08) != 0) high |= 0xF0;
        dst[2 * i + 1] = high;

        int8_t low = src[i] & 0x0F;
        if ((low & 0x08) != 0) low |= 0xF0;
        dst[2 * i] = low;
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
}
#elif THINKER_USE_ARCS
#include "ops/arcs/luna/opi_psram_cpy.h"

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
}

void cpu_memcpy(void *dst, const void *src, size_t size) {
#if !(defined(WIN32) || defined(linux))
    if ((src != dst) && (0 != size)) memcpy(dst, src, size);
    if (((uint32_t)dst & 0x28000000) == 0x28000000) HAL_FlushDCache_by_Addr(dst, size);
#else
    memcpy(dst, src, size);
#endif
}

#elif THINKER_USE_VENUSA
#if THINKER_USE_MTQ
#include "../ops/venusA/luna/luna_math.h"
#include "../ops/venusA/luna/luna_mtq_math.h"
#endif
#include "../ops/venusA/luna/include/cache.h"
#include "../ops/venusA/luna/luna_misc_math.h"

#if THINKER_USE_MTQ
extern uint32_t luna_api_split_cnn[];
extern uint32_t luna_api_split_depthwise[];
extern uint32_t luna_api_split_pool[];
extern uint32_t luna_api_split_deconv[];
#endif

#define ALG_DMA_CH      5
static int32_t g_dma_start_id_ = 0;

void dma_wait_complete(int chn) {
    if (0 < g_dma_start_id_) {
        g_dma_start_id_--;
        luna_gpdma_wait(chn);
    }
    return;
}

void dma_cpy_async(int chn, void *dst, void *src, int32_t size) {
    if (0 < g_dma_start_id_) {
        g_dma_start_id_--;
        luna_gpdma_wait(chn);
    }
    if (0 == g_dma_start_id_) {
        luna_gpdma_start(chn, dst, src, size);
        g_dma_start_id_++;
    }
}

void opi_psram_cpy_out(void *dst, void *src, int32_t size) {
    dma_cpy_async(ALG_DMA_CH, dst, src, size);
    dma_wait_complete(ALG_DMA_CH);
}

void getWeightData(tDMA_List *dma_list, int32_t channel) {
    dma_wait_complete(ALG_DMA_CH);
    if (dma_list->cout_ < dma_list->total_) {
        int32_t index = dma_list->cout_;
        tTensor *src = dma_list->dma_[index].src_tensors_;
        tTensor *dst = dma_list->dma_[index].dst_tensors_;
        int32_t size = dma_list->dma_[index].size_;
        dma_cpy_async(ALG_DMA_CH, (int8_t *)dst->dptr_, (int8_t *)src->dptr_, size);
        dma_list->cout_++;
    }
}

void cpu_memcpy(void *dst, const void *src, size_t size) {
#if !(defined(WIN32) || defined(linux))
    if ((src != dst) && (0 != size)) memcpy(dst, src, size);
    if (((uint32_t)dst & 0x28000000) == 0x28000000) HAL_FlushDCache_by_Addr(dst, size);
#else
    memcpy(dst, src, size);
#endif
}

#if THINKER_USE_MTQ
luna_mtq_sq_elem_t *g_sq_addr_user_ch = NULL;
luna_mtq_cq_elem_t *g_cq_addr_user_ch = NULL;
uint32_t g_submit_pos = 0;
uint32_t g_param_size = 0;
int8_t* g_param_addr = NULL;
static uint32_t g_last_counter;

int32_t luna_execute_cmd_hook_for_get_list_length(const uint32_t *api, void* param, uint32_t param_size, void* userdata) {
    g_submit_pos += 1;
    g_param_size += param_size;
    
    if (api == luna_api_split_cnn ||
        api == luna_api_split_depthwise ||
        api == luna_api_split_pool ||
        api == luna_api_split_deconv) {
        g_param_size += sizeof(luna_cnn_static_para_t);
    }
    return 0;
}

int32_t luna_execute_cmd_hook_for_build_list(const uint32_t *api, void* param, uint32_t param_size, void* userdata) {
    const void* p_memset_api = api;
    void* p_param_s = g_param_addr;
    g_param_addr += param_size;
    memcpy(p_param_s, param, param_size);
    
    if (api == luna_api_split_cnn ||
        api == luna_api_split_depthwise ||
        api == luna_api_split_pool ||
        api == luna_api_split_deconv) {
        void* p_static_param = ((luna_cnn_para_t *)param)->cnn_static_para;
        void* p_static_param_s = g_param_addr;
        g_param_addr += sizeof(luna_cnn_static_para_t);
        memcpy(p_static_param_s, p_static_param, sizeof(luna_cnn_static_para_t));
        ((volatile luna_cnn_para_t *)p_param_s)->cnn_static_para = p_static_param_s;
    }
    
    *((volatile uint32_t *)p_param_s) |= (0x1 << 31);
    
    luna_mtq_sq_elem_t *p_sq_elem = &(g_sq_addr_user_ch[g_submit_pos]);
    p_sq_elem->task_type = MTQ_TASK_TYPE_LUNA_TASK;
    p_sq_elem->mark_idx = MTQ_MARK_IDX_IOWR_OVER;
    p_sq_elem->blocking_type = MTQ_BLOCKING_TYPE_BLOCKING_TASK;
    p_sq_elem->reture_cq_bypass = 0;
    p_sq_elem->op_interrupt_enable = 0;
    p_sq_elem->reserved = 0;
    p_sq_elem->op_id = g_submit_pos;
    p_sq_elem->task_base_addr.task_base_addr = (uint32_t)p_memset_api;
    p_sq_elem->task_param = (uint32_t)p_param_s;
    
    luna_mtq_cq_elem_t *p_cq_elem = &(g_cq_addr_user_ch[g_submit_pos]);
    memset(p_cq_elem, 0, sizeof(luna_mtq_cq_elem_t));
    
    g_submit_pos += 1;
    return 0;
}
#endif
#endif

#ifdef WIN32
double tick_count(void) {
    struct timespec tv;
    timespec_get(&tv, 1);
    return (double)(((int64_t)tv.tv_sec) * 1000 +
                   ((int64_t)tv.tv_nsec) / 1000000.0);
}
#elif defined(linux)
#include <sys/time.h>
double tick_count(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)(tv.tv_sec * 1000 * 1000 + tv.tv_usec) / 1000.0;
}
#else
uint64_t tick_count(void) {
    return (uint64_t)__get_rv_cycle();
}
#endif