
#include "c_api/thinker_define.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "luna/luna_math.h"

static int32_t my_log2(float x) {
  int8_t *in_addr = (int8_t *)&x;
  uint32_t ix = (uint32_t)(*((uint32_t *)in_addr));
  uint32_t exp = (ix >> 23) & 0xFF;
  return (int32_t)(exp - 127);
}

static void luna_meanpool_para_init(PoolAttrs *attrs, s_conv_struct *conv_attrs,
                                    tTensor *X, tTensor *Y) {
  memset(conv_attrs, 0, sizeof(s_conv_struct));

  conv_attrs->input_c = X->shape_.dims_[1];
  conv_attrs->input_h = X->shape_.dims_[2];
  conv_attrs->input_w = X->shape_.dims_[3];
  conv_attrs->output_c = Y->shape_.dims_[1];
  conv_attrs->output_h = Y->shape_.dims_[2];
  conv_attrs->output_w = Y->shape_.dims_[3];
  conv_attrs->weight_h = attrs->kernel[0];
  conv_attrs->weight_w = attrs->kernel[1];
  conv_attrs->stride_h = attrs->stride[0];
  conv_attrs->stride_w = attrs->stride[1];
  conv_attrs->padding_h_up = attrs->pad[0];
  conv_attrs->padding_h_down = attrs->pad[2];
  conv_attrs->padding_w_left = attrs->pad[1];
  conv_attrs->padding_w_right = attrs->pad[3];
  conv_attrs->input_h_after_padding = conv_attrs->input_h +
                                      conv_attrs->padding_h_up +
                                      conv_attrs->padding_h_down;
  conv_attrs->input_w_after_padding = conv_attrs->input_w +
                                      conv_attrs->padding_w_left +
                                      conv_attrs->padding_w_right;
  conv_attrs->is_bias = 0;
  conv_attrs->pooling_type = PoolMethod_AVE;
}

int32_t avgpool2dint_luna(const tTensor *X, tTensor *Y, tTensor *Temp,
                          PoolAttrs *attrs) {
  int32_t ret = -1;
  if (Int8 == X->dtype_) {
    s_conv_struct pool_struct_;
    luna_meanpool_para_init(attrs, &pool_struct_, (tTensor *)X, Y);
    int32_t batch = X->shape_.dims_[0];
    int32_t in_batch_size =
        pool_struct_.input_c * pool_struct_.input_h * pool_struct_.input_w;
    int32_t ou_batch_size = pool_struct_.output_c * pool_struct_.output_h *
                            pool_struct_.output_w * (Y->dtype_ & 0xF);

    int32_t shift = 0;
    int32_t one_kernel_size = pool_struct_.weight_h * pool_struct_.weight_w;

    if (0 == (one_kernel_size & (one_kernel_size - 1))) {
      int16_t *p_tmp = (int16_t *)Temp->dptr_;
      shift = my_log2((float)one_kernel_size);
      for (int32_t n = 0; n < batch; n++) {
        int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
        int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
        ret = luna_mean_pooling_int16(p_in, (int16_t *)p_tmp, &pool_struct_);
        ret |= luna_scale_q15_int8(p_tmp, 1, p_out, ou_batch_size, shift);
      }
    } else {
      int32_t q_x = (int32_t)X->scale_;
      int32_t q_o = (int32_t)Y->scale_;
      int32_t *p_tmp1 = (int32_t *)Temp->dptr_;
      int32_t *p_tmp2 = (int32_t *)(p_tmp1 + ou_batch_size);
      for (int32_t n = 0; n < batch; n++) {
        int8_t *p_in = (int8_t *)X->dptr_ + n * in_batch_size;
        int8_t *p_out = (int8_t *)Y->dptr_ + n * ou_batch_size;
        ret = luna_mean_pooling_int16(p_in, (int16_t *)p_tmp1, &pool_struct_);
        ret |= luna_scale_q15_int32((int16_t *)p_tmp1, 1, p_tmp2, ou_batch_size,
                                    0);
        ret |= luna_memset(p_out, 1, ou_batch_size);
        ret |= luna_scale_q7_int32(p_out, one_kernel_size, p_tmp1,
                                   ou_batch_size, 0);
        ret |= luna_div_q31_int32(p_tmp2, q_x, p_tmp1, 0, p_tmp1, q_o,
                                  ou_batch_size);
        ret |= luna_scale_q31_int8(p_tmp1, 1, p_out, ou_batch_size, 0);
      }
    }
  }

  return ret;
}
