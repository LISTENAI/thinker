#ifndef _UPSAMPLEINT_LUNA_H_
#define _UPSAMPLEINT_LUNA_H_
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "core/comm/thinker_log.h"
#include "core/comm/utils.h"
#include "core/operator_attrs.h"
#include "luna/luna_math.h"

typedef struct _resize_w_table {
  int32_t w_int;
  uint32_t w_frac;
} resize_w_table_t;

static inline int32_t saturate_i16(int32_t val) {
  return (val <= -32768) ? -32768 : ((val >= 32767) ? 32767 : val);
}

static inline int32_t saturate_i8(int32_t val) {
  return (val <= -128) ? -128 : ((val >= 127) ? 127 : val);
}

static void gen_w_table(int32_t input_width, int32_t output_width,
                        resize_w_table_t *table) {
  int32_t w_lim = input_width - 1;
  int64_t add_per =
      (int64_t)(((((uint64_t)input_width) << 32) + (output_width >> 1)) /
                ((uint64_t)output_width));

  int64_t acc = 1 << 16;

  for (int32_t i = 0; i < output_width; ++i) {
    int32_t w_int = acc >> 32;
    uint32_t w_frac = (uint32_t)(acc) >> 17;
    if (w_int >= w_lim) {
      w_int = w_lim;
      w_frac = 0;
    }

    table[i].w_int = w_int;
    table[i].w_frac = w_frac;
    acc += add_per;
  }
}

static void resize_general(int8_t *output, const int8_t *input,
                           int32_t batch_size, int32_t channel_size,
                           int32_t input_height, int32_t input_width,
                           int32_t output_height, int32_t output_width,
                           int32_t *workspace) {
  int64_t h_add_per =
      (int64_t)(((((uint64_t)input_width) << 32) + (output_width >> 1)) /
                ((uint64_t)output_width));
  int32_t height_lim = input_height - 1;

  resize_w_table_t *w_control = (resize_w_table_t *)workspace;
  gen_w_table(input_width, output_width, w_control);

  for (int32_t n = 0; n < batch_size; ++n) {
    const int8_t *n_start =
        input + n * channel_size * input_height * input_width;
    for (int32_t c = 0; c < channel_size; ++c) {
      const int8_t *c_start = n_start + c * input_height * input_width;
      int64_t h_acc = 1 << 24;

      for (int32_t h = 0; h < output_height; ++h) {
        int32_t h_in_int = h_acc >> 32;

        uint8_t h_frac = ((uint8_t)(h_acc >> 25) & 127);

        h_acc += h_add_per;

        if (h_in_int >= height_lim) {
          h_in_int = height_lim;
          h_frac = 0;
        }

        const int8_t *row0 = c_start + h_in_int * input_width;
        const int8_t *row1 = row0 + (height_lim == h_in_int ? 0 : input_width);

        for (int32_t w = 0; w < output_width; ++w) {
          int32_t w_in_int = w_control[w].w_int;
          int32_t next_int = w_in_int + (input_width - 1 == w_in_int ? 0 : 1);

          int32_t x0x = (int32_t)row0[w_in_int] * (128 - h_frac) +
                        (int32_t)row1[w_in_int] * h_frac;
          int32_t x1x = (int32_t)row0[next_int] * (128 - h_frac) +
                        (int32_t)row1[next_int] * h_frac;

          int32_t x =
              x0x +
              ((int32_t)((((x1x - x0x) * w_control[w].w_frac) << 1) + 0x8000) >>
               16);

          output[w] = (int8_t)(saturate_i8((x + (1 << 6)) >> 7));
        }
        output += output_width;
      }
    }
  }
}

static void resize_2x(int8_t *output, const int8_t *src, int32_t batch_size,
                      int32_t channel_size, int32_t input_height,
                      int32_t input_width, int32_t output_height,
                      int32_t output_width) {
  for (int32_t n = 0; n < batch_size; ++n) {
    const int8_t *n_strat = src + n * input_height * input_width * channel_size;
    for (int32_t c = 0; c < channel_size; ++c) {
      const int8_t *c_strat = n_strat + c * input_height * input_width;
      for (int32_t h = 0; h < output_height - 1; ++h) {
        int32_t h_int = h >> 1;
        const int8_t *row0 = c_strat + h_int * input_width;
        const int8_t *row1 = row0 + (0 == (h & 0x1) ? 0 : input_width);

        for (int32_t w = 0; w < output_width - 1; ++w) {
          int32_t w_int = w >> 1;                               // div
          int32_t next_int = w_int + (0 == (w & 0x1) ? 0 : 1);  // Yes !=1  No
                                                                // =1
          int32_t add = (w >= (((input_width - 1) >> 2) << 3) ? 1 : 0);
          int32_t ou_value =
              ((((int32_t)row0[w_int] + (int32_t)row0[next_int]) >> 1) +
               (((int32_t)row1[w_int] + (int32_t)row1[next_int]) >> 1) + add) >>
              1;
          output[w] = saturate_i8(ou_value);
        }
        output[output_width - 1] = output[output_width - 2];
        output += output_width;
      }
      memcpy(output, output - output_width, output_width);
      output += output_width;
    }
  }
}

static void resize(int8_t *input, int8_t *output, int32_t *workspace,
                   int32_t batch_size, int32_t channel_size, int32_t input_h,
                   int32_t input_w, int32_t output_h, int32_t output_w) {
  float h_scale = (float)(output_h) / input_h;
  float w_scale = (float)(output_w) / input_w;

  if (1.0f == h_scale && 1.0f == w_scale) {
    uint32_t size = batch_size * channel_size * output_h * output_w;
    memcpy(output, input, size);
  } else if (2.0f == h_scale && 2.0f == w_scale) {
    resize_2x(output, input, batch_size, channel_size, input_h, input_w,
              output_h, output_w);
  } else {
    resize_general(output, input, batch_size, channel_size, input_h, input_w,
                   output_h, output_w, workspace);
  }
}

int32_t upsampleint_luna(const tTensor *X, tTensor *Y, tTensor *workspace) {
  int32_t ret = -1;

  if (X->dtype_ == Int8) {
    int8_t *input = (int8_t *)X->dptr_;
    int8_t *output = (int8_t *)Y->dptr_;
    int32_t *p_tmp = (int32_t *)workspace->dptr_;
    int32_t N = X->shape_.dims_[0];
    int32_t C = X->shape_.dims_[1];
    int32_t in_h = X->shape_.dims_[2];
    int32_t in_w = X->shape_.dims_[3];
    int32_t ou_h = Y->shape_.dims_[2];
    int32_t ou_w = Y->shape_.dims_[3];
    resize(input, output, p_tmp, N, C, in_h, in_w, ou_h, ou_w);
  }

  return ret;
}

#endif
