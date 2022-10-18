#undef __OP__
#define __OP__ Resize
#include <math.h>

#include "core/comm/thinker_log.h"
#include "core/operator_attrs.h"
#include "core/operator_register.h"

typedef enum coordinate_transformation_mode {
  khalf_pixel = 0,
  kpytorch_half_pixel,
  kalign_corners,
  kasymmetric,
  ktf_crop_and_resize
} ctmode;

typedef enum {
  knearnest = 0,
  klinear,
  kcubic,
} ResizeMode;

typedef enum {
  kround_prefer_floor = 0,
  kround_prefer_ceil,
  kfloor,
  kceil
} nearestMode;

enum resize_io {
  kData = 0,
  kRio,
  kScales,
  kSizes,
  kOut,
};

float GetCoordinateFunc(int32_t x, ctmode mode, float scale,
                        int32_t length_original, int32_t length_resized,
                        int32_t start_x, int32_t end_x) {
  float x_original;
  float x_resized = (float)x;
  switch (mode) {
    case khalf_pixel:
      x_original = (x_resized + 0.5) / scale - 0.5;
      break;
    case kpytorch_half_pixel:
      x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0;
      break;
    case kalign_corners:
      // bug: 当length_resized为1时
      x_original = x_resized * (length_original - 1) / (length_resized - 1);
      break;
    case kasymmetric:
      x_original = x_resized / scale;
      break;
    case ktf_crop_and_resize:
      x_original = length_resized > 1
                       ? start_x * (length_original - 1) +
                             x_resized * (end_x - start_x) *
                                 (length_original - 1) / (length_resized - 1)
                       : 0.5 * (start_x + end_x) * (length_original - 1);
      break;
    default:
      THINKER_LOG_FATAL("Resize: Unsupported coordinate_transformation_mode!");
      break;
  }
  return x_original;
}

int32_t GetNearestPixelFunc(float x, nearestMode nmode) {
  switch (nmode) {
    case kround_prefer_ceil:
      return (int32_t)round(x);
    case kfloor:
      return (int32_t)floor(x);
    case kceil:
      return (int32_t)ceil(x);
    case kround_prefer_floor:
      if ((int32_t)x + 0.5 == x)
        return (int32_t)x;
      else
        return (int32_t)round(x);
    default:
      THINKER_LOG_FATAL("Resize: Unsupported nearestMode!");
      break;
  }
}

int32_t Resize_nearest_float4d(float *input, float *output, float *scale,
                               int32_t *dims_original, int32_t *dims_resized,
                               int32_t *roi, ctmode ctm, nearestMode nmode) {
  int32_t scaleh_i = (int32_t)scale[2];
  int32_t scalew_i = (int32_t)scale[3];
  if (ctm == kasymmetric && nmode == kfloor && scalew_i == scaleh_i &&
      scalew_i == scale[3] && scaleh_i == scale[2] && scalew_i > 1) {
    int32_t b_c = dims_resized[0] * dims_resized[1];
    int32_t out_h = dims_resized[2], in_h = dims_original[2];
    int32_t out_w = dims_resized[3], in_w = dims_original[3];
    for (int32_t b = 0; b < b_c; ++b) {
      const float *in_dptr = input + b * in_h * in_w;
      float *out_dptr = output + b * out_h * out_w;
      for (int32_t h = 0; h < out_h; ++h) {
        for (int32_t w = 0; w < out_w; ++w) {
          out_dptr[h * out_w + w] = in_dptr[h / scaleh_i * in_w + w / scaleh_i];
        }
      }
    }
  } else if (ctm == khalf_pixel && nmode == kfloor) {
    int32_t b_c = dims_resized[0] * dims_resized[1];
    for (int32_t b = 0; b < b_c; ++b) {
      for (int32_t h = 0; h < dims_resized[2]; ++h) {
        float th = ((float)h + 0.5) / scale[2] - 0.5;
        int32_t h_original = (int32_t)floor(th);
        if (h_original < 0) {
          h_original = 0;
        } else if (h_original > dims_original[2] - 1) {
          h_original = dims_original[2] - 1;
        }
        for (int32_t w = 0; w < dims_resized[3]; ++w) {
          float tw = ((float)w + 0.5) / scale[3] - 0.5;
          int32_t w_original = (int32_t)floor(tw);
          int32_t W = dims_original[3];
          if (w_original < 0)
            w_original = 0;
          else if (w_original >= W)
            w_original = W - 1;
          output[0] = input[w_original + h_original * W];
          ++output;
        }
      }
      input += dims_original[2] * dims_original[3];
    }
  } else if (ctm == kpytorch_half_pixel && nmode == kfloor) {
    int32_t b_c = dims_resized[0] * dims_resized[1];
    for (int32_t b = 0; b < b_c; ++b) {
      for (int32_t h = 0; h < dims_resized[2]; ++h) {
        float th = dims_resized[2] > 1 ? ((float)h + 0.5) / scale[2] - 0.5 : 0;
        int32_t h_original = (int32_t)floor(th);
        if (h_original < 0) {
          h_original = 0;
        } else if (h_original > dims_original[2] - 1) {
          h_original = dims_original[2] - 1;
        }
        for (int32_t w = 0; w < dims_resized[3]; ++w) {
          float tw =
              dims_resized[3] > 1 ? ((float)w + 0.5) / scale[3] - 0.5 : 0;
          int32_t w_original = (int32_t)floor(tw);
          int32_t W = dims_original[3];
          if (w_original < 0)
            w_original = 0;
          else if (w_original >= W)
            w_original = W - 1;
          output[0] = input[w_original + h_original * W];
          ++output;
        }
      }
      input += dims_original[2] * dims_original[3];
    }
  } else if (ctm == ktf_crop_and_resize && nmode == kfloor) {
    int32_t b_c = dims_resized[0] * dims_resized[1];
    for (int32_t b = 0; b < b_c; ++b) {
      for (int32_t h = 0; h < dims_resized[2]; ++h) {
        float th = dims_resized[2] > 1
                       ? roi[2] * (dims_original[2] - 1) +
                             (float)h * (roi[6] - roi[2]) *
                                 (dims_original[2] - 1) / (dims_resized[2] - 1)
                       : 0.5 * (roi[2] + roi[6]) * (dims_resized[2] - 1);
        int32_t h_original = (int32_t)floor(th);
        if (h_original < 0) {
          h_original = 0;
        } else if (h_original > dims_original[2] - 1) {
          h_original = dims_original[2] - 1;
        }
        for (int32_t w = 0; w < dims_resized[3]; ++w) {
          float tw = dims_resized[3] > 1
                         ? roi[3] * (dims_original[3] - 1) +
                               (float)w * (roi[7] - roi[3]) *
                                   (dims_original[3] - 1) /
                                   (dims_resized[3] - 1)
                         : 0.5 * (roi[3] + roi[7]) * (dims_resized[3] - 1);
          int32_t w_original = (int32_t)floor(tw);
          int32_t W = dims_original[3];
          if (w_original < 0)
            w_original = 0;
          else if (w_original >= W)
            w_original = W - 1;
          output[0] = input[w_original + h_original * W];
          ++output;
        }
      }
      input += dims_original[2] * dims_original[3];
    }
  } else if (ctm == kalign_corners && nmode == kfloor) {
    int32_t b_c = dims_resized[0] * dims_resized[1];
    for (int32_t b = 0; b < b_c; ++b) {
      for (int32_t h = 0; h < dims_resized[2]; ++h) {
        float th = (float)h * (dims_original[2] - 1) / (dims_resized[2] - 1);
        int32_t h_original = (int32_t)floor(th);

        if (h_original < 0) {
          h_original = 0;
        } else if (h_original > dims_original[2] - 1) {
          h_original = dims_original[2] - 1;
        }
        for (int32_t w = 0; w < dims_resized[3]; ++w) {
          float tw = (float)w * (dims_original[3] - 1) / (dims_resized[3] - 1);
          ;
          int32_t w_original = (int32_t)floor(tw);
          int32_t W = dims_original[3];
          if (w_original < 0)
            w_original = 0;
          else if (w_original >= W)
            w_original = W - 1;
          output[0] = input[w_original + h_original * W];
          ++output;
        }
      }
      input += dims_original[2] * dims_original[3];
    }
  } else {
    for (int32_t n = 0; n < dims_resized[0]; ++n) {
      for (int32_t c = 0; c < dims_resized[1]; ++c) {
        for (int32_t h = 0; h < dims_resized[2]; ++h) {
          float th = GetCoordinateFunc(h, ctm, scale[2], dims_original[2],
                                       dims_resized[2], roi[2], roi[6]);
          int32_t h_original = GetNearestPixelFunc(th, nmode);
          if (h_original < 0)
            h_original = 0;
          else if (h_original > dims_original[2] - 1)
            h_original = dims_original[2] - 1;
          for (int32_t w = 0; w < dims_resized[3]; ++w) {
            float tw = GetCoordinateFunc(w, ctm, scale[3], dims_original[3],
                                         dims_resized[3], roi[3], roi[7]);
            int32_t w_original = GetNearestPixelFunc(tw, nmode);
            int32_t W = dims_original[3];
            if (w_original < 0)
              w_original = 0;
            else if (w_original >= W)
              w_original = W - 1;
            output[0] = input[w_original + h_original * W];
            ++output;
          }
        }
        input += dims_original[2] * dims_original[3];
      }
    }
  }

  return 0;
}

int32_t Resize_nearest_float3d(float *input, float *output, float *scale,
                               int32_t *dims_original, int32_t *dims_resized,
                               int32_t *roi, ctmode ctm, nearestMode nmode) {
  for (int32_t n = 0; n < dims_resized[0]; ++n) {
    for (int32_t c = 0; c < dims_resized[1]; ++c) {
      for (int32_t h = 0; h < dims_resized[2]; ++h) {
        float th = GetCoordinateFunc(h, ctm, scale[2], dims_original[2],
                                     dims_resized[2], roi[2], roi[6]);
        int32_t h_original = GetNearestPixelFunc(th, nmode);
        if (h_original < 0)
          h_original = 0;
        else if (h_original > dims_original[2] - 1)
          h_original = dims_original[2] - 1;

        output[0] = input[h_original];
        ++output;
      }
      input += dims_original[2];
    }
  }
  return 0;
}
int32_t Resize_linear_float(float *input, float *output, float *scale,
                            int32_t *dims_original, int32_t *dims_resized,
                            int32_t *roi, ctmode ctm) {
  if (ctm == kasymmetric) {
    int32_t b_c = dims_resized[0] * dims_resized[1];
    for (int32_t b = 0; b < b_c; ++b) {
      for (int32_t h = 0; h < dims_resized[2]; ++h) {
        for (int32_t w = 0; w < dims_resized[3]; ++w) {
          float h_original = (float)h / scale[2];
          float w_original = (float)w / scale[3];
          float x = w_original, y = h_original;
          int32_t x0, x1, y0, y1;
          // 如果值在[0,H-1]或[0,W-1]之外
          if (x < 0)
            x0 = x1 = 0;
          else if (x > dims_original[3] - 1)
            x0 = x1 = dims_original[3] - 1;
          else {
            x0 = (int32_t)w_original;
            x1 = x0 + 1;
          }
          if (y < 0)
            y0 = y1 = 0;
          else if (y > dims_original[2] - 1)
            y0 = y1 = dims_original[2] - 1;
          else {
            y0 = (int32_t)h_original;
            y1 = y0 + 1;
          }
          /**
           * H(x0,y1)    G(x1,y1)
           *      X(x,y)
           * I(x0,y0)    J(x1,y0)
           **/
          float wx = x - x0, wy = y - y0;
          float wI = (1 - wy) * (1 - wx);
          float wJ = (1 - wy) * wx;
          float wH = wy * (1 - wx);
          float wG = wx * wy;
          int32_t W = dims_original[3];
          // todo: 加上exclude_outside系数
          float value = wI * input[x0 + y0 * W] + wJ * input[x1 + y0 * W] +
                        wH * input[x0 + y1 * W] + wG * input[x1 + y1 * W];
          output[0] = value;
          ++output;
        }
        input += dims_original[2] * dims_original[3];
      }
    }
  } else if (ctm == khalf_pixel) {
    int32_t b_c = dims_resized[0] * dims_resized[1];
    for (int32_t b = 0; b < b_c; ++b) {
      for (int32_t h = 0; h < dims_resized[2]; ++h) {
        for (int32_t w = 0; w < dims_resized[3]; ++w) {
          float h_original = ((float)h + 0.5) / scale[2] - 0.5;
          float w_original = ((float)w + 0.5) / scale[3] - 0.5;
          float x = w_original, y = h_original;
          int32_t x0, x1, y0, y1;
          // 如果值在[0,H-1]或[0,W-1]之外
          if (x < 0)
            x0 = x1 = 0;
          else if (x > dims_original[3] - 1)
            x0 = x1 = dims_original[3] - 1;
          else {
            x0 = (int32_t)w_original;
            x1 = x0 + 1;
          }
          if (y < 0)
            y0 = y1 = 0;
          else if (y > dims_original[2] - 1)
            y0 = y1 = dims_original[2] - 1;
          else {
            y0 = (int32_t)h_original;
            y1 = y0 + 1;
          }
          /**
           * H(x0,y1)    G(x1,y1)
           *      X(x,y)
           * I(x0,y0)    J(x1,y0)
           **/
          float wx = x - x0, wy = y - y0;
          float wI = (1 - wy) * (1 - wx);
          float wJ = (1 - wy) * wx;
          float wH = wy * (1 - wx);
          float wG = wx * wy;
          int32_t W = dims_original[3];
          // todo: 加上exclude_outside系数
          float value = wI * input[x0 + y0 * W] + wJ * input[x1 + y0 * W] +
                        wH * input[x0 + y1 * W] + wG * input[x1 + y1 * W];
          output[0] = value;
          ++output;
        }
        input += dims_original[2] * dims_original[3];
      }
    }
  } else if (ctm == kpytorch_half_pixel) {
    int32_t b_c = dims_resized[0] * dims_resized[1];
    for (int32_t b = 0; b < b_c; ++b) {
      for (int32_t h = 0; h < dims_resized[2]; ++h) {
        for (int32_t w = 0; w < dims_resized[3]; ++w) {
          float h_original =
              dims_resized[2] > 1 ? ((float)h + 0.5) / scale[2] - 0.5 : 0;
          float w_original =
              dims_resized[3] > 1 ? ((float)w + 0.5) / scale[3] - 0.5 : 0;
          float x = w_original, y = h_original;
          int32_t x0, x1, y0, y1;
          // 如果值在[0,H-1]或[0,W-1]之外
          if (x < 0)
            x0 = x1 = 0;
          else if (x > dims_original[3] - 1)
            x0 = x1 = dims_original[3] - 1;
          else {
            x0 = (int32_t)w_original;
            x1 = x0 + 1;
          }
          if (y < 0)
            y0 = y1 = 0;
          else if (y > dims_original[2] - 1)
            y0 = y1 = dims_original[2] - 1;
          else {
            y0 = (int32_t)h_original;
            y1 = y0 + 1;
          }
          /**
           * H(x0,y1)    G(x1,y1)
           *      X(x,y)
           * I(x0,y0)    J(x1,y0)
           **/
          float wx = x - x0, wy = y - y0;
          float wI = (1 - wy) * (1 - wx);
          float wJ = (1 - wy) * wx;
          float wH = wy * (1 - wx);
          float wG = wx * wy;
          int32_t W = dims_original[3];
          // todo: 加上exclude_outside系数
          float value = wI * input[x0 + y0 * W] + wJ * input[x1 + y0 * W] +
                        wH * input[x0 + y1 * W] + wG * input[x1 + y1 * W];
          output[0] = value;
          ++output;
        }
        input += dims_original[2] * dims_original[3];
      }
    }
  } else if (ctm == kalign_corners) {
    int32_t b_c = dims_resized[0] * dims_resized[1];
    for (int32_t b = 0; b < b_c; ++b) {
      for (int32_t h = 0; h < dims_resized[2]; ++h) {
        for (int32_t w = 0; w < dims_resized[3]; ++w) {
          float h_original =
              (float)h * (dims_original[2] - 1) / (dims_original[2] - 1);
          float w_original =
              (float)w * (dims_original[3] - 1) / (dims_original[3] - 1);
          float x = w_original, y = h_original;
          int32_t x0, x1, y0, y1;
          // 如果值在[0,H-1]或[0,W-1]之外
          if (x < 0)
            x0 = x1 = 0;
          else if (x > dims_original[3] - 1)
            x0 = x1 = dims_original[3] - 1;
          else {
            x0 = (int32_t)w_original;
            x1 = x0 + 1;
          }
          if (y < 0)
            y0 = y1 = 0;
          else if (y > dims_original[2] - 1)
            y0 = y1 = dims_original[2] - 1;
          else {
            y0 = (int32_t)h_original;
            y1 = y0 + 1;
          }
          /**
           * H(x0,y1)    G(x1,y1)
           *      X(x,y)
           * I(x0,y0)    J(x1,y0)
           **/
          float wx = x - x0, wy = y - y0;
          float wI = (1 - wy) * (1 - wx);
          float wJ = (1 - wy) * wx;
          float wH = wy * (1 - wx);
          float wG = wx * wy;
          int32_t W = dims_original[3];
          // todo: 加上exclude_outside系数
          float value = wI * input[x0 + y0 * W] + wJ * input[x1 + y0 * W] +
                        wH * input[x0 + y1 * W] + wG * input[x1 + y1 * W];
          output[0] = value;
          ++output;
        }
        input += dims_original[2] * dims_original[3];
      }
    }
  } else if (ctm == ktf_crop_and_resize) {
    int32_t b_c = dims_resized[0] * dims_resized[1];
    for (int32_t b = 0; b < b_c; ++b) {
      for (int32_t h = 0; h < dims_resized[2]; ++h) {
        for (int32_t w = 0; w < dims_resized[3]; ++w) {
          float h_original =
              dims_resized[2] > 1
                  ? roi[2] * (dims_original[2] - 1) +
                        (float)h * (roi[6] - roi[2]) * (dims_original[2] - 1) /
                            (dims_resized[2] - 1)
                  : 0.5 * (roi[2] + roi[6]) * (dims_original[2] - 1);

          float w_original =
              dims_resized[3] > 1
                  ? roi[3] * (dims_original[3] - 1) +
                        (float)w * (roi[7] - roi[3]) * (dims_original[3] - 1) /
                            (dims_resized[3] - 1)
                  : 0.5 * (roi[3] + roi[7]) * (dims_original[3] - 1);
          float x = w_original, y = h_original;
          int32_t x0, x1, y0, y1;
          // 如果值在[0,H-1]或[0,W-1]之外
          if (x < 0)
            x0 = x1 = 0;
          else if (x > dims_original[3] - 1)
            x0 = x1 = dims_original[3] - 1;
          else {
            x0 = (int32_t)w_original;
            x1 = x0 + 1;
          }
          if (y < 0)
            y0 = y1 = 0;
          else if (y > dims_original[2] - 1)
            y0 = y1 = dims_original[2] - 1;
          else {
            y0 = (int32_t)h_original;
            y1 = y0 + 1;
          }
          /**
           * H(x0,y1)    G(x1,y1)
           *      X(x,y)
           * I(x0,y0)    J(x1,y0)
           **/
          float wx = x - x0, wy = y - y0;
          float wI = (1 - wy) * (1 - wx);
          float wJ = (1 - wy) * wx;
          float wH = wy * (1 - wx);
          float wG = wx * wy;
          int32_t W = dims_original[3];
          // todo: 加上exclude_outside系数
          float value = wI * input[x0 + y0 * W] + wJ * input[x1 + y0 * W] +
                        wH * input[x0 + y1 * W] + wG * input[x1 + y1 * W];
          output[0] = value;
          ++output;
        }
        input += dims_original[2] * dims_original[3];
      }
    }
  } else {
    for (int32_t n = 0; n < dims_resized[0]; ++n) {
      for (int32_t c = 0; c < dims_resized[1]; ++c) {
        for (int32_t h = 0; h < dims_resized[2]; ++h)
          for (int32_t w = 0; w < dims_resized[3]; ++w) {
            float h_original =
                GetCoordinateFunc(h, ctm, scale[2], dims_original[2],
                                  dims_resized[2], roi[2], roi[6]);
            float w_original =
                GetCoordinateFunc(w, ctm, scale[3], dims_original[3],
                                  dims_resized[3], roi[3], roi[7]);
            float x = w_original, y = h_original;
            int32_t x0, x1, y0, y1;
            // 如果值在[0,H-1]或[0,W-1]之外
            if (x < 0)
              x0 = x1 = 0;
            else if (x > dims_original[3] - 1)
              x0 = x1 = dims_original[3] - 1;
            else {
              x0 = (int32_t)w_original;
              x1 = x0 + 1;
            }
            if (y < 0)
              y0 = y1 = 0;
            else if (y > dims_original[2] - 1)
              y0 = y1 = dims_original[2] - 1;
            else {
              y0 = (int32_t)h_original;
              y1 = y0 + 1;
            }
            /**
             * H(x0,y1)    G(x1,y1)
             *      X(x,y)
             * I(x0,y0)    J(x1,y0)
             **/
            float wx = x - x0, wy = y - y0;
            float wI = (1 - wy) * (1 - wx);
            float wJ = (1 - wy) * wx;
            float wH = wy * (1 - wx);
            float wG = wx * wy;
            int32_t W = dims_original[3];
            // todo: 加上exclude_outside系数
            float value = wI * input[x0 + y0 * W] + wJ * input[x1 + y0 * W] +
                          wH * input[x0 + y1 * W] + wG * input[x1 + y1 * W];
            output[0] = value;
            ++output;
          }
        input += dims_original[2] * dims_original[3];
      }
    }
  }
  return 0;
}

int32_t Resize_linear_float3d(float *input, float *output, float *scale,
                              int32_t *dims_original, int32_t *dims_resized,
                              int32_t *roi, ctmode ctm) {
  for (int32_t n = 0; n < dims_resized[0]; ++n) {
    for (int32_t c = 0; c < dims_resized[1]; ++c) {
      for (int32_t h = 0; h < dims_resized[2]; ++h) {
        float h_original = GetCoordinateFunc(h, ctm, scale[2], dims_original[2],
                                             dims_resized[2], roi[2], roi[6]);

        float y = h_original;
        int32_t y0, y1;
        if (y < 0)
          y0 = y1 = 0;
        else if (y > dims_original[2] - 1)
          y0 = y1 = dims_original[2] - 1;
        else {
          y0 = (int32_t)h_original;
          y1 = y0 + 1;
        }
        /**
         * H(x0,y1)    G(x1,y1)
         *      X(x,y)
         * I(x0,y0)    J(x1,y0)
         **/
        float wy = y - y0;
        float wI = (1 - wy);
        float wH = wy;
        // todo: 加上exclude_outside系数
        float value = wI * input[y0] + wH * input[y1];
        output[0] = value;
        ++output;
      }
      input += dims_original[2];
    }
  }
  return 0;
}

float Bicubic(float x, float a) {
  x = fabs(x);
  if (x <= 1)
    return (a + 2) * pow(x, 3) - (a + 3) * pow(x, 2) + 1;
  else if (x < 2)
    return a * pow(x, 3) - 5 * a * pow(x, 2) + 8 * a * x - 4 * a;
  else
    return 0;
}

int32_t Resize_cubic_float(float *input, float *output, float *scale,
                           int32_t *dims_original, int32_t *dims_resized,
                           int32_t *roi, ctmode ctm, float cubic_coeff_a) {
  for (int32_t n = 0; n < dims_resized[0]; ++n) {
    for (int32_t c = 0; c < dims_resized[1]; ++c) {
      for (int32_t h = 0; h < dims_resized[2]; ++h)
        for (int32_t w = 0; w < dims_resized[3]; ++w) {
          float h_original =
              GetCoordinateFunc(h, ctm, scale[2], dims_original[2],
                                dims_resized[2], roi[2], roi[6]);
          float w_original =
              GetCoordinateFunc(w, ctm, scale[3], dims_original[3],
                                dims_resized[3], roi[3], roi[7]);
          int32_t x = (int32_t)floor(w_original),
                  y = (int32_t)floor(h_original);
          int32_t H = dims_original[2];
          int32_t W = dims_original[3];
          // 获取周围4*4图像的坐标
          int32_t dx[4], dy[4];
          for (int32_t i = 0; i < 4; ++i) {
            dx[i] = x + i - 1;
            dy[i] = y + i - 1;
            dx[i] = dx[i] < 0 ? 0 : dx[i];
            dx[i] = dx[i] >= W ? W - 1 : dx[i];
            dy[i] = dy[i] < 0 ? 0 : dy[i];
            dy[i] = dy[i] >= H ? H - 1 : dy[i];
          }
          // 获取周围4*4图像的权重
          float wx[4], wy[4];
          float u = w_original - x;
          float v = h_original - y;
          wx[0] = Bicubic(1 + u, cubic_coeff_a);
          wx[1] = Bicubic(u, cubic_coeff_a);
          wx[2] = Bicubic(1 - u, cubic_coeff_a);
          wx[3] = Bicubic(2 - u, cubic_coeff_a);
          wy[0] = Bicubic(1 + v, cubic_coeff_a);
          wy[1] = Bicubic(v, cubic_coeff_a);
          wy[2] = Bicubic(1 - v, cubic_coeff_a);
          wy[3] = Bicubic(2 - v, cubic_coeff_a);
          // 加权求和得到最终结果
          float value = 0;

          // todo: 加上exclude_outside系数
          for (int32_t i = 0; i < 4; ++i)
            for (int32_t j = 0; j < 4; ++j) {
              value += wx[i] * wy[j] * input[dx[i] + dy[j] * W];
            }
          output[0] = value;
          ++output;
        }
      input += dims_original[2] * dims_original[3];
    }
  }
  return 0;
}

int32_t X(Forward)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                   tDMA_List *list) {
  assert(num_tensor == (op->num_input_ + op->num_output_));
  tTensor *X = ((tTensor **)tensors)[0];
  tTensor *S = ((tTensor **)tensors)[kScales];
  tTensor *Y = ((tTensor **)tensors)[op->num_input_];

  ResizeAttrs *attr = (ResizeAttrs *)((int8_t *)op + op->attr_offset_);

  ResizeMode mode = (ResizeMode)attr->mode;
  ctmode ctm = (ctmode)attr->coord_trans_mode;

  if (X->dtype_ == Float32) {
    float *input = (float *)tensors[kData]->dptr_;
    float *output = (float *)tensors[op->num_input_]->dptr_;
    float scale[4];
    int32_t dims_original[4], dims_resized[4], roi[8];
    // todo: output_dimension = floor(input_dimension * (roi_end - roi_start) *
    // scale) if input "sizes" is not specified. 加上考虑roi的情况
    for (int32_t i = 0; i < X->shape_.ndim_; ++i) {
      dims_original[i] = X->shape_.dims_[i];
      dims_resized[i] = Y->shape_.dims_[i];
      roi[i] = 0;
      roi[i + X->shape_.ndim_] = dims_resized[i];
    }

    if (S->shape_.ndim_ != 0 && S->shape_.dims_[0] != 0) {
      const float *scale_input = (float *)S->dptr_;
      for (int32_t i = 0; i < X->shape_.ndim_; ++i) {
        scale[i] = scale_input[i];
      }
    } else {
      for (int32_t i = 0; i < X->shape_.ndim_; ++i) {
        scale[i] = dims_resized[i] * 1.0 / dims_original[i];
      }
    }

    nearestMode nmode = (nearestMode)attr->nearest_mode;
    float cubic_coeff_a = attr->cubic_coeff_a;
    switch (mode) {
      case knearnest:
        if (X->shape_.ndim_ == 3) {
          return Resize_nearest_float3d(input, output, scale, dims_original,
                                        dims_resized, roi, ctm, nmode);
        } else {
          return Resize_nearest_float4d(input, output, scale, dims_original,
                                        dims_resized, roi, ctm, nmode);
        }

      case klinear:
        if (X->shape_.ndim_ == 3) {
          return Resize_linear_float3d(input, output, scale, dims_original,
                                       dims_resized, roi, ctm);
        } else {
          return Resize_linear_float(input, output, scale, dims_original,
                                     dims_resized, roi, ctm);
        }
      case kcubic:
        return Resize_cubic_float(input, output, scale, dims_original,
                                  dims_resized, roi, ctm, cubic_coeff_a);
      default:
        THINKER_LOG_FATAL("Resize: Unsupported ResizeMode!");
        break;
    }
  }
  return 0;
}

#include "core/operator_template.h"
#undef __OP__