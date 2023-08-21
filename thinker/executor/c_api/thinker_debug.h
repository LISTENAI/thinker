/**
 * @file	thinker_debug.h
 * @brief	debug tool of x Engine @listenai
 *
 * @author	LISTENAI
 * @version	1.0
 * @date	2020/5/11
 *
 * @Version Record:
 *    -- v1.0: create 2020/5/11
 * Copyright (C) 2022 listenai Co.Ltd
 * All rights reserved.
 */

#ifndef _THINKER_DEBUG_H_
#define _THINKER_DEBUG_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../core/comm/thinker_log.h"
#include "thinker_define.h"

// THINKER_DUMP
#ifndef THINKER_DUMP
#define THINKER_DUMP 0
#endif

#ifndef THINKER_DUMP
#define THINKER_DUMP_CRC32 0
#endif

#if THINKER_DUMP

#if THINKER_DUMP_CRC32

void write_file(char *output_name, tTensor *tensor) {
  char save_path[256];
  char temp[240];
  uint32_t crc32 = 0;
  int32_t lenofstr = strlen(output_name);
  for (int32_t i = 0; i < lenofstr; i++) {
    if (output_name[i] == '/') {
      output_name[i] = '_';
    }
  }
  uint32_t shape_dim = tensor->shape_.ndim_;
  uint32_t *shape = tensor->shape_.dims_;
  sprintf(save_path, "./data/%s##", output_name);
  size_t size = 1;
  for (int32_t j = 0; j < shape_dim; ++j) {
    size *= shape[j];
    strcpy(temp, save_path);
    sprintf(save_path, "%s_%d", temp, shape[j]);
  }
  strcpy(temp, save_path);
  sprintf(save_path, "%s.bin", temp);

  int8_t *data = (int8_t *)tensor->dptr_;
  int32_t w = tensor->shape_.dims_[3];
  int32_t h = tensor->shape_.dims_[2];
  int32_t c = tensor->shape_.dims_[1];
  int32_t b = tensor->shape_.dims_[0];

  int32_t data_size = b * c * h * w * (tensor->dtype_ & 0xf);

  crc32_calc(data, data_size, &crc32);

  printf("crc32 = 0x%08x, data = [0x%08x-0x%08x-0x%08x], name = %s\n", crc32,
         ((uint32_t *)(data))[0], ((uint32_t *)(data + data_size / 2))[0],
         ((uint32_t *)(data + data_size - 4))[0], save_path);
}

#elif THINKER_DUMP_BIN

void write_file(char *output_name, tTensor *tensor) {
  char save_path[256];
  char temp[256];
  int32_t lenofstr = strlen(output_name);
  for (int32_t i = 0; i < lenofstr; i++) {
    if (output_name[i] == '/') {
      output_name[i] = '_';
    }
  }
  uint32_t shape_dim = tensor->shape_.ndim_;
  uint32_t *shape = tensor->shape_.dims_;
  sprintf(save_path, "./data/%s##", output_name);
  size_t size = 1;
  for (int32_t j = 0; j < shape_dim; ++j) {
    size *= shape[j];
    strcpy(temp, save_path);
    sprintf(save_path, "%s_%d", temp, shape[j]);
  }
  strcpy(temp, save_path);
  sprintf(save_path, "%s.bin", temp);
  FILE *fp = fopen(save_path, "wb");

  float *data = (float *)tensor->dptr_;
  float *output_data = NULL;
  int32_t w = tensor->shape_.dims_[3];
  int32_t h = tensor->shape_.dims_[2];
  int32_t c = tensor->shape_.dims_[1];
  int32_t b = tensor->shape_.dims_[0];

  int32_t nchw = b * c * h * w * (tensor->dtype_ & 0xf);

  fwrite(data, nchw, 1, fp);

  fclose(fp);
}

#else

// n Alignment size that must be a power of two
static size_t alignSize(size_t sz, int32_t n) { return (sz + n - 1) & -n; }

void write_file(char *output_name, tTensor *tensor) {
  char save_path[256];
  char temp[240];
  int32_t lenofstr = strlen(output_name);
  for (int32_t i = 0; i < lenofstr; i++) {
    if (output_name[i] == '/') {
      output_name[i] = '_';
    }
  }
  uint32_t shape_dim = tensor->shape_.ndim_;
  uint32_t *shape = tensor->shape_.dims_;
  sprintf(save_path, "./data/%s##", output_name);
  size_t size = 1;
  for (int32_t j = 0; j < shape_dim; ++j) {
    size *= shape[j];
    strcpy(temp, save_path);
    sprintf(save_path, "%s_%d", temp, shape[j]);
  }
  strcpy(temp, save_path);
  sprintf(save_path, "%s.txt", temp);
  FILE *fp = fopen(save_path, "wt");
  if (fp == NULL) {
    printf("file:%s is not create file\n", save_path);
    return;
  }

  if (tensor->dtype_ == Float32) {
    float *data = (float *)tensor->dptr_;
    float *output_data = NULL;
    int32_t w = tensor->shape_.dims_[3];
    int32_t h = tensor->shape_.dims_[2];
    int32_t c = tensor->shape_.dims_[1];
    int32_t b = tensor->shape_.dims_[0];

    int32_t cstep = w * h;
    if (NHWC4 == tensor->layout_ || NHWC == tensor->layout_) {
      output_data = (float *)malloc(sizeof(float) * size);
      int32_t c_align = c;
      if (NHWC4 == tensor->layout_) {
        c_align = c < 4 ? c : ((c + 3) / 4 * 4);
      }
      for (int32_t t0 = 0; t0 < b; t0++)
        for (int32_t t1 = 0; t1 < c; t1++)
          for (int32_t t2 = 0; t2 < h; t2++)
            for (int32_t t3 = 0; t3 < w; t3++) {
              int32_t index0 = t0 * c * h * w + t1 * h * w + t2 * w + t3;
              int32_t index1 =
                  t0 * h * w * c_align + t2 * w * c_align + t3 * c_align + t1;
              output_data[index0] = data[index1];
            }
    } else if (NC4HW4_T == tensor->layout_) {
      const int32_t c_r4 = (c + 3) / 4 * 4;
      output_data = (float *)malloc(sizeof(float) * size);
      for (int32_t t0 = 0; t0 < b; t0++)
        for (int32_t t1 = 0; t1 < c_r4 / 4; t1++)
          for (int32_t t2 = 0; t2 < h * w; t2++)
            for (int32_t t3 = 0; t3 < 4; t3++) {
              int32_t index0 =
                  t0 * c * h * w + (t1 * 4 + t3) * h * w + t2;  // nchw
              if (index0 < size) {
                int32_t index1 =
                    t0 * c_r4 * h * w + t1 * h * w * 4 + t2 * 4 + t3;  // nc4hw4
                output_data[index0] = data[index1];
              }
            }
    } else {
      output_data = (float *)tensor->dptr_;
    }

    for (int32_t i = 0; i < size; i++) {
      float data_val = output_data[i];
      fprintf(fp, "%.6f\n", data_val);
    }

    if (output_data != (float *)tensor->dptr_ && output_data != NULL) {
      free(output_data);
      output_data = NULL;
    }
  } else if (tensor->dtype_ == Float64) {
    float *data = (float *)tensor->dptr_;
    float *output_data = NULL;
    int32_t w = tensor->shape_.dims_[3];
    int32_t h = tensor->shape_.dims_[2];
    int32_t c = tensor->shape_.dims_[1];
    int32_t b = tensor->shape_.dims_[0];
    int32_t cstep = alignSize(w * h, 16);

    output_data = (float *)tensor->dptr_;

    for (int32_t i = 0; i < size; i++) {
      float data_val = output_data[i];
      fprintf(fp, "%.6f\n", data_val);
    }
  } else if (tensor->dtype_ == Int8) {
    int8_t *output_data = (int8_t *)tensor->dptr_;
    if ((NHWC == tensor->layout_ || NHWC4 == tensor->layout_))  // NHWC,NHWC4
    {
      size_t cstep = 0;
      if (4 == shape_dim) {
        cstep = shape[1];
        if (NHWC4 == tensor->layout_) {
          cstep = (cstep + 3) / 4 * 4;
        }
        for (int32_t batch = 0; batch < shape[0]; batch++) {
          for (int32_t c = 0; c < shape[1]; c++) {
            for (int32_t h = 0; h < shape[2]; h++) {
              for (int32_t w = 0; w < shape[3]; w++) {
                int32_t index = batch * shape[2] * shape[3] * cstep +
                                h * shape[3] * cstep + w * cstep + c;
                int8_t data_val = output_data[index];
                fprintf(fp, "%d\n", (int32_t)data_val);
              }
            }
          }
        }
      } else  // shape_dim is 1,2,3
      {
        for (int32_t i = 0; i < size; i++) {
          int8_t data_val = output_data[i];
          fprintf(fp, "%d\n", (int32_t)data_val);
        }
      }
    } else {
      for (int32_t i = 0; i < size; i++) {
        int8_t data_val = output_data[i];
        fprintf(fp, "%d\n", (int32_t)data_val);
      }
    }
  } else if (tensor->dtype_ == Uint8) {
    uint8_t *output_data = (uint8_t *)tensor->dptr_;
    for (int32_t i = 0; i < size; i++) {
      uint8_t data_val = output_data[i];
      fprintf(fp, "%d\n", (int32_t)data_val);
    }
  } else if (tensor->dtype_ == Int16) {
    int16_t *output_data = (int16_t *)tensor->dptr_;
    for (int32_t i = 0; i < size; i++) {
      int16_t data_val = output_data[i];
      fprintf(fp, "%d\n", (int32_t)data_val);
    }
  } else if (tensor->dtype_ == Int64) {
    uint64_t *output_data = (uint64_t *)tensor->dptr_;
    for (int32_t i = 0; i < size; i++) {
      uint64_t data_val = output_data[i];
      fprintf(fp, "%d\n", (int32_t)data_val);
    }
  } else if (tensor->dtype_ == Int32) {
    uint32_t *output_data = (uint32_t *)tensor->dptr_;
    for (int32_t i = 0; i < size; i++) {
      uint32_t data_val = output_data[i];
      fprintf(fp, "%d\n", (int32_t)data_val);
    }
  } else {
    THINKER_LOG_FATAL("dtype not support.");
  }
  fclose(fp);
}
#endif

#endif  // end THINKER_DUMP

// THINKER_PROFILE
#if THINKER_PROFILE
#ifdef WIN32
#include <inttypes.h>
#include <time.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

static double tick_count(void) {
#ifdef WIN32
  struct timespec tv;
  timespec_get(&tv, 1);
  return (double)(((int64_t)tv.tv_sec) * 1000 +
                  ((int64_t)tv.tv_nsec) / 1000000.0);
#else
  struct timeval tv;
  gettimeofday(&tv, 0);
  return (double)(tv.tv_sec * 1000 * 1000 + tv.tv_usec) / 1000.0;
#endif /* WIN32 */
}

#define PROFILE_BEGIN double start_t = tick_count();

#define PROFILE_END                                                          \
  {                                                                          \
    double finish_t = tick_count();                                          \
    float total_t = (float)(finish_t - start_t);                             \
    printf("%8s | %8.4f | (", op_api->name(), total_t);                      \
    printf("%4d", (int32_t)(local_tensor[0]->shape_.dims_[0]));              \
    for (uint32_t i = 1; i < local_tensor[0]->shape_.ndim_; ++i) {           \
      printf(", %4d", (int32_t)(local_tensor[0]->shape_.dims_[i]));          \
    }                                                                        \
    printf(") | (");                                                         \
    printf("%4d", (int32_t)(local_tensor[op->num_input_]->shape_.dims_[0])); \
    for (uint32_t i = 1; i < local_tensor[op->num_input_]->shape_.ndim_;     \
         ++i) {                                                              \
      printf(", %4d",                                                        \
             (int32_t)(local_tensor[op->num_input_]->shape_.dims_[i]));      \
    }                                                                        \
    if (strcmp(op_api->name(), "Conv") == 0) {                               \
      typedef struct _ConvAttrs {                                            \
        uint8_t dilation[2];                                                 \
        uint16_t kernel[2];                                                  \
        uint8_t pad[4];                                                      \
        uint8_t stride[2];                                                   \
        uint16_t group;                                                      \
        uint16_t layout;                                                     \
      } _ConvAttrs;                                                          \
      const _ConvAttrs *attrs =                                              \
          (const _ConvAttrs *)((char *)op + op->attr_offset_);               \
      printf(") | (");                                                       \
      printf("%d", (int32_t)(local_tensor[1]->shape_.dims_[0]));             \
      for (uint32_t i = 1; i < local_tensor[1]->shape_.ndim_; ++i) {         \
        printf(", %d", (int32_t)(local_tensor[1]->shape_.dims_[i]));         \
      }                                                                      \
      printf(") | (%2d, %2d) | (%d, %d, %d, %d) | (%d, %d",                  \
             (int32_t)attrs->kernel[0], (int32_t)attrs->kernel[1],           \
             (int32_t)attrs->pad[0], (int32_t)attrs->pad[1],                 \
             (int32_t)attrs->pad[2], (int32_t)attrs->pad[3],                 \
             (int32_t)attrs->stride[0], (int32_t)attrs->stride[1]);          \
    }                                                                        \
    printf(")\n");                                                           \
  }
#else  // THINKER_PROFILE
#define PROFILE_BEGIN
#define PROFILE_END
#endif
// end THINKER_PROFILE

#endif