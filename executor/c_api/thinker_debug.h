/**
 * @file    thinker_debug.h
 * @brief   Debug tools for x Engine @listenai
 *
 * @author  LISTENAI
 * @version 1.0
 * @date    2020/5/11
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

// Debug dump configuration
#ifndef THINKER_DUMP
#define THINKER_DUMP 0
#endif

#ifndef THINKER_DUMP_CRC32
#define THINKER_DUMP_CRC32 0
#endif

#if THINKER_DUMP

#if THINKER_DUMP_CRC32

/**
 * Write tensor data with CRC32 checksum
 * @param output_name Name of the output file
 * @param tensor Tensor data to write
 */
void write_file(char *output_name, tTensor *tensor) {
    char save_path[256];
    char temp[256];
    uint32_t crc32_calc = 0;
    
    // Replace '/' with '_' in filename
    for (int32_t i = 0; i < strlen(output_name); i++) {
        if (output_name[i] == '/') {
            output_name[i] = '_';
        }
    }
    
    // Build file path with shape information
    uint32_t shape_dim = tensor->shape_.ndim_;
    snprintf(save_path, "./workspace/data/%s##", output_name);
    size_t size = 1;
    for (int32_t j = 0; j < shape_dim; ++j) {
        size *= tensor->shape_.dims_[j];
        strcpy(temp, save_path);
        snprintf(save_path, "%s_%d", temp, tensor->shape_.dims_[j]);
    }
    strcpy(temp, save_path);
    snprintf(save_path, "%s.bin", temp);
    
    // Calculate data size and CRC32
    int8_t *data = (int8_t *)tensor->dptr_;
    int32_t data_size = tensor->shape_.dims_[0] * tensor->shape_.dims_[1] * 
                       tensor->shape_.dims_[2] * tensor->shape_.dims_[3] * 
                       (tensor->dtype_ & 0xf);
    
    crc32_calc = crc32_calc(data, data_size);
    
    // Print CRC32 and sample data
    printf("crc32_calc = 0x%08x, data = [0x%08x-0x%08x-0x%08x], name = %s\n", crc32_calc,
           ((uint32_t *)(data))[0], ((uint32_t *)(data + data_size / 2))[0],
           ((uint32_t *)(data + data_size - 4))[0], save_path);
}

#elif THINKER_DUMP_BIN

/**
 * Write tensor data as binary file
 * @param output_name Name of the output file
 * @param tensor Tensor data to write
 */
void write_file(char *output_name, tTensor *tensor) {
    char save_path[256];
    char temp[256];
    
    // Replace '/' with '_' in filename
    for (int32_t i = 0; i < strlen(output_name); i++) {
        if (output_name[i] == '/') {
            output_name[i] = '_';
        }
    }
    
    // Build file path with shape information
    uint32_t shape_dim = tensor->shape_.ndim_;
    snprintf(save_path, "./workspace/data/%s##", output_name);
    size_t size = 1;
    for (int32_t j = 0; j < shape_dim; ++j) {
        size *= tensor->shape_.dims_[j];
        strcpy(temp, save_path);
        snprintf(save_path, "%s_%d", temp, tensor->shape_.dims_[j]);
    }
    strcpy(temp, save_path);
    snprintf(save_path, "%s.bin", temp);
    
    // Write binary data
    FILE *fp = fopen(save_path, "wb");
    if (fp == NULL) return;
    
    float *data = (float *)tensor->dptr_;
    int32_t nchw = tensor->shape_.dims_[0] * tensor->shape_.dims_[1] * 
                   tensor->shape_.dims_[2] * tensor->shape_.dims_[3] * 
                   (tensor->dtype_ & 0xf);
    
    fwrite(data, nchw, 1, fp);
    fclose(fp);
}

#else

/**
 * Align size to power of two boundary
 * @param sz Size to align
 * @param n Alignment boundary (must be power of two)
 * @return Aligned size
 */
static size_t alignSize(size_t sz, int32_t n) { 
    return (sz + n - 1) & -n; 
}

/**
 * Write tensor data to text file with format conversion
 * @param output_name Name of the output file
 * @param tensor Tensor data to write
 */
void write_file(char *output_name, tTensor *tensor) {
    char save_path[256];
    char temp[256];
    
    // Replace '/' with '_' in filename
    for (int32_t i = 0; i < strlen(output_name); i++) {
        if (output_name[i] == '/') {
            output_name[i] = '_';
        }
    }
    
    // Build file path with shape information
    uint32_t shape_dim = tensor->shape_.ndim_;
    snprintf(save_path, sizeof(save_path), "./workspace/data/%s##", output_name);
    size_t size = 1;
    for (int32_t j = 0; j < shape_dim; ++j) {
        size *= tensor->shape_.dims_[j];
        strncpy(temp, save_path, sizeof(temp) - 1);
        temp[sizeof(temp) - 1] = '\0';
        int remaining = sizeof(save_path) - strlen(save_path) - 1;
        if (remaining > 0) {
            snprintf(save_path + strlen(save_path), remaining, "_%d", tensor->shape_.dims_[j]);
        }
    }
    strcpy(temp, save_path);
    temp[sizeof(temp) - 1] = '\0';
    int remaining = sizeof(save_path) - strlen(save_path) - 1;
    if (remaining > 0) {
        snprintf(save_path + strlen(save_path), remaining, ".txt");
    }
    
    FILE *fp = fopen(save_path, "wt");
    if (fp == NULL) {
        printf("file:%s is not create file\n", save_path);
        return;
    }
    
    // Handle different data types
    if (tensor->dtype_ == Float32) {
        float *data = (float *)tensor->dptr_;
        
        // Write float data
        for (int32_t i = 0; i < size; i++) {
            fprintf(fp, "%.6f\n", data[i]);
        }
    } else if (tensor->dtype_ == Int8) {
        int8_t *data = (int8_t *)tensor->dptr_;
        
        // Write int8 data
        for (int32_t i = 0; i < size; i++) {
            fprintf(fp, "%d\n", (int32_t)data[i]);
        }
    } else if (tensor->dtype_ == Uint8) {
        uint8_t *data = (uint8_t *)tensor->dptr_;
        
        // Write uint8 data
        for (int32_t i = 0; i < size; i++) {
            fprintf(fp, "%d\n", (int32_t)data[i]);
        }
    } else if (tensor->dtype_ == Int16) {
        int16_t *data = (int16_t *)tensor->dptr_;
        
        // Write int16 data
        for (int32_t i = 0; i < size; i++) {
            fprintf(fp, "%d\n", (int32_t)data[i]);
        }
    } else if (tensor->dtype_ == Int64) {
        uint64_t *data = (uint64_t *)tensor->dptr_;
        
        // Write int64 data
        for (int32_t i = 0; i < size; i++) {
            fprintf(fp, "%d\n", (int32_t)data[i]);
        }
    } else if (tensor->dtype_ == Int32) {
        uint32_t *data = (uint32_t *)tensor->dptr_;
        
        // Write int32 data
        for (int32_t i = 0; i < size; i++) {
            fprintf(fp, "%d\n", (int32_t)data[i]);
        }
    } else {
        THINKER_LOG_FATAL("dtype not support.");
    }
    
    fclose(fp);
}

#endif

#endif  // end THINKER_DUMP

// Performance profiling macros
#if THINKER_PROFILE
#ifdef WIN32
#include <inttypes.h>
#include <time.h>
#elif defined linux
#include <sys/time.h>
#include <unistd.h>
#else
#include "core_feature_cache.h"
#endif

#define PROFILE_BEGIN double start_t = tick_count();

#define PROFILE_END                                                          \
  {                                                                          \
    double finish_t = tick_count();                                          \
    float total_t = (float)(finish_t - start_t);                             \
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
    if (strcmp(op_api->name(), "Conv2dInt") == 0) {                          \
      typedef struct _ConvAttrs {                                            \
      uint8_t dilation[3];                                                   \
      uint16_t kernel[3];                                                    \
      uint8_t pad[6];                                                        \
      uint8_t stride[3];                                                     \
      int16_t group;                                                         \
      int16_t layout;                                                        \
      uint8_t quant_type;                                                    \
      uint8_t act_type;                                                      \
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

#endif