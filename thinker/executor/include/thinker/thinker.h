/** @file */

// thinker.h - THINKER API interface and data structures visible to user code.

// Copyright (c) 2021-2022 LISTENAI, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef THINKER_EXECUTOR_INCLUDE_THINKER_THINKER_H_
#define THINKER_EXECUTOR_INCLUDE_THINKER_THINKER_H_

#include <stdint.h>

#include "thinker_status.h"
#include "thinker_type.h"

#define T_FORCE_STOP_VALUE  (20221109)

#if defined(WIN32)
#if defined(x_EXPORTS)
#define THINKER_DLL __declspec(dllexport)
#elif defined(x_STATIC)
#define THINKER_DLL
#else
#define THINKER_DLL __declspec(dllimport)
#endif
#else /* defined(WIN32) */
#define THINKER_DLL __attribute__((visibility("default")))
#endif /* defined(WIN32) */

#define THINKER_API(RetType, Name, Param) \
  THINKER_DLL RetType Name Param;         \
  typedef RetType(*Proc_##Name) Param;

#ifdef __cplusplus
extern "C" {
#endif /* C++ */

THINKER_API(const char *, tGetVersion, (const int8_t idx));

THINKER_API(tStatus, tInitialize, ());
THINKER_API(tStatus, tUninitialize, ());

THINKER_API(tStatus, tGetMemoryPlan,
            (tMemory * memory, int32_t *num_memory, const int8_t *res,
             const uint64_t size));

THINKER_API(tStatus, tModelInit,
            (tModelHandle * model, const int8_t *res, const uint64_t size,
             const tMemory *memory, const int32_t num_memory));
THINKER_API(tStatus, tModelFini, (tModelHandle model));

// input
THINKER_API(int32_t, tGetInputCount, (const tModelHandle model));
THINKER_API(const char *, tGetInputName,
            (const tModelHandle model, const int32_t idx));

// output
THINKER_API(int32_t, tGetOutputCount, (const tModelHandle model));
THINKER_API(const char *, tGetOutputName,
            (const tModelHandle model, const int32_t idx));

// input
THINKER_API(tStatus, tGetInputInfo,
            (const tExecHandle hdl, const int32_t idx, tData *input));
THINKER_API(tDType, tGetInputDataType,
            (const tModelHandle model, const int32_t idx));
THINKER_API(tDType, tGetOutputDataType,
            (const tModelHandle model, const int32_t idx));

// input shape
THINKER_API(tShape, tGetInputShape,
            (const tModelHandle model, const int32_t idx));
THINKER_API(tShape, tGetOutputShape,
            (const tModelHandle model, const int32_t idx));

// executor
THINKER_API(tStatus, tCreateExecutor,
            (const tModelHandle model, tExecHandle *hdl,
             const tMemory *memory_list, const int32_t num_memory));
THINKER_API(tStatus, tReleaseExecutor, (tExecHandle hdl));

THINKER_API(tStatus, tSetInput,
            (const tExecHandle hdl, const int32_t idx, const tData *input));
THINKER_API(tStatus, tSetInputByName,
            (const tExecHandle hdl, const char *name, const tData *input));

THINKER_API(tStatus, tGetOutput,
            (const tExecHandle hdl, const int32_t idx, tData *output));
THINKER_API(tStatus, tGetOutputByName,
            (const tExecHandle hdl, const char *name, tData *output));

THINKER_API(tStatus, tForward, (const tExecHandle hdl));


THINKER_API(tStatus, tExecutorStart, (tExecHandle hdl));
THINKER_API(tStatus, tExecutorStop, (tExecHandle hdl));

/**
 * @brief   thinkerApi
 *
 *  统一管理所有API符号.
 */
typedef struct _thinkerApi {
  Proc_tGetVersion tGetVersion;

  Proc_tInitialize tInitialize;
  Proc_tUninitialize tUninitialize;

  Proc_tGetMemoryPlan tGetMemoryPlan;

  Proc_tModelInit tModelInit;
  Proc_tModelFini tModelFini;

  Proc_tGetInputCount tGetInputCount;
  Proc_tGetInputInfo tGetInputInfo;
  Proc_tGetInputName tGetInputName;
  Proc_tGetOutputCount tGetOutputCount;
  Proc_tGetOutputName tGetOutputName;
  Proc_tGetInputDataType tGetInputDataType;
  Proc_tGetOutputDataType tGetOutputDataType;
  Proc_tGetInputShape tGetInputShape;
  Proc_tGetOutputShape tGetOutputShape;

  Proc_tCreateExecutor tCreateExecutor;
  Proc_tReleaseExecutor tReleaseExecutor;

  Proc_tSetInput tSetInput;
  Proc_tSetInputByName tSetInputByName;
  Proc_tGetOutput tGetOutput;
  Proc_tGetOutputByName tGetOutputByName;
  Proc_tForward tForward;
  Proc_tExecutorStart tExecutorStart;
  Proc_tExecutorStop  tExecutorStop;
} thinkerApi;// aligned 4*sizeof(pointer)

/**
 * @brief	thinkerGetApi
 *
 *  Get thinker all API Symbols which store to the thinkerApi or other advanced
 * API by name.
 *
 * @author	LISTENAI
 * @return	const void* - Return the api address in success, otherwise
 * return NULL.
 * @see
 * @exception
 */
THINKER_API(const thinkerApi *, thinkerGetApi, ());

#define API_LIST_THINKER(func) func(thinkerGetApi)

#ifdef __cplusplus
} /* extern "C" */
#endif /* C++ */

// #endif  // __THINKER_ENGINE_HPP__
#endif  // THINKER_EXECUTOR_INCLUDE_THINKER_THINKER_H_"
