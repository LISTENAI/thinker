/** @file */
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

/**
 * Get library version
 * @param idx: Version index
 * @return: Version string
 */
THINKER_API(const char *, tGetVersion, (const int8_t idx));

/**
 * Initialize the THINKER system
 * @return: Status code
 */
THINKER_API(tStatus, tInitialize, ());

/**
 * Uninitialize the THINKER system
 * @return: Status code
 */
THINKER_API(tStatus, tUninitialize, ());

/**
 * Get memory plan for model loading
 * @param memory: Memory allocation array
 * @param num_memory: Number of memory entries
 * @param res: Resource data pointer
 * @param size: Resource size
 * @return: Status code
 */
THINKER_API(tStatus, tGetMemoryPlan,
            (tMemory * memory, int32_t *num_memory, const int8_t *res,
             const uint64_t size));

/**
 * Initialize model from resource
 * @param model: Model handle pointer
 * @param res: Resource data pointer
 * @param size: Resource size
 * @param memory: Memory allocation array
 * @param num_memory: Number of memory entries
 * @return: Status code
 */
THINKER_API(tStatus, tModelInit,
            (tModelHandle * model, const int8_t *res, const uint64_t size,
             const tMemory *memory, int32_t num_memory));

/**
 * Finalize model
 * @param model: Model handle
 * @return: Status code
 */
THINKER_API(tStatus, tModelFini, (tModelHandle model));

/**
 * Get number of input tensors
 * @param model: Model handle
 * @return: Number of input tensors
 */
THINKER_API(int32_t, tGetInputCount, (const tModelHandle model));

/**
 * Get input tensor name by index
 * @param model: Model handle
 * @param idx: Input index
 * @return: Input tensor name
 */
THINKER_API(const char *, tGetInputName,
            (const tModelHandle model, const int32_t idx));

/**
 * Get number of output tensors
 * @param model: Model handle
 * @return: Number of output tensors
 */
THINKER_API(int32_t, tGetOutputCount, (const tModelHandle model));

/**
 * Get output tensor name by index
 * @param model: Model handle
 * @param idx: Output index
 * @return: Output tensor name
 */
THINKER_API(const char *, tGetOutputName,
            (const tModelHandle model, const int32_t idx));

/**
 * Get input tensor information
 * @param hdl: Executor handle
 * @param idx: Input index
 * @param input: Input data structure
 * @return: Status code
 */
THINKER_API(tStatus, tGetInputInfo,
            (const tExecHandle hdl, const int32_t idx, tData *input));

/**
 * Get input data type
 * @param model: Model handle
 * @param idx: Input index
 * @return: Data type
 */
THINKER_API(tDType, tGetInputDataType,
            (const tModelHandle model, const int32_t idx));

/**
 * Get output data type
 * @param model: Model handle
 * @param idx: Output index
 * @return: Data type
 */
THINKER_API(tDType, tGetOutputDataType,
            (const tModelHandle model, const int32_t idx));

/**
 * Get input tensor shape
 * @param model: Model handle
 * @param idx: Input index
 * @return: Tensor shape
 */
THINKER_API(tShape, tGetInputShape,
            (const tModelHandle model, const int32_t idx));

/**
 * Get output tensor shape
 * @param model: Model handle
 * @param idx: Output index
 * @return: Tensor shape
 */
THINKER_API(tShape, tGetOutputShape,
            (const tModelHandle model, const int32_t idx));

/**
 * Create executor instance
 * @param model: Model handle
 * @param hdl: Executor handle pointer
 * @param memory_list: Memory list
 * @param num_memory: Number of memory entries
 * @return: Status code
 */
THINKER_API(tStatus, tCreateExecutor,
            (const tModelHandle model, tExecHandle *hdl,
             const tMemory *memory_list, const int32_t num_memory));

/**
 * Release executor instance
 * @param hdl: Executor handle
 * @return: Status code
 */
THINKER_API(tStatus, tReleaseExecutor, (tExecHandle hdl));

/**
 * Set input tensor by index
 * @param hdl: Executor handle
 * @param idx: Input index
 * @param input: Input data
 * @return: Status code
 */
THINKER_API(tStatus, tSetInput,
            (const tExecHandle hdl, const int32_t idx, const tData *input));

/**
 * Set input tensor by name
 * @param hdl: Executor handle
 * @param name: Input tensor name
 * @param input: Input data
 * @return: Status code
 */
THINKER_API(tStatus, tSetInputByName,
            (const tExecHandle hdl, const char *name, const tData *input));

/**
 * Get output tensor by index
 * @param hdl: Executor handle
 * @param idx: Output index
 * @param input: Output data structure
 * @return: Status code
 */
THINKER_API(tStatus, tGetOutput,
            (const tExecHandle hdl, const int32_t idx, tData *input));

/**
 * Get output tensor by name
 * @param hdl: Executor handle
 * @param name: Output tensor name
 * @param input: Output data structure
 * @return: Status code
 */
THINKER_API(tStatus, tGetOutputByName,
            (const tExecHandle hdl, const char *name, tData *input));

/**
 * Execute forward pass
 * @param hdl: Executor handle
 * @return: Status code
 */
THINKER_API(tStatus, tForward, (const tExecHandle hdl));

/**
 * Update dynamic shapes
 * @param hdl: Executor handle
 * @param axis_names: Axis names array
 * @param axis_sizes: Axis sizes array
 * @param num: Number of axes
 * @return: Status code
 */
THINKER_API(tStatus, tUpdateShape, (tExecHandle hdl, const char **axis_names, const uint32_t *axis_sizes, int32_t num));

/**
 * Start executor
 * @param hdl: Executor handle
 * @return: Status code
 */
THINKER_API(tStatus, tExecutorStart, (tExecHandle hdl));

/**
 * Stop executor
 * @param hdl: Executor handle
 * @return: Status code
 */
THINKER_API(tStatus, tExecutorStop, (tExecHandle hdl));

#if THINKER_USE_MTQ
/**
 * Get Luna list size
 * @param hdl: Executor handle
 * @param list_size: List size pointer
 * @param list_length: List length pointer
 * @param total_param: Total parameter pointer
 * @return: Status code
 */
THINKER_API(tStatus, tGetLunaListSize, (const tExecHandle hdl, uint32_t *list_size, uint32_t *list_length, uint32_t *total_param));

/**
 * Build Luna list
 * @param hdl: Executor handle
 * @param base_addr: Base address
 * @param sq_len: Sequence length
 * @return: Status code
 */
THINKER_API(tStatus, tBuildLunaList, (const tExecHandle hdl, int8_t *base_addr, uint32_t sq_len));

/**
 * Subdivide Luna list
 * @param sq_addr: Sequence address
 * @param sq_len: Sequence length
 * @param total_param_size: Total parameter size
 * @return: Status code
 */
THINKER_API(tStatus, tSubLunaList, (int8_t *sq_addr, uint32_t sq_len, uint32_t total_param_size));

/**
 * Get list result
 * @param hdl: Executor handle
 * @return: Status code
 */
THINKER_API(tStatus, tGetListResult, (const tExecHandle hdl));
#endif

/**
 * API function pointer structure
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
    Proc_tUpdateShape tUpdateShape;
    Proc_tExecutorStart tExecutorStart;
    Proc_tExecutorStop  tExecutorStop;

#if THINKER_USE_MTQ
    Proc_tGetLunaListSize tGetLunaListSize;
    Proc_tBuildLunaList tBuildLunaList;
    Proc_tSubLunaList tSubLunaList;
    Proc_tGetListResult tGetListResult;
#endif

    void * reserve[3];  // aligned 4*sizeof(pointer)
} thinkerApi;

/**
 * Get all THINKER API function pointers
 * @return: Pointer to thinkerApi structure
 */
THINKER_API(const thinkerApi *, thinkerGetApi, ());

#define API_LIST_THINKER(func) func(thinkerGetApi)

#ifdef __cplusplus
} /* extern "C" */
#endif /* C++ */

#endif  // THINKER_EXECUTOR_INCLUDE_THINKER_THINKER_H_