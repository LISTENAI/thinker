#ifndef MOSS_MODEL_H
#define MOSS_MODEL_H

#include "moss_hal_types.h"
#include "moss_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MOSS_MODEL_ABI_VERSION  3

typedef struct {
    uint32_t               abi_version;
    const char*            target_arch;
    const char*            model_name;
    uint32_t               workspace_size;
    int32_t                num_inputs;
    int32_t                num_outputs;
    const MossTensorInfo*  input_infos;
    const MossTensorInfo*  output_infos;
    int32_t                num_args;
    int32_t                workspace_arg_idx;
    int32_t                weight_arg_idx;
    MossKernelFunc         entry_func;
    uint32_t               global_workspace_size;
    int32_t                global_workspace_arg_idx;
} MossModelDesc;

#ifdef __cplusplus
}
#endif

#endif /* MOSS_MODEL_H */
