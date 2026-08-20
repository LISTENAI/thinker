#ifndef MOSS_TENSOR_H
#define MOSS_TENSOR_H

#include "moss_hal_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MOSS_MAX_RANK  8
#define MOSS_MAX_NAME  64

typedef struct {
    char         name[MOSS_MAX_NAME];
    MossDtype    dtype;
    MossMemSpace mem_space;
    int32_t      rank;
    int32_t      shape[MOSS_MAX_RANK];
    float        scale;
    int32_t      zero_point;
} MossTensorInfo;

typedef struct {
    const MossTensorInfo* info;
    void*                 data_ptr;
    int32_t               shape[MOSS_MAX_RANK];
} MossTensorBinding;

#ifdef __cplusplus
}
#endif

#endif /* MOSS_TENSOR_H */
