#ifndef  __SOURCE_EXECUTOR_CORE_SHAPE_INFER_SHAPE_INFER_H_
#define  __SOURCE_EXECUTOR_CORE_SHAPE_INFER_SHAPE_INFER_H_
#include <stdio.h>
#include <assert.h>
#include "thinker_define.h"
#include "scalar_op.h"

#if defined(__cplusplus)
extern "C" {
#endif

tStatus tShapeInferInit(const char *res, tShapeInfer *shape_infer);
tStatus tShapeInferFini(tShapeInfer *shape_infer);
tStatus tSetShapeInferInputByTensors(tShapeInfer *shape_infer, double *scalars, tTensor *tensors);
tStatus tSetShapeInferInputByNames(tShapeInfer *shape_infer, double *scalars, const char **axis_names, const uint32_t *axis_sizes, int num);
tStatus tShapeInferForward(tShapeInfer *shape_infer, double *scalars, tTensor *tensors);

#if defined(__cplusplus)
}
#endif


#endif  //__SOURCE_EXECUTOR_CORE_SHAPE_INFER_SHAPE_INFER_H_