#ifndef __SOURCE_EXECUTOR_CORE_SCALAR_OP_H__
#define __SOURCE_EXECUTOR_CORE_SCALAR_OP_H__

#include <math.h>
#include <assert.h>
#include "thinker_api.h"


#define _SCALAR_ADD(a, b) (a + b)
#define _SCALAR_SUB(a, b) (a - b)
#define _SCALAR_MUL(a, b) (a * b)
#define _SCALAR_DIV(a, b) (a / b)
#define _SCALAR_MIN(a, b) (a < b ? a : b)
#define _SCALAR_MAX(a, b) (a > b ? a : b)

#define _SCALAR_UNARY_FUNC(func, data, ids, input_num)                         \
  CHECK_EQ(input_num, 1);                                                      \
  data[ids[input_num]] = func(data[ids[0]]);

#define _SCALAR_BINARY_FUNC(func, data, ids, input_num)                        \
  data[ids[input_num]] = data[ids[0]];                    \
  for(int i = 1; i < input_num; i++) \
    data[ids[input_num]] = func(data[ids[input_num]],  data[ids[i]])

typedef enum ScalarOpType {
  ADD   = 1,
  MUL   = 2,
  DIV   = 3,
  POW   = 4,
  FLOOR = 5,
  CEIL  = 6,
  SQRT  = 7,
  MIN   = 8,
  MAX   = 9,
} ScalarOpType;

int32_t ScalarFunc(double *scalars, const ScalarOpType *op_type,
                   const int32_t *io_ids, int input_num);

#endif // __SOURCE_EXECUTOR_CORE_SCALAR_OP_H__