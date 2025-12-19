#include "scalar_op.h"
#include "thinker_log.h"

int32_t ScalarFunc(double *scalars, const ScalarOpType *op_type,
                   const int32_t *io_ids, int input_num)
{
  int32_t ret = 0;
  switch (*op_type)
  {
  case ADD:
    _SCALAR_BINARY_FUNC(_SCALAR_ADD, scalars, io_ids, input_num);
    break;
  case MUL:
    _SCALAR_BINARY_FUNC(_SCALAR_MUL, scalars, io_ids, input_num);
    break;
  case DIV:
    _SCALAR_BINARY_FUNC(_SCALAR_DIV, scalars, io_ids, input_num);
    break;
  case POW:
    _SCALAR_BINARY_FUNC(pow, scalars, io_ids, input_num);
    break;
  case FLOOR:
    _SCALAR_UNARY_FUNC(floor, scalars, io_ids, input_num);
    break;
  case CEIL:
    _SCALAR_UNARY_FUNC(ceil, scalars, io_ids, input_num);
    break;
  case SQRT:
    _SCALAR_UNARY_FUNC(sqrt, scalars, io_ids, input_num);
    break;
  case MIN:
    _SCALAR_BINARY_FUNC(_SCALAR_MIN, scalars, io_ids, input_num);
    break;
  case MAX:
    _SCALAR_BINARY_FUNC(_SCALAR_MAX, scalars, io_ids, input_num);
    break;
  default:
    assert(0);
    break;
  }
  return ret;
}