#ifndef __THINKER_LOG_H__
#define __THINKER_LOG_H__
#include <stdio.h>
#include <stdlib.h>

#include "thinker_status.h"

#define CHECK_BINARY_OP(op, x, y)                                            \
  if (!((x)op(y))) {                                                         \
    printf("%s:%d | %s failed.\n", __FILE__, __LINE__, (#x " " #op " " #y)); \
    abort();                                                                 \
  }

#define CHECK_LT(x, y) CHECK_BINARY_OP(<, x, y)
#define CHECK_GT(x, y) CHECK_BINARY_OP(>, x, y)
#define CHECK_LE(x, y) CHECK_BINARY_OP(<=, x, y)
#define CHECK_GE(x, y) CHECK_BINARY_OP(>=, x, y)
#define CHECK_NE(x, y) CHECK_BINARY_OP(!=, x, y)
#define CHECK_EQ(x, y) CHECK_BINARY_OP(==, x, y)

#define THINKER_LOG_FATAL(msg)                        \
  {                                                   \
    printf("%s:%d | %s \n", __FILE__, __LINE__, msg); \
    abort();                                          \
  }

#define THINKER_LOG_WARNING(msg) \
  { printf("%s:%d | %s \n", __FILE__, __LINE__, msg); }

#define CHECK_BINARY_OP_RETURN(op, x, y, value) \
  if (!((x)op(y))) {                            \
    return (value);                             \
  }
#define CHECK_EQ_RETURN(x, y, value) CHECK_BINARY_OP_RETURN(==, x, y, value)
#define CHECK_NE_RETURN(x, y, value) CHECK_BINARY_OP_RETURN(!=, x, y, value)
#define CHECK_LT_RETURN(x, y, value) CHECK_BINARY_OP_RETURN(<, x, y, value)
#define CHECK_GT_RETURN(x, y, value) CHECK_BINARY_OP_RETURN(>, x, y, value)
#define CHECK_LE_RETURN(x, y, value) CHECK_BINARY_OP_RETURN(<=, x, y, value)
#define CHECK_GE_RETURN(x, y, value) CHECK_BINARY_OP_RETURN(>=, x, y, value)

#endif  // __THINKER_LOG_H__
