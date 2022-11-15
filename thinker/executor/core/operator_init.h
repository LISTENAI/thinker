#ifndef _OPERATOR_INIT_H_
#define _OPERATOR_INIT_H_

#define DECLARE(X) extern int32_t _##X##Registry();
#define CALL(X) _##X##Registry,

#include "operator_list.h"
typedef int32_t (*XRegistryFunc)();

OP_LIST(DECLARE)

XRegistryFunc ops_list[] = {OP_LIST(CALL)};

void init_ops_list() {
  for (size_t i = 0; i < sizeof(ops_list) / sizeof(XRegistryFunc); i++) {
    XRegistryFunc func = ops_list[i];
    func();
  }
}

#endif  //_OPS_OPERATOR_H_