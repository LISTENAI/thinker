#ifndef _OPERATOR_INIT_H_
#define _OPERATOR_INIT_H_

// Macro to declare registry functions
#define DECLARE(X) extern int32_t _##X##Registry();

// Macro to call registry functions
#define CALL(X) _##X##Registry,

// Include the operator list header
#include "operator_list.h"

// Function pointer type for registry functions
typedef int32_t (*XRegistryFunc)();

// Declare all operator registry functions using OP_LIST macro
OP_LIST(DECLARE)

// Create array of registry function pointers
XRegistryFunc ops_list[] = {OP_LIST(CALL)};

// Initialize all operators by calling their registry functions
void init_ops_list() {
    for (size_t i = 0; i < sizeof(ops_list) / sizeof(XRegistryFunc); i++) {
        XRegistryFunc func = ops_list[i];
        func();  // Call each registry function
    }
}

#endif  //_OPS_OPERATOR_H_