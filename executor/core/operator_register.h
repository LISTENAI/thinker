#ifndef _OPERATOR_REGISTER_H_
#define _OPERATOR_REGISTER_H_

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"

#ifdef __cplusplus
extern "C" {
#endif

// Alignment macros for memory alignment
#define ALIGN2(n) ((n + 1) & ~1)
#define ALIGN4(n) ((n + 3) & ~3)
#define ALIGN8(n) ((n + 7) & ~7)
#define ALIGN16(n) ((n + 15) & ~15)
#define ALIGN32(n) ((n + 31) & ~31)
#define ALIGN64(n) ((n + 63) & ~63)

// Macro concatenation helpers
#define CONCAT_(f, __OP__) _##__OP__##f
#define CONCAT(f, __OP__) CONCAT_(f, __OP__)
#define X(f) CONCAT(f, __OP__)

// Structure defining the interface for an operator
typedef struct tOperatorAPI {
    const char *(*name)();              // Function pointer to get operator name
    const char *(*groupname)();         // Function pointer to get group name

    int32_t (*init)(tOperator *op, tTensor **tensors, int32_t num_tensor, tHypeparam *init_params); // Initialization function
    int32_t (*fini)(tOperator *op, tTensor **tensors, int32_t num_tensor);                        // Finalization function

    int32_t (*forward)(tOperator *op, tTensor **tensors, int32_t num_tensor, tDMA_List *list);    // Forward pass function
} tOperatorAPI;

// Function declarations
int32_t RegistryOperatorAPI(tOperatorAPI api);
tOperatorAPI *GetOperatorAPI(const char *op_name);
int32_t GetOperatorCount();

#ifdef __cplusplus
}
#endif

#endif //_OPERATOR_REGISTER_H_