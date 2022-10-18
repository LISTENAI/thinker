#ifndef _OPERATOR_REGISTER_H_
#define _OPERATOR_REGISTER_H_
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include "c_api/thinker_define.h"
#include "core/comm/thinker_log.h"
#ifdef __cplusplus
extern "C" {
#endif  // C++

#define ALIGN64(n) ((n + 63) & ~63)
#define ALIGN32(n) ((n + 31) & ~31)
#define ALIGN16(n) ((n + 15) & ~15)

#define CONCAT_(f, __OP__) _##__OP__##f
#define CONCAT(f, __OP__) CONCAT_(f, __OP__)
#define X(f) CONCAT(f, __OP__)

typedef struct tOperatorAPI {
  const char *(*name)();  // 命名规则
  const char *(*groupname)();

  int32_t (*init)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                  tHypeparam *init_params);
  int32_t (*fini)(tOperator *op, tTensor **tensors, int32_t num_tensor);

  int32_t (*forward)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                     tDMA_List *list);
} tOperatorAPI;

int32_t RegistryOperatorAPI(tOperatorAPI api);
tOperatorAPI *GetOperatorAPI(const char *op_name);
int32_t GetOperatorCount();

#ifdef __cplusplus
}
#endif /* C++ */

#endif  //_OPERATOR_REGISTER_H_
