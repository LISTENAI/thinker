#ifndef _OPERATOR_TEMPLATE_H_
#define _OPERATOR_TEMPLATE_H_
#include "./operator_register.h"

#ifndef __OP__
#error "Marco X must be defined before include operator_template.h ."
#endif

#define XName_(x) #x
#define XName(x) XName_(x)

#ifdef __cplusplus
extern "C" {
#endif  // C++

const char *X(Name)() { return XName(__OP__); }

#ifndef __USER_INIT__
int32_t X(Init)(tOperator *op, tTensor **tensors, int32_t num_tensor,
                tHypeparam *init_params) {
  return 0;
}
int32_t X(Fini)(tOperator *op, tTensor **tensors, int32_t num_tensor) {
  return 0;
}
#endif

#ifdef __GNUC__
__attribute((constructor))
#endif
int32_t
X(Registry)() {
  tOperatorAPI api;
  api.name = X(Name);
  api.init = X(Init);
  api.fini = X(Fini);
  api.forward = X(Forward);
  RegistryOperatorAPI(api);
  return 0;
}

#ifdef __cplusplus
}
#endif

#endif  //_OPERATOR_TEMPLATE_H_