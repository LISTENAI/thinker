#include "operator_register.h"

#include <stdlib.h>
#include <string.h>

static tOperatorAPI op_list[1024] = {{0}};
static int32_t op_count = 0;

int32_t RegistryOperatorAPI(tOperatorAPI api) {
  int32_t i = 0;
  for (i = 0; i < op_count; ++i) {
    if (strcmp(api.name(), op_list[i].name()) == 0) {
      return -1;
    }
  }
  int32_t op_id = op_count;
  op_list[op_id] = api;
  ++op_count;
  return op_id;
}

tOperatorAPI *GetOperatorAPI(const char *op_name) {
  int32_t i = 0;
  for (i = 0; i < op_count; ++i) {
    if (strcmp(op_name, op_list[i].name()) == 0) {
      return &(op_list[i]);
    }
  }
  return NULL;
}

int32_t GetOperatorCount() { return op_count; }