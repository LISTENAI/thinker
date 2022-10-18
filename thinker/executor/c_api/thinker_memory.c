#include "thinker_memory.h"

tStatus tMemoryMalloc(tMemory *memory) {
  if (memory->dev_type_ == SHARE_MEM) {
#if THINKER_USE_VENUS
    memory->dptr_ = (addr_type)(0x5FE00000);
    uint64_t size = memory->size_;
    return T_SUCCESS;
#endif
  }
  return T_ERR_INVALID_PLATFROM;
}

void tMemoryFree(tMemory *memory) { return; }
