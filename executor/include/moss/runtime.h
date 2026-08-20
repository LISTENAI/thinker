#ifndef __MOSS_RUNTIME_LIB_RUNTIME_H__
#define __MOSS_RUNTIME_LIB_RUNTIME_H__

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "moss_runtime.h"

typedef int (*opfunc) (int argc, void** argv);

typedef struct _mossTensortmp {
	char* name;
	int8_t memorytype_datatype; //memorytype_high:4bit;
	int8_t dim;
	int16_t shape[7];
	union {
		float scale;
		int32_t q;
		};
} MossTensorInfo;

typedef struct _mossModel {
   uint32_t spaceSize;
   uint32_t spacePtrArguIndex;
   uint32_t weightPtrArguIndex;
   int32_t numIn;
   int32_t numOutput;
   MossTensorInfo** inoutList;//inList;
   //uint32_t * inoutIndex;

   int32_t numArgument;
   void* *args; //uint32_t
   int32_t (* mainGraphInterface) (uint32_t, void**);
} MossModel;

#endif //__MOSS_RUNTIME_LIB_RUNTIME_H__
