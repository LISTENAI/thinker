#ifndef __MOSS_RUNTIME_INCLUDE_H__
#define __MOSS_RUNTIME_INCLUDE_H__

#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum __MEM_TYPE__ {
  FLASH = 0,
  PSRAM = 1,
  SHARE_MEM = 2,
  UNCERTAIN = 3,
} MemType;

typedef struct _mossTensorinfo {
	char* name;
	int8_t memorytype_datatype; ////memory type-high4bit, datatype-low4bit
	int8_t dim;
	int16_t shape[7];
	union {
		float scale;
		int32_t q;
		};
	int8_t* dataptr;
} MossTensor;

int32_t mSetModuleWeightBasePtr(void* model, int8_t* weightbase);
uint32_t mGetModuleWorkSpaceSize(void* model);
int32_t mSetModuleWorkSpacePtr(void* model, int8_t* spacebase);

uint32_t mGetModuleInputNum(void* model);
int32_t mGetModuleInputInfo(void* model, int inputId, MossTensor* in);
uint32_t mGetModuleOutputNum(void* model);
int32_t mGetModuleOutputInfo(void* model, int outputId, MossTensor* out);

int32_t mSetModuleInput(void* model, int inputId, MossTensor* input);
int32_t mSetModuleOutput(void* model, int outputId, MossTensor* output);

int32_t mCreateExecutor(void* model, void** executor);
int32_t mReleaseExecutor(void* executor);

int32_t mModelForward(void* model);
//int32_t ModelForwardDebug(void* model, uint32_t times);


#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif //__MOSS_RUNTIME_INCLUDE_H__
