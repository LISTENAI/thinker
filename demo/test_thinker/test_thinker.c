#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "thinker/thinker.h"
#include "thinker/thinker_status.h"

#define PSRAM_SIZE  (8*1024*1024)
#define SHARE_SIZE  (640*1024)

#define THINKER_CHECK(func_call, func_name) \
    do { \
        tStatus ret = func_call; \
        if (ret != T_SUCCESS) { \
            printf("Failed to %s: ret = %d\n", func_name, ret); \
            return -1; \
        } \
    } while (0)

static int8_t g_psram_buf[PSRAM_SIZE];
static int8_t g_share_buf[SHARE_SIZE];

static int32_t load_binary_file(const char *file, int8_t **ptr, uint64_t *size) 
{
    FILE *fp = fopen(file, "rb");
    if (!fp) {
        printf("Failed to open file: %s\n", file);
        return -1;
    }

    fseek(fp, 0 ,SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0 ,SEEK_SET);
    *ptr = (int8_t *)malloc(*size);
	if (!*ptr) {
		printf("Memory allocation failed for file: %s\n", file);
		fclose(fp);
        return -1;
    }
    fread(*ptr, *size, 1, fp);
    fclose(fp);
	return 0;
}

static int32_t save_binary_file(const char *file, int8_t *ptr, int32_t size)
{
	FILE *fp = fopen(file, "ab+");
    if (!fp) {
        printf("Failed to open file: %s\n", file);
        return -1;
    }
	fwrite(ptr, size, 1, fp);

	fclose(fp);
	return 0;
}

int thinker_task_test(int argc, char *argv[])
{
	if (argc < 3) {
        printf("Usage: %s <model_file> <input_files> [<output_files>]\n", argv[0]);
        return -1;
    }
	int i, j;
	int32_t use_psram_size = 0;
	int32_t use_share_size = 0;

    memset(g_psram_buf, 0, PSRAM_SIZE);
    memset(g_share_buf, 0, SHARE_SIZE);

    // Load model file
    int8_t *model_data = NULL;
    uint64_t model_size = 0;
    if (load_binary_file(argv[1], &model_data, &model_size) != 0) {
        printf("Failed to load model file\n");
        return -1;
    }

	// Initialize thinker
    THINKER_CHECK(tInitialize(), "tInitialize");

	// Get memory plan
	int num_memory = 0;
	tMemory memory_list[7];
	THINKER_CHECK(tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)model_data, model_size), "tGetMemoryPlan");


	// Allocate memory
	{
		for(i = 0; i < num_memory; i++)
		{
			int mem_size = memory_list[i].size_;
			if (memory_list[i].dptr_ == 0)
			{
				if (1 == memory_list[i].dev_type_ || 3 == memory_list[i].dev_type_)
				{
					memory_list[i].dptr_ = (uint64_t)(g_psram_buf + use_psram_size);
					use_psram_size += (mem_size+63)&(~63);
				}
				else if (2 == memory_list[i].dev_type_)
				{
					memory_list[i].dptr_ = (uint64_t)(g_share_buf + use_share_size);
					use_share_size += (mem_size+63)&(~63);
				}
			}
		}
		if (use_psram_size > PSRAM_SIZE) {
			printf("psram size is too much\n");
			return -1;
		}
		if (use_share_size > SHARE_SIZE) {
			printf("share size is too much\n");
			return -1;
		}
	}

	// Initialize model
    tModelHandle model_hdl;   //typedef uint64_t
	THINKER_CHECK(tModelInit(&model_hdl, (int8_t*)model_data, model_size, memory_list, num_memory), "tModelInit");

	// Create executor
    tExecHandle hdl;
    THINKER_CHECK(tCreateExecutor(model_hdl, &hdl, memory_list, num_memory), "tCreateExecutor");

	// Process inputs
	uint32_t input_count = tGetInputCount(model_hdl);
	if (argc - 2 < input_count) {
		printf("num of input file is not correct\n");
		return -1;
	}
	tData input; 
    int8_t *input_data = NULL;
    uint64_t input_size = 0;
	for (i = 0; i < input_count; i++) {
		if (load_binary_file(argv[i + 2], &input_data, &input_size) != 0) {
			printf("Failed to load model file\n");
			return -1;
		}
		THINKER_CHECK(tGetInputInfo(hdl, i, &input), "tGetInputInfo"); 
		input.dptr_ = (int8_t *)input_data;
		THINKER_CHECK(tSetInput(hdl, i, &input), "tSetInput");
	}

	THINKER_CHECK(tForward(hdl), "tForward");

	tData output;
	int output_count = tGetOutputCount(model_hdl);
	if (argc - 2 - input_count >= output_count)
	{
		for(i = 0; i < output_count; i++)
		{
			THINKER_CHECK(tGetOutput(hdl, i, &output), "tGetOutput");
			int shape_size = (output.dtype_ & 0xF);
			for(j = 0; j < output.shape_.ndim_; j++){
				shape_size *= output.shape_.dims_[j];
			}
			save_binary_file(argv[i + input_count + 2], output.dptr_, shape_size);
		}
	}

	tUninitialize();
    return 0;
}

int main(int argc, char *argv[]) 
{
	int loop_count = 1;
	thinker_task_test(argc, argv);

	return 0;
}
