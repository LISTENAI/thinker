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

// Load binary file from disk
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

// Save binary file to disk
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
	if (argc < 6) {
        printf("Usage: %s <model_file> <num of input files> <num of dynamic axis> <input_files> <dynamic_axis_name:value> [<output_files>]\n", argv[0]);
        return -1;
    }
	
    int32_t i, j;
    int32_t use_psram_size = 0;
    int32_t use_share_size = 0;

	// Initialize memory buffers
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
		// Not necessary, as it has been confirmed in the offline packaging tool
		/*if (use_psram_size > PSRAM_SIZE) {
			printf("psram size exceeded\n");
			return -1;
		}
		if (use_share_size > SHARE_SIZE) {
			printf("share size exceeded\n");
			return -1;
		}*/
	}

	// Initialize model
    tModelHandle model_hdl;
    THINKER_CHECK(tModelInit(&model_hdl, (int8_t*)model_data, model_size, memory_list, num_memory), "tModelInit");

	// Create executor
    tExecHandle hdl;
    THINKER_CHECK(tCreateExecutor(model_hdl, &hdl, memory_list, num_memory), "tCreateExecutor");

	// Process inputs and dynamic axes
	uint32_t input_file_count = atoi(argv[2]);
	int32_t dynamic_axis_count = atoi(argv[3]);
	uint32_t input_count = tGetInputCount(model_hdl);
	if (input_file_count != input_count) {
		printf("num of input file is not correct\n");
		return -1;
	}

	// Parse dynamic axes parameters
	char **dynamic_axes_name = (char **)malloc(dynamic_axis_count * sizeof(char *));
	uint32_t *dynamic_shape = (uint32_t *)malloc(dynamic_axis_count * sizeof(uint32_t));

	for (i = 0; i < dynamic_axis_count; i++) {
		char *axis_ptr = argv[input_file_count + 4 + i];
		char *token = strtok(axis_ptr, ",");
		char *colon_ptr = strchr(token, ':');

		if (colon_ptr != NULL) {
			*colon_ptr = '\0';
			dynamic_axes_name[i] = strdup(token);
			dynamic_shape[i] = atoi(colon_ptr + 1);
		} else {
			printf("Invalid dynamic_axis format: %s\n", token);
			return -1;
		}
	}

	THINKER_CHECK(tUpdateShape(hdl, (const char**)dynamic_axes_name, dynamic_shape, dynamic_axis_count), "tUpdateShape");

	// Set inputs and run inference
	tData input; 
    int8_t *input_data = NULL;
    uint64_t input_size = 0;
	for (i = 0; i < input_count; i++) {
		if (load_binary_file(argv[i + 4], &input_data, &input_size) != 0) {
			printf("Failed to load model file\n");
			return -1;
		}
		THINKER_CHECK(tGetInputInfo(hdl, i, &input), "tGetInputInfo"); 
		
		input.dptr_ = (int8_t *)input_data;
		THINKER_CHECK(tSetInput(hdl, i, &input), "tSetInput");
	}

	// Note: The following commented section is a simplified version, 
	// directly sets the dynamic axis size.Replace lines 138 to 171 with the new code.
	/*
	tData input; 
    int8_t *input_data = NULL;
    uint64_t input_size = 0;
	for (i = 0; i < input_count; i++) {
		if (load_binary_file(argv[i + 4], &input_data, &input_size) != 0) {
			printf("Failed to load model file\n");
			return -1;
		}
		THINKER_CHECK(tGetInputInfo(hdl, i, &input), "tGetInputInfo"); 

		if (i == 1)
			input.shape_.dims_[1] = 12;
		input.dptr_ = (int8_t *)input_data;
		THINKER_CHECK(tSetInput(hdl, i, &input), "tSetInput");
	}
	THINKER_CHECK(tUpdateShape(hdl, NULL, NULL, 0), "tUpdateShape");
	*/

	THINKER_CHECK(tForward(hdl), "tForward");

	tData output;
	int output_count = tGetOutputCount(model_hdl);
	if (argc - 4 - input_count - dynamic_axis_count >= output_count)
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
