#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "thinker/thinker.h"
#include "thinker/thinker_status.h"

#define INPUT_FILE_PATH     "bin/input.bin"
#define MODEL_FILE_PATH     "bin/model.bin"
#define PSRAM_SIZE  (8*1024*1024)
#define SHARE_SIZE  (640*1024)

static int8_t g_psram_buf[PSRAM_SIZE];
static int8_t g_share_buf[SHARE_SIZE];

static void load_bin_file(const char *file, int8_t **ptr, uint64_t *size) 
{
    FILE *fp = fopen(file, "rb");

    fseek(fp, 0 ,SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0 ,SEEK_SET);
    *ptr = (int8_t *)malloc(*size);
    fread(*ptr, *size, 1, fp);
    fclose(fp);
}

static void save_bin_file(const char *file, int8_t *ptr, int32_t size)
{
	FILE *fp = fopen(file, "ab+");

	fwrite(ptr, size, 1, fp);

	fclose(fp);
}

int thinker_task_test(int loop_count, char *argv[])
{
	int i, j, k;
	int32_t use_psram_size = 0;
	int32_t use_share_size = 0;

    memset(g_psram_buf, 0, PSRAM_SIZE);
    memset(g_share_buf, 0, SHARE_SIZE);

    int8_t *input_data = NULL;
    int8_t *res_data = NULL;
    uint64_t input_size = 0;
    uint64_t res_size = 0;

	char *input_file = argv[1];
	char *model_file = argv[2];
	char *output_file = argv[3];
	int32_t in_c = atoi(argv[4]);
	int32_t in_h = atoi(argv[5]);
	int32_t in_w = atoi(argv[6]);
	// int32_t in_type = atoi(argv[7]);  //0:float 1:int8_t
	// int32_t in_scale = atoi(argv[8]);

	load_bin_file(input_file, &input_data, &input_size);
    load_bin_file(model_file, &res_data, &res_size);
    // load_bin_file(INPUT_FILE_PATH, &input_data, &input_size);
    // load_bin_file(MODEL_FILE_PATH, &res_data, &res_size);

    tStatus ret = T_SUCCESS;
	ret = tInitialize();

	int num_memory = 0;
	tMemory memory_list[7];
	ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res_data, res_size);

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

    tModelHandle model_hdl;   //typedef uint64_t
    ret = tModelInit(&model_hdl, (int8_t*)res_data, res_size, memory_list, num_memory);
    if (ret != T_SUCCESS) {
        printf("tInitModel: ret = %d\n", ret);
    }

    tExecHandle hdl;
    ret = tCreateExecutor(model_hdl, &hdl, memory_list, num_memory);
    if (ret != T_SUCCESS) {
        printf("tCreateExecutor: ret = %d\n", ret);
    }

  	tData input; 
	input.dptr_ = (char*)input_data;
	input.dtype_ = Float32;
	input.scale_ = 1.0;
	input.shape_.ndim_ = 4;
	input.shape_.dims_[0] = 1;
	input.shape_.dims_[1] = in_c;
    input.shape_.dims_[2] = in_h;
    input.shape_.dims_[3] = in_w;

	ret = tSetInput(hdl, 0, &input);
	if (ret != T_SUCCESS) 
	{
		printf("tSetInput: %d\n", ret);
	}

	uint32_t clk = 0;
	for(i = 0; i < loop_count; i++)
	{
		ret = tForward(hdl);        //error
		if (ret != T_SUCCESS) 
		{
			printf("tForward: %d\n", ret);
		}

		tData output[5];
		int getoutputcount = tGetOutputCount(model_hdl);
		for(j = 0; j < getoutputcount; j++)
		{
			void * data = (void *)(g_psram_buf + use_psram_size);
			ret = tGetOutput(hdl, j, &output[j]);
			if (ret != T_SUCCESS)
			{
				printf("tGetOutput: %d\n", ret);
			}
			int shape_size = (output[j].dtype_ & 0xF);
			for(k = 0; k < output[j].shape_.ndim_; k++){
				shape_size *= output[j].shape_.dims_[k];
			}
			use_psram_size += (shape_size+63)&(~63);
			memcpy((char*)data, output[j].dptr_, shape_size);
			output[j].dptr_ = (void *)data;
			save_bin_file(output_file, data, shape_size);
		}
	}
	tUninitialize();
    return ret;
}

int main(int argc, char *argv[]) 
{
	int loop_count = 1;
	thinker_task_test(loop_count, argv);

	return 0;
}
