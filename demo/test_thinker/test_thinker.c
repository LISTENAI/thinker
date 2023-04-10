#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "thinker/thinker.h"
#include "thinker/thinker_status.h"

#define PSRAM_SIZE  (8*1024*1024)
#define SHARE_SIZE  (640*1024)

static int8_t g_psram_buf[PSRAM_SIZE];
static int8_t g_share_buf[SHARE_SIZE];

const char *classes[] = {
	"apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
	"bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
	"chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup",
	"dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house",
	"kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man",
	"maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid",
	"otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
	"possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew",
	"skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper",
	"table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
	"wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
};

static void load_bin_file(const char *file, int8_t **ptr, uint64_t *size) 
{
    FILE *fp = fopen(file, "rb");
	if (fp == NULL){
		printf("open file failed, check the path!\n");
	}

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
	int32_t scale = atoi(argv[7]);

	load_bin_file(input_file, &input_data, &input_size);
    load_bin_file(model_file, &res_data, &res_size);

    tStatus ret = T_SUCCESS;
	ret = tInitialize();
	if (ret != T_SUCCESS) {
        printf("tInitialize failed, error code:%d\n", ret);
		return ret;
    }

	int num_memory = 0;
	tMemory memory_list[7];
	ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res_data, res_size);
    if (ret != T_SUCCESS) {
        printf("tGetMemoryPlan failed, error code:%d\n", ret);
		return ret;
    }

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
        printf("tInitModel failed, error code:%d\n", ret);
		return ret;
    }
	else{
		printf("init model successful!\n");
	}

    tExecHandle hdl;
    ret = tCreateExecutor(model_hdl, &hdl, memory_list, num_memory);
    if (ret != T_SUCCESS) {
        printf("tCreateExecutor failed, error code:%d\n", ret);
		return ret;
    }
	else{
		printf("create executor successful!\n");
	}

  	tData input; 
	input.dptr_ = (char*)input_data;//在此处设断点
	input.dtype_ = Int8;
	input.scale_ = scale;
	input.shape_.ndim_ = 4;
	input.shape_.dims_[0] = 1;
	input.shape_.dims_[1] = in_c;
    input.shape_.dims_[2] = in_h;
    input.shape_.dims_[3] = in_w;

	uint32_t clk = 0;
	for(i = 0; i < loop_count; i++)
	{
		ret = tSetInput(hdl, 0, &input);
		if (ret != T_SUCCESS) {
			printf("tSetInput failed, error coe:%d\n", ret);
			return ret;
		}

		ret = tForward(hdl);
		if (ret != T_SUCCESS) {
			printf("tForward failed, error code:%d\n", ret);
			return ret;
		}
		else{
			printf("forward successful!\n");
		}

		tData output[5];
		int getoutputcount = tGetOutputCount(model_hdl);

		for(j = 0; j < getoutputcount; j++)
		{
			ret = tGetOutput(hdl, j, &output[j]);
			if (ret != T_SUCCESS) {
				printf("tGetOutput_%d failed, error code: %d\n", j, ret);
				return ret;
			}
		}

		int8_t *output_data = (int8_t *)output[0].dptr_;
		int output_length = output[0].shape_.dims_[1];

		int predicted_category_index = 0;
		int8_t max_probability = output_data[0];

		for (int idx = 1; idx < output_length; idx++) {
			if (output_data[i] > max_probability)
			{
				max_probability = output_data[idx];
				predicted_category_index = idx;
			}
		}

		printf("Predicted category index: %d\n", predicted_category_index);
		printf("Predicted label: %s\n", classes[predicted_category_index]);	

		save_bin_file(output_file, output_data, output_length);
	}
	tUninitialize();
    return ret;
}

int main(int argc, char *argv[]) 
{
	if(argc < 8)
	{
		printf("commad:path of input file, path of model, path of output file, channel of input, height of input, width of input, QValue of input, loop num(opt)\n");
		return -1;
	}
	int loop_count = 1;
	if( argc == 9)
		loop_count = atoi(argv[8]);
	int32_t ret = thinker_task_test(loop_count, argv);

	return ret;
}
