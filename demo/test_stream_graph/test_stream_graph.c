#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "thinker/thinker.h"
#include "thinker/thinker_status.h"

#define PSRAM_SIZE  (8*1024*1024)
#define SHARE_SIZE  (640*1024)

#define MAX_INPUT_COUNT 32
#define MAX_OUTPUT_COUNT 32
#define MAX_KV_PAIR_COUNT 32
#define MAX_MEMORY_COUNT 64
#define MAX_PATH_LEN 256

#define THINKER_CHECK(func_call, func_name) \
    do { \
        tStatus ret = func_call; \
        if (ret != T_SUCCESS) { \
            printf("Failed to %s: ret = %d\n", func_name, ret); \
            return -1; \
        } \
    } while (0)

typedef struct {
    int32_t input_idx;
    int32_t output_idx;
    int8_t *cache_data;
    int32_t cache_size;
} KvCachePair;

typedef struct {
    int32_t input_idx;
    char file[MAX_PATH_LEN];
    int8_t *data;
    uint64_t size;
} StreamInput;

static int8_t g_psram_buf[PSRAM_SIZE];
static int8_t g_share_buf[SHARE_SIZE];
static tMemory g_memory_list[MAX_MEMORY_COUNT];
static StreamInput g_stream_inputs[MAX_INPUT_COUNT];
static KvCachePair g_kv_pairs[MAX_KV_PAIR_COUNT];

static int32_t load_binary_file(const char *file, int8_t **ptr, uint64_t *size)
{
    FILE *fp = fopen(file, "rb");
    if (!fp) {
        printf("Failed to open file: %s\n", file);
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    *size = (uint64_t)ftell(fp);
    fseek(fp, 0, SEEK_SET);
    *ptr = (int8_t *)malloc((size_t)*size);
    if (!*ptr) {
        printf("Memory allocation failed for file: %s\n", file);
        fclose(fp);
        return -1;
    }
    fread(*ptr, (size_t)*size, 1, fp);
    fclose(fp);
    return 0;
}

static int32_t save_binary_file(const char *file, int8_t *ptr, int32_t size)
{
    FILE *fp = fopen(file, "wb");
    if (!fp) {
        printf("Failed to open file: %s\n", file);
        return -1;
    }
    fwrite(ptr, size, 1, fp);
    fclose(fp);
    return 0;
}

static int32_t parse_index_file(const char *arg, int32_t *index, char *file, size_t file_size)
{
    const char *colon = strchr(arg, ':');
    char *end = NULL;
    long parsed_index;

    if (!colon) {
        printf("Invalid input format: %s\n", arg);
        return -1;
    }

    parsed_index = strtol(arg, &end, 10);
    if (end != colon || parsed_index < 0 || parsed_index >= MAX_INPUT_COUNT) {
        printf("Invalid input index: %s\n", arg);
        return -1;
    }

    *index = (int32_t)parsed_index;
    strncpy(file, colon + 1, file_size - 1);
    file[file_size - 1] = '\0';
    return 0;
}

static int32_t parse_index_pair(const char *arg, int32_t *left, int32_t *right)
{
    const char *colon = strchr(arg, ':');
    char *end = NULL;
    long parsed_left;
    long parsed_right;

    if (!colon) {
        printf("Invalid pair format: %s\n", arg);
        return -1;
    }

    parsed_left = strtol(arg, &end, 10);
    if (end != colon || parsed_left < 0 || parsed_left >= MAX_INPUT_COUNT) {
        printf("Invalid left index: %s\n", arg);
        return -1;
    }

    parsed_right = strtol(colon + 1, &end, 10);
    if (*end != '\0' || parsed_right < 0 || parsed_right >= MAX_OUTPUT_COUNT) {
        printf("Invalid right index: %s\n", arg);
        return -1;
    }

    *left = (int32_t)parsed_left;
    *right = (int32_t)parsed_right;
    return 0;
}

static int32_t data_size(const tData *data)
{
    uint32_t i;
    int32_t size = data->dtype_ & 0xF;

    for (i = 0; i < data->shape_.ndim_; i++) {
        size *= data->shape_.dims_[i];
    }
    return size;
}

static StreamInput *find_stream_input(StreamInput *inputs, int32_t input_file_count,
                                      int32_t input_idx)
{
    int32_t i;
    for (i = 0; i < input_file_count; i++) {
        if (inputs[i].input_idx == input_idx) {
            return &inputs[i];
        }
    }
    return NULL;
}

static KvCachePair *find_kv_input(KvCachePair *kv_pairs, int32_t kv_pair_count,
                                  int32_t input_idx)
{
    int32_t i;
    for (i = 0; i < kv_pair_count; i++) {
        if (kv_pairs[i].input_idx == input_idx) {
            return &kv_pairs[i];
        }
    }
    return NULL;
}

static int32_t output_is_kv_present(KvCachePair *kv_pairs, int32_t kv_pair_count,
                                    int32_t output_idx)
{
    int32_t i;
    for (i = 0; i < kv_pair_count; i++) {
        if (kv_pairs[i].output_idx == output_idx) {
            return 1;
        }
    }
    return 0;
}

static int32_t alloc_kv_cache(tExecHandle hdl, KvCachePair *kv_pairs, int32_t kv_pair_count,
                              int32_t *use_psram_size)
{
    int32_t i;

    for (i = 0; i < kv_pair_count; i++) {
        tData input;
        int32_t aligned_size;

        THINKER_CHECK(tGetInputInfo(hdl, kv_pairs[i].input_idx, &input), "tGetInputInfo");
        kv_pairs[i].cache_size = data_size(&input);
        aligned_size = (kv_pairs[i].cache_size + 63) & (~63);
        if (*use_psram_size + aligned_size > PSRAM_SIZE) {
            printf("psram size exceeded when allocating kv cache input %d\n",
                   kv_pairs[i].input_idx);
            return -1;
        }
        kv_pairs[i].cache_data = g_psram_buf + *use_psram_size;
        *use_psram_size += aligned_size;
        memset(kv_pairs[i].cache_data, 0, aligned_size);
    }
    return 0;
}

static int32_t update_kv_cache(tExecHandle hdl, KvCachePair *kv_pairs, int32_t kv_pair_count)
{
    int32_t i;

    for (i = 0; i < kv_pair_count; i++) {
        tData output;
        int32_t output_size;
        int32_t copy_size;

        THINKER_CHECK(tGetOutput(hdl, kv_pairs[i].output_idx, &output), "tGetOutput");
        output_size = data_size(&output);
        if (output_size > kv_pairs[i].cache_size) {
            printf("kv output %d size %d exceeds cache input %d size %d\n",
                   kv_pairs[i].output_idx, output_size, kv_pairs[i].input_idx,
                   kv_pairs[i].cache_size);
            return -1;
        }
        copy_size = output_size;
        memcpy(kv_pairs[i].cache_data, output.dptr_, copy_size);
        if (copy_size < kv_pairs[i].cache_size) {
            memset(kv_pairs[i].cache_data + copy_size, 0, kv_pairs[i].cache_size - copy_size);
        }
    }
    return 0;
}

static int32_t setup_memory(tMemory *memory_list, int32_t num_memory,
                            int32_t *use_psram_size, int32_t *use_share_size)
{
    int32_t i;

    for (i = 0; i < num_memory; i++) {
        int32_t mem_size = memory_list[i].size_;
        if (memory_list[i].dptr_ == 0) {
            int32_t aligned_size = (mem_size + 63) & (~63);
            if (memory_list[i].dev_type_ == 1 || memory_list[i].dev_type_ == 3) {
                if (*use_psram_size + aligned_size > PSRAM_SIZE) {
                    printf("psram size exceeded when allocating memory block %d\n", i);
                    return -1;
                }
                memory_list[i].dptr_ = (uint64_t)(g_psram_buf + *use_psram_size);
                *use_psram_size += aligned_size;
            } else if (memory_list[i].dev_type_ == 2) {
                if (*use_share_size + aligned_size > SHARE_SIZE) {
                    printf("share size exceeded when allocating memory block %d\n", i);
                    return -1;
                }
                memory_list[i].dptr_ = (uint64_t)(g_share_buf + *use_share_size);
                *use_share_size += aligned_size;
            }
        }
    }

    if (*use_psram_size > PSRAM_SIZE) {
        printf("psram size exceeded\n");
        return -1;
    }
    if (*use_share_size > SHARE_SIZE) {
        printf("share size exceeded\n");
        return -1;
    }
    return 0;
}

static int32_t save_step_outputs(tExecHandle hdl, tModelHandle model_hdl,
                                 KvCachePair *kv_pairs, int32_t kv_pair_count,
                                 const char *output_prefix, int32_t step)
{
    int32_t i;
    int32_t output_count = tGetOutputCount(model_hdl);

    if (!output_prefix) {
        return 0;
    }

    for (i = 0; i < output_count; i++) {
        tData output;
        char file[MAX_PATH_LEN];
        int32_t output_size;

        if (output_is_kv_present(kv_pairs, kv_pair_count, i)) {
            continue;
        }

        THINKER_CHECK(tGetOutput(hdl, i, &output), "tGetOutput");
        output_size = data_size(&output);
        snprintf(file, sizeof(file), "%s_step%d_output%d.bin", output_prefix, step, i);
        if (save_binary_file(file, output.dptr_, output_size) != 0) {
            return -1;
        }
    }
    return 0;
}

static int32_t run_stream(tExecHandle hdl, tModelHandle model_hdl, int32_t step_count,
                          StreamInput *stream_inputs, int32_t input_file_count,
                          KvCachePair *kv_pairs, int32_t kv_pair_count,
                          const char *output_prefix)
{
    int32_t step;
    int32_t input_idx;
    int32_t input_count = tGetInputCount(model_hdl);

    for (step = 0; step < step_count; step++) {
        for (input_idx = 0; input_idx < input_count; input_idx++) {
            tData input;
            KvCachePair *kv_pair = find_kv_input(kv_pairs, kv_pair_count, input_idx);
            StreamInput *stream_input = find_stream_input(stream_inputs, input_file_count, input_idx);
            int32_t size;

            THINKER_CHECK(tGetInputInfo(hdl, input_idx, &input), "tGetInputInfo");
            size = data_size(&input);

            if (kv_pair) {
                input.dptr_ = kv_pair->cache_data;
            } else if (stream_input) {
                uint64_t offset = 0;
                if (stream_input->size >= (uint64_t)size * (uint64_t)step_count) {
                    offset = (uint64_t)size * (uint64_t)step;
                }
                if (stream_input->size < offset + (uint64_t)size) {
                    printf("Input file is too small for input %d step %d\n", input_idx, step);
                    return -1;
                }
                input.dptr_ = stream_input->data + offset;
            } else {
                printf("Missing stream input or kv cache for input %d\n", input_idx);
                return -1;
            }

            THINKER_CHECK(tSetInput(hdl, input_idx, &input), "tSetInput");
        }

        THINKER_CHECK(tForward(hdl), "tForward");
        printf("forward successful: step %d\n", step);

        if (save_step_outputs(hdl, model_hdl, kv_pairs, kv_pair_count,
                              output_prefix, step) != 0) {
            return -1;
        }
        if (update_kv_cache(hdl, kv_pairs, kv_pair_count) != 0) {
            return -1;
        }
    }
    return 0;
}

int thinker_task_test(int argc, char *argv[])
{
    int32_t i;
    int32_t arg_idx;
    int32_t step_count;
    int32_t input_file_count;
    int32_t kv_pair_count;
    int32_t use_psram_size = 0;
    int32_t use_share_size = 0;
    int8_t *model_data = NULL;
    uint64_t model_size = 0;
    int32_t num_memory = 0;
    tModelHandle model_hdl;
    tExecHandle hdl;
    const char *output_prefix = NULL;

    if (argc < 5) {
        printf("Usage: %s <model_file> <step_count> <num_input_files> <num_kv_pairs> ", argv[0]);
        printf("<input_idx:input_file>... <kv_input_idx:kv_output_idx>... [output_prefix]\n");
        return -1;
    }

    step_count = atoi(argv[2]);
    input_file_count = atoi(argv[3]);
    kv_pair_count = atoi(argv[4]);
    if (step_count <= 0 || input_file_count < 0 || input_file_count > MAX_INPUT_COUNT ||
        kv_pair_count < 0 || kv_pair_count > MAX_KV_PAIR_COUNT) {
        printf("Invalid step/input/kv count\n");
        return -1;
    }
    if (argc < 5 + input_file_count + kv_pair_count) {
        printf("Not enough arguments\n");
        return -1;
    }

    memset(g_psram_buf, 0, PSRAM_SIZE);
    memset(g_share_buf, 0, SHARE_SIZE);
    memset(g_memory_list, 0, sizeof(g_memory_list));
    memset(g_stream_inputs, 0, sizeof(g_stream_inputs));
    memset(g_kv_pairs, 0, sizeof(g_kv_pairs));

    arg_idx = 5;
    for (i = 0; i < input_file_count; i++) {
        if (parse_index_file(argv[arg_idx++], &g_stream_inputs[i].input_idx,
                             g_stream_inputs[i].file, sizeof(g_stream_inputs[i].file)) != 0) {
            return -1;
        }
        if (load_binary_file(g_stream_inputs[i].file, &g_stream_inputs[i].data,
                             &g_stream_inputs[i].size) != 0) {
            return -1;
        }
    }

    for (i = 0; i < kv_pair_count; i++) {
        if (parse_index_pair(argv[arg_idx++], &g_kv_pairs[i].input_idx,
                             &g_kv_pairs[i].output_idx) != 0) {
            return -1;
        }
    }

    if (argc > arg_idx) {
        output_prefix = argv[arg_idx];
    }

    if (load_binary_file(argv[1], &model_data, &model_size) != 0) {
        return -1;
    }

    THINKER_CHECK(tInitialize(), "tInitialize");
    THINKER_CHECK(tGetMemoryPlan(g_memory_list, &num_memory, model_data, model_size),
                  "tGetMemoryPlan");
    if (num_memory > MAX_MEMORY_COUNT) {
        printf("Too many memory blocks: %d\n", num_memory);
        return -1;
    }
    if (setup_memory(g_memory_list, num_memory, &use_psram_size, &use_share_size) != 0) {
        return -1;
    }

    THINKER_CHECK(tModelInit(&model_hdl, model_data, model_size, g_memory_list, num_memory),
                  "tModelInit");
    THINKER_CHECK(tCreateExecutor(model_hdl, &hdl, g_memory_list, num_memory),
                  "tCreateExecutor");

    if (alloc_kv_cache(hdl, g_kv_pairs, kv_pair_count, &use_psram_size) != 0) {
        return -1;
    }

    printf("memory used: psram=%d/%d, share=%d/%d\n",
           use_psram_size, PSRAM_SIZE, use_share_size, SHARE_SIZE);

    if (run_stream(hdl, model_hdl, step_count, g_stream_inputs, input_file_count,
                   g_kv_pairs, kv_pair_count, output_prefix) != 0) {
        return -1;
    }

    tUninitialize();
    return 0;
}

int main(int argc, char *argv[])
{
    return thinker_task_test(argc, argv);
}
