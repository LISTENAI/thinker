#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "thinker/thinker.h"
#include "thinker/thinker_status.h"

#define PSRAM_SIZE  (8*1024*1024)
#define SHARE_SIZE  (640*1024)

#define MAX_RESOURCES 16
#define MAX_LINKS 64
#define MAX_IO_COUNT 16
#define MAX_MEMORY_COUNT 64
#define MAX_NAME_LEN 64
#define MAX_PATH_LEN 256
#define MAX_LINE_LEN 512

#define THINKER_CHECK(func_call, func_name) \
    do { \
        tStatus ret = func_call; \
        if (ret != T_SUCCESS) { \
            printf("Failed to %s: ret = %d\n", func_name, ret); \
            return -1; \
        } \
    } while (0)

typedef struct {
    char name[MAX_NAME_LEN];
    char model_file[MAX_PATH_LEN];
    char input_files[MAX_IO_COUNT][MAX_PATH_LEN];
    char output_files[MAX_IO_COUNT][MAX_PATH_LEN];
    uint8_t has_input_file[MAX_IO_COUNT];
    uint8_t has_output_file[MAX_IO_COUNT];
    int8_t *model_data;
    uint64_t model_size;
    tMemory memory_list[MAX_MEMORY_COUNT];
    int32_t num_memory;
    tModelHandle model_hdl;
    tExecHandle exec_hdl;
    tData outputs[MAX_IO_COUNT];
    int32_t output_count;
} ResourceConfig;

typedef struct {
    char src_name[MAX_NAME_LEN];
    int32_t src_output;
    char dst_name[MAX_NAME_LEN];
    int32_t dst_input;
    int32_t src_resource;
    int32_t dst_resource;
} ResourceLink;

typedef struct {
    ResourceConfig resources[MAX_RESOURCES];
    int32_t resource_count;
    ResourceLink links[MAX_LINKS];
    int32_t link_count;
    char base_dir[MAX_PATH_LEN];
} MultiResourceConfig;

static int8_t g_psram_buf[PSRAM_SIZE];
static int8_t g_share_buf[SHARE_SIZE];
static MultiResourceConfig g_config;

static char *trim_left(char *str)
{
    while (*str && isspace((unsigned char)*str)) {
        str++;
    }
    return str;
}

static void trim_right(char *str)
{
    size_t len = strlen(str);
    while (len > 0 && isspace((unsigned char)str[len - 1])) {
        str[len - 1] = '\0';
        len--;
    }
}

static char *trim(char *str)
{
    str = trim_left(str);
    trim_right(str);
    return str;
}

static void strip_comment(char *line)
{
    char quote = '\0';
    char *p = line;
    while (*p) {
        if ((*p == '\'' || *p == '"') && (p == line || p[-1] != '\\')) {
            quote = quote == *p ? '\0' : *p;
        }
        if (*p == '#' && quote == '\0') {
            *p = '\0';
            return;
        }
        p++;
    }
}

static void strip_quotes(char *str)
{
    size_t len = strlen(str);
    if (len >= 2 && ((str[0] == '"' && str[len - 1] == '"') ||
                     (str[0] == '\'' && str[len - 1] == '\''))) {
        memmove(str, str + 1, len - 2);
        str[len - 2] = '\0';
    }
}

static void copy_string(char *dst, const char *src, size_t dst_size)
{
    if (dst_size == 0) {
        return;
    }
    strncpy(dst, src, dst_size - 1);
    dst[dst_size - 1] = '\0';
}

static int32_t next_io_index(uint8_t *used)
{
    int32_t i;
    for (i = 0; i < MAX_IO_COUNT; i++) {
        if (!used[i]) {
            return i;
        }
    }
    return -1;
}

static int32_t parse_key_value(char *line, char **key, char **value)
{
    char *colon = strchr(line, ':');
    if (!colon) {
        return -1;
    }

    *colon = '\0';
    *key = trim(line);
    *value = trim(colon + 1);
    strip_quotes(*value);
    return 0;
}

static int32_t parse_io_key(const char *key, const char *prefix)
{
    size_t len = strlen(prefix);
    char *end = NULL;
    long index;

    if (strncmp(key, prefix, len) != 0) {
        return -1;
    }
    if (!isdigit((unsigned char)key[len])) {
        return -1;
    }

    index = strtol(key + len, &end, 10);
    if (*end != '\0' || index < 0 || index >= MAX_IO_COUNT) {
        return -1;
    }
    return (int32_t)index;
}

static int32_t parse_endpoint(const char *value, char *name, size_t name_size, int32_t *index)
{
    char tmp[MAX_LINE_LEN];
    char *colon = NULL;
    char *end = NULL;
    long parsed_index;

    copy_string(tmp, value, sizeof(tmp));
    colon = strrchr(tmp, ':');
    if (!colon) {
        return -1;
    }

    *colon = '\0';
    parsed_index = strtol(colon + 1, &end, 10);
    if (*end != '\0' || parsed_index < 0 || parsed_index >= MAX_IO_COUNT) {
        return -1;
    }

    copy_string(name, trim(tmp), name_size);
    *index = (int32_t)parsed_index;
    return 0;
}

static int32_t find_resource(const MultiResourceConfig *config, const char *name)
{
    int32_t i;
    for (i = 0; i < config->resource_count; i++) {
        if (strcmp(config->resources[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

static void set_config_dir(const char *yaml_file, char *base_dir, size_t base_dir_size)
{
    const char *slash = strrchr(yaml_file, '/');
#ifdef _WIN32
    const char *backslash = strrchr(yaml_file, '\\');
    if (!slash || (backslash && backslash > slash)) {
        slash = backslash;
    }
#endif

    if (!slash) {
        copy_string(base_dir, ".", base_dir_size);
        return;
    }

    if ((size_t)(slash - yaml_file) >= base_dir_size) {
        copy_string(base_dir, ".", base_dir_size);
        return;
    }

    memcpy(base_dir, yaml_file, (size_t)(slash - yaml_file));
    base_dir[slash - yaml_file] = '\0';
}

static void resolve_path(const char *base_dir, const char *path, char *out, size_t out_size)
{
    if (path[0] == '/' || path[0] == '\0' ||
        (isalpha((unsigned char)path[0]) && path[1] == ':')) {
        copy_string(out, path, out_size);
        return;
    }

    if (strcmp(base_dir, ".") == 0) {
        copy_string(out, path, out_size);
    } else {
        snprintf(out, out_size, "%s/%s", base_dir, path);
    }
}

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

static int32_t parse_resource_property(ResourceConfig *resource, const char *key,
                                       const char *value)
{
    int32_t index;

    if (strcmp(key, "name") == 0) {
        copy_string(resource->name, value, sizeof(resource->name));
        return 0;
    }
    if (strcmp(key, "model") == 0) {
        copy_string(resource->model_file, value, sizeof(resource->model_file));
        return 0;
    }

    index = parse_io_key(key, "input");
    if (index >= 0) {
        copy_string(resource->input_files[index], value, sizeof(resource->input_files[index]));
        resource->has_input_file[index] = 1;
        return 0;
    }

    index = parse_io_key(key, "output");
    if (index >= 0) {
        copy_string(resource->output_files[index], value, sizeof(resource->output_files[index]));
        resource->has_output_file[index] = 1;
        return 0;
    }

    return 0;
}

static int32_t parse_io_item(ResourceConfig *resource, int32_t io_section, char *line)
{
    char *key = NULL;
    char *value = NULL;
    int32_t index;

    if (parse_key_value(line, &key, &value) != 0) {
        return -1;
    }
    if (strcmp(key, "file") != 0 && strcmp(key, "path") != 0) {
        return 0;
    }

    if (io_section == 1) {
        index = next_io_index(resource->has_input_file);
        if (index < 0) {
            return -1;
        }
        copy_string(resource->input_files[index], value, sizeof(resource->input_files[index]));
        resource->has_input_file[index] = 1;
    } else if (io_section == 2) {
        index = next_io_index(resource->has_output_file);
        if (index < 0) {
            return -1;
        }
        copy_string(resource->output_files[index], value, sizeof(resource->output_files[index]));
        resource->has_output_file[index] = 1;
    }

    return 0;
}

static int32_t parse_yaml_config(const char *yaml_file, MultiResourceConfig *config)
{
    FILE *fp = fopen(yaml_file, "r");
    char line[MAX_LINE_LEN];
    int32_t section = 0;
    int32_t io_section = 0;
    int32_t current_resource = -1;
    int32_t current_link = -1;

    if (!fp) {
        printf("Failed to open yaml file: %s\n", yaml_file);
        return -1;
    }

    memset(config, 0, sizeof(*config));
    set_config_dir(yaml_file, config->base_dir, sizeof(config->base_dir));

    while (fgets(line, sizeof(line), fp)) {
        char *text;
        char *key = NULL;
        char *value = NULL;
        int32_t indent = 0;

        strip_comment(line);
        while (line[indent] == ' ') {
            indent++;
        }
        text = trim(line);
        if (text[0] == '\0') {
            continue;
        }

        if (strcmp(text, "resources:") == 0) {
            section = 1;
            io_section = 0;
            current_resource = -1;
            continue;
        }
        if (strcmp(text, "links:") == 0) {
            section = 2;
            io_section = 0;
            current_link = -1;
            continue;
        }

        if (section == 1) {
            if (strncmp(text, "- ", 2) == 0 && indent <= 2) {
                if (config->resource_count >= MAX_RESOURCES) {
                    printf("Too many resources in yaml\n");
                    fclose(fp);
                    return -1;
                }
                current_resource = config->resource_count++;
                io_section = 0;
                text = trim(text + 2);
                if (text[0] == '\0') {
                    continue;
                }
            }

            if (current_resource < 0) {
                continue;
            }

            if (strcmp(text, "inputs:") == 0) {
                io_section = 1;
                continue;
            }
            if (strcmp(text, "outputs:") == 0) {
                io_section = 2;
                continue;
            }
            if (strncmp(text, "- ", 2) == 0) {
                text = trim(text + 2);
                if (parse_io_item(&config->resources[current_resource], io_section, text) != 0) {
                    printf("Failed to parse io item: %s\n", text);
                    fclose(fp);
                    return -1;
                }
                continue;
            }

            if (parse_key_value(text, &key, &value) == 0) {
                if (io_section == 0) {
                    parse_resource_property(&config->resources[current_resource], key, value);
                }
            }
        } else if (section == 2) {
            if (strncmp(text, "- ", 2) == 0) {
                if (config->link_count >= MAX_LINKS) {
                    printf("Too many links in yaml\n");
                    fclose(fp);
                    return -1;
                }
                current_link = config->link_count++;
                config->links[current_link].src_resource = -1;
                config->links[current_link].dst_resource = -1;
                text = trim(text + 2);
                if (text[0] == '\0') {
                    continue;
                }
            }

            if (current_link < 0) {
                continue;
            }
            if (parse_key_value(text, &key, &value) == 0) {
                if (strcmp(key, "from") == 0) {
                    if (parse_endpoint(value, config->links[current_link].src_name,
                                       sizeof(config->links[current_link].src_name),
                                       &config->links[current_link].src_output) != 0) {
                        printf("Invalid link source: %s\n", value);
                        fclose(fp);
                        return -1;
                    }
                } else if (strcmp(key, "to") == 0) {
                    if (parse_endpoint(value, config->links[current_link].dst_name,
                                       sizeof(config->links[current_link].dst_name),
                                       &config->links[current_link].dst_input) != 0) {
                        printf("Invalid link target: %s\n", value);
                        fclose(fp);
                        return -1;
                    }
                }
            }
        }
    }

    fclose(fp);
    return 0;
}

static int32_t finalize_config(MultiResourceConfig *config)
{
    int32_t i;
    int32_t j;

    if (config->resource_count == 0) {
        printf("No resources configured\n");
        return -1;
    }

    for (i = 0; i < config->resource_count; i++) {
        char path[MAX_PATH_LEN];
        ResourceConfig *resource = &config->resources[i];

        if (resource->name[0] == '\0' || resource->model_file[0] == '\0') {
            printf("Resource %d must configure name and model\n", i);
            return -1;
        }

        resolve_path(config->base_dir, resource->model_file, path, sizeof(path));
        copy_string(resource->model_file, path, sizeof(resource->model_file));
        for (j = 0; j < MAX_IO_COUNT; j++) {
            if (resource->has_input_file[j]) {
                resolve_path(config->base_dir, resource->input_files[j], path, sizeof(path));
                copy_string(resource->input_files[j], path, sizeof(resource->input_files[j]));
            }
            if (resource->has_output_file[j]) {
                resolve_path(config->base_dir, resource->output_files[j], path, sizeof(path));
                copy_string(resource->output_files[j], path, sizeof(resource->output_files[j]));
            }
        }
    }

    for (i = 0; i < config->link_count; i++) {
        ResourceLink *link = &config->links[i];
        link->src_resource = find_resource(config, link->src_name);
        link->dst_resource = find_resource(config, link->dst_name);
        if (link->src_resource < 0 || link->dst_resource < 0) {
            printf("Invalid link: %s:%d -> %s:%d\n",
                   link->src_name, link->src_output, link->dst_name, link->dst_input);
            return -1;
        }
        if (link->src_resource >= link->dst_resource) {
            printf("Linked source must be listed before target: %s -> %s\n",
                   link->src_name, link->dst_name);
            return -1;
        }
    }

    return 0;
}

static ResourceLink *find_input_link(MultiResourceConfig *config, int32_t resource_index,
                                     int32_t input_index)
{
    int32_t i;
    for (i = 0; i < config->link_count; i++) {
        ResourceLink *link = &config->links[i];
        if (link->dst_resource == resource_index && link->dst_input == input_index) {
            return link;
        }
    }
    return NULL;
}

static int32_t output_data_size(const tData *output)
{
    uint32_t i;
    int32_t size = output->dtype_ & 0xF;

    for (i = 0; i < output->shape_.ndim_; i++) {
        size *= output->shape_.dims_[i];
    }
    return size;
}

static int32_t init_resource(ResourceConfig *resource, int32_t *use_psram_size,
                             int32_t *use_share_size)
{
    int32_t i;

    if (load_binary_file(resource->model_file, &resource->model_data,
                         &resource->model_size) != 0) {
        return -1;
    }

    THINKER_CHECK(tGetMemoryPlan(resource->memory_list, &resource->num_memory,
                                 (int8_t *)resource->model_data,
                                 resource->model_size),
                  "tGetMemoryPlan");
    if (resource->num_memory > MAX_MEMORY_COUNT) {
        printf("Too many memory blocks: %d\n", resource->num_memory);
        return -1;
    }

    for (i = 0; i < resource->num_memory; i++) {
        int32_t mem_size = resource->memory_list[i].size_;
        if (resource->memory_list[i].dptr_ == 0) {
            int32_t aligned_size = (mem_size + 63) & (~63);
            if (resource->memory_list[i].dev_type_ == 1 ||
                resource->memory_list[i].dev_type_ == 3) {
                if (*use_psram_size + aligned_size > PSRAM_SIZE) {
                    printf("psram size exceeded when allocating %s memory block %d\n",
                           resource->name, i);
                    return -1;
                }
                resource->memory_list[i].dptr_ = (uint64_t)(g_psram_buf + *use_psram_size);
                *use_psram_size += aligned_size;
            } else if (resource->memory_list[i].dev_type_ == 2) {
                if (*use_share_size + aligned_size > SHARE_SIZE) {
                    printf("share size exceeded when allocating %s memory block %d\n",
                           resource->name, i);
                    return -1;
                }
                resource->memory_list[i].dptr_ = (uint64_t)(g_share_buf + *use_share_size);
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

    THINKER_CHECK(tModelInit(&resource->model_hdl, (int8_t *)resource->model_data,
                             resource->model_size, resource->memory_list,
                             resource->num_memory),
                  "tModelInit");
    THINKER_CHECK(tCreateExecutor(resource->model_hdl, &resource->exec_hdl,
                                  resource->memory_list, resource->num_memory),
                  "tCreateExecutor");
    printf("init resource successful: %s\n", resource->name);
    return 0;
}

static int32_t run_resource(MultiResourceConfig *config, int32_t resource_index)
{
    ResourceConfig *resource = &config->resources[resource_index];
    uint32_t input_count = tGetInputCount(resource->model_hdl);
    int32_t i;

    if (input_count > MAX_IO_COUNT) {
        printf("Too many inputs for resource: %s\n", resource->name);
        return -1;
    }

    for (i = 0; i < (int32_t)input_count; i++) {
        ResourceLink *link = find_input_link(config, resource_index, i);
        tData input;

        THINKER_CHECK(tGetInputInfo(resource->exec_hdl, i, &input), "tGetInputInfo");
        if (link) {
            ResourceConfig *src = &config->resources[link->src_resource];
            if (link->src_output >= src->output_count) {
                printf("Linked output is not ready: %s:%d\n",
                       src->name, link->src_output);
                return -1;
            }
            input.dptr_ = src->outputs[link->src_output].dptr_;
            printf("link %s:%d -> %s:%d\n", src->name, link->src_output,
                   resource->name, i);
        } else {
            int8_t *input_data = NULL;
            uint64_t input_size = 0;
            if (!resource->has_input_file[i]) {
                printf("Missing input file or link for %s input %d\n", resource->name, i);
                return -1;
            }
            if (load_binary_file(resource->input_files[i], &input_data, &input_size) != 0) {
                return -1;
            }
            input.dptr_ = input_data;
        }
        THINKER_CHECK(tSetInput(resource->exec_hdl, i, &input), "tSetInput");
    }

    THINKER_CHECK(tForward(resource->exec_hdl), "tForward");
    printf("forward successful: %s\n", resource->name);

    resource->output_count = tGetOutputCount(resource->model_hdl);
    if (resource->output_count > MAX_IO_COUNT) {
        printf("Too many outputs for resource: %s\n", resource->name);
        return -1;
    }

    for (i = 0; i < resource->output_count; i++) {
        THINKER_CHECK(tGetOutput(resource->exec_hdl, i, &resource->outputs[i]),
                      "tGetOutput");
        if (resource->has_output_file[i]) {
            int32_t size = output_data_size(&resource->outputs[i]);
            if (save_binary_file(resource->output_files[i],
                                 resource->outputs[i].dptr_, size) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

int thinker_task_test(int argc, char *argv[])
{
    int32_t use_psram_size = 0;
    int32_t use_share_size = 0;
    int32_t i;

    if (argc < 2) {
        printf("Usage: %s <multi_resource_yaml>\n", argv[0]);
        return -1;
    }

    memset(g_psram_buf, 0, PSRAM_SIZE);
    memset(g_share_buf, 0, SHARE_SIZE);
    memset(&g_config, 0, sizeof(g_config));

    if (parse_yaml_config(argv[1], &g_config) != 0) {
        return -1;
    }
    if (finalize_config(&g_config) != 0) {
        return -1;
    }

    THINKER_CHECK(tInitialize(), "tInitialize");

    for (i = 0; i < g_config.resource_count; i++) {
        if (init_resource(&g_config.resources[i], &use_psram_size, &use_share_size) != 0) {
            return -1;
        }
    }

    printf("memory used: psram=%d/%d, share=%d/%d\n",
           use_psram_size, PSRAM_SIZE, use_share_size, SHARE_SIZE);

    for (i = 0; i < g_config.resource_count; i++) {
        if (run_resource(&g_config, i) != 0) {
            return -1;
        }
    }

    tUninitialize();
    return 0;
}

int main(int argc, char *argv[])
{
    return thinker_task_test(argc, argv);
}
