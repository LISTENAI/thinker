#include <map>
#include "common.h"

TEST_CASE("test Conv2d_Input_Normal_s1","[interface]")
{
    SECTION("model init")
    {
        int8_t *res;
        uint64_t res_len = 0;
        load_bin_file("./model/Conv2d_Input_Normal_s1/model.bin", &res, &res_len); 
        const char *version = tGetVersion(0);
        printf("%s\n", version);

        auto ret =tInitialize();
        REQUIRE(ret == T_SUCCESS);
        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        REQUIRE(ret == T_SUCCESS);

        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t size = memory_list[i].size_;
            if (memory_list[i].dptr_ == 0)
                memory_list[i].dptr_ = (uint64_t)malloc(sizeof(int8_t)*size);
        }

        tModelHandle model_hdl;   //typedef uint64_t
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);
        tUninitialize();
    }

    SECTION("Conv2d_Input_Normal_s1")
    {
        #define PSRAM_SIZE  (2*1024*1024)
        #define SHARE_SIZE  (640*1024)

        static int8_t g_psram_buf[PSRAM_SIZE];
        static int8_t g_share_buf[SHARE_SIZE];

        int32_t use_psram_size = 0;
        int32_t use_share_size = 0;

        memset(g_psram_buf, 0, PSRAM_SIZE);
        memset(g_share_buf, 0, SHARE_SIZE);

        int8_t *res;
        int8_t *input_data = NULL;
        int8_t *result = NULL;
        uint64_t res_len = 0;
        uint64_t input_size = 0;
        uint64_t result_size = 0;
        load_bin_file("./model/Conv2d_Input_Normal_s1/model.bin", &res, &res_len);
        load_bin_file("./model/Conv2d_Input_Normal_s1/input.bin", &input_data, &input_size);
        load_bin_file("./model/Conv2d_Input_Normal_s1/output.bin", &result, &result_size);

        tStatus ret = T_SUCCESS;
        ret = tInitialize();
        REQUIRE(ret == T_SUCCESS);

        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t mem_size = memory_list[i].size_;
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
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tExecHandle hdl;
        ret = tCreateExecutor(model_hdl, &hdl, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tData input; 
        input.dptr_ = (int8_t*)input_data;
        input.dtype_ = Float32;
        input.scale_ = 1.0f;
        input.shape_.ndim_ = 4;
        input.shape_.dims_[0] = 1;
        input.shape_.dims_[1] = 8;
        input.shape_.dims_[2] = 64;
        input.shape_.dims_[3] = 128;
        ret = tSetInput(hdl, 0, &input);
        REQUIRE(ret == T_SUCCESS);

        ret = tForward(hdl);
        REQUIRE(ret == T_SUCCESS);

        tData output;
        ret = tGetOutput(hdl, 0, &output);
        REQUIRE(ret == T_SUCCESS);
        uint32_t shape_dim = output.shape_.ndim_;
        uint32_t *shape = output.shape_.dims_;
        uint32_t size = 1;
        for (uint32_t j = 0; j < shape_dim; ++j) {
            size *= shape[j];
        }
        const float *output_data = (float *)output.dptr_;
        const float *result_data = (float *)result;
        REQUIRE(size * 4 == result_size);
        for (uint32_t j = 0; j < size; j++) 
        {
            REQUIRE(output_data[j] == result_data[j]);
        }

        ret = tReleaseExecutor(hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tModelFini(model_hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tUninitialize();
        REQUIRE(ret == T_SUCCESS);
    }
}

TEST_CASE("test_conv1d","[interface]")
{
    SECTION("model init")
    {
        int8_t *res;
        uint64_t res_len = 0;
        load_bin_file("./model/test_conv1d/model.bin", &res, &res_len);  
        auto ret =tInitialize();
        REQUIRE(ret == T_SUCCESS);
        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        REQUIRE(ret == T_SUCCESS);

        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t size = memory_list[i].size_;
            if (memory_list[i].dptr_ == 0)
                memory_list[i].dptr_ = (uint64_t)malloc(sizeof(int8_t)*size);
        }

        tModelHandle model_hdl;   //typedef uint64_t
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);
        tUninitialize();
    }

    SECTION("test_conv1d")
    {
        #define PSRAM_SIZE  (2*1024*1024)
        #define SHARE_SIZE  (640*1024)

        static int8_t g_psram_buf[PSRAM_SIZE];
        static int8_t g_share_buf[SHARE_SIZE];

        int32_t use_psram_size = 0;
        int32_t use_share_size = 0;

        memset(g_psram_buf, 0, PSRAM_SIZE);
        memset(g_share_buf, 0, SHARE_SIZE);

        int8_t *res;
        int8_t *input_data = NULL;
        int8_t *result = NULL;
        uint64_t res_len = 0;
        uint64_t input_size = 0;
        uint64_t result_size = 0;
        load_bin_file("./model/test_conv1d/model.bin", &res, &res_len);
        load_bin_file("./model/test_conv1d/input.bin", &input_data, &input_size);
        load_bin_file("./model/test_conv1d/output.bin", &result, &result_size);

        tStatus ret = T_SUCCESS;
        ret = tInitialize();
        REQUIRE(ret == T_SUCCESS);

        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t mem_size = memory_list[i].size_;
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
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tExecHandle hdl;
        ret = tCreateExecutor(model_hdl, &hdl, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tData input; 
        input.dptr_ = (int8_t*)input_data;
        input.dtype_ = Float32;
        input.scale_ = 1.0f;
        input.shape_.ndim_ = 3;
        input.shape_.dims_[0] = 1;
        input.shape_.dims_[1] = 5;
        input.shape_.dims_[2] = 5;
        ret = tSetInput(hdl, 0, &input);
        REQUIRE(ret == T_SUCCESS);

        ret = tForward(hdl);
        REQUIRE(ret == T_SUCCESS);

        tData output;
        ret = tGetOutput(hdl, 0, &output);
        REQUIRE(ret == T_SUCCESS);
        uint32_t shape_dim = output.shape_.ndim_;
        uint32_t *shape = output.shape_.dims_;
        uint32_t size = 1;
        for (uint32_t j = 0; j < shape_dim; ++j) {
            size *= shape[j];
        }
        const float *output_data = (float *)output.dptr_;
        const float *result_data = (float *)result;
        REQUIRE(size * 4 == result_size);
        for (uint32_t j = 0; j < size; j++) 
        {
            REQUIRE(output_data[j] == result_data[j]);
        }

        ret = tReleaseExecutor(hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tModelFini(model_hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tUninitialize();
        REQUIRE(ret == T_SUCCESS);
    }
}

TEST_CASE("test_batchnorm","[interface]")
{
    SECTION("model init")
    {
        int8_t *res;
        uint64_t res_len = 0;
        load_bin_file("./model/test_batchnorm/model.bin", &res, &res_len);  
        auto ret =tInitialize();
        REQUIRE(ret == T_SUCCESS);
        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        REQUIRE(ret == T_SUCCESS);

        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t size = memory_list[i].size_;
            if (memory_list[i].dptr_ == 0)
                memory_list[i].dptr_ = (uint64_t)malloc(sizeof(int8_t)*size);
        }

        tModelHandle model_hdl;   //typedef uint64_t
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);
        tUninitialize();
    }

    SECTION("test_batchnorm")
    {
        #define PSRAM_SIZE  (2*1024*1024)
        #define SHARE_SIZE  (640*1024)

        static int8_t g_psram_buf[PSRAM_SIZE];
        static int8_t g_share_buf[SHARE_SIZE];

        int32_t use_psram_size = 0;
        int32_t use_share_size = 0;

        memset(g_psram_buf, 0, PSRAM_SIZE);
        memset(g_share_buf, 0, SHARE_SIZE);

        int8_t *res;
        int8_t *input_data = NULL;
        int8_t *result = NULL;
        uint64_t res_len = 0;
        uint64_t input_size = 0;
        uint64_t result_size = 0;
        load_bin_file("./model/test_batchnorm/model.bin", &res, &res_len);
        load_bin_file("./model/test_batchnorm/input.bin", &input_data, &input_size);
        load_bin_file("./model/test_batchnorm/output.bin", &result, &result_size);

        tStatus ret = T_SUCCESS;
        ret = tInitialize();
        REQUIRE(ret == T_SUCCESS);

        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t mem_size = memory_list[i].size_;
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
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tExecHandle hdl;
        ret = tCreateExecutor(model_hdl, &hdl, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tData input; 
        input.dptr_ = (int8_t*)input_data;
        input.dtype_ = Float32;
        input.scale_ = 1.0f;
        input.shape_.ndim_ = 4;
        input.shape_.dims_[0] = 1;
        input.shape_.dims_[1] = 10;
        input.shape_.dims_[2] = 10;
        input.shape_.dims_[3] = 10;
        ret = tSetInput(hdl, 0, &input);
        REQUIRE(ret == T_SUCCESS);

        ret = tForward(hdl);
        REQUIRE(ret == T_SUCCESS);

        tData output;
        ret = tGetOutput(hdl, 0, &output);
        REQUIRE(ret == T_SUCCESS);
        uint32_t shape_dim = output.shape_.ndim_;
        uint32_t *shape = output.shape_.dims_;
        uint32_t size = 1;
        for (uint32_t j = 0; j < shape_dim; ++j) {
            size *= shape[j];
        }
        const float *output_data = (float *)output.dptr_;
        const float *result_data = (float *)result;
        REQUIRE(size * 4 == result_size);
        for (uint32_t j = 0; j < size; j++) 
        {
            REQUIRE(output_data[j] == result_data[j]);
        }

        ret = tReleaseExecutor(hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tModelFini(model_hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tUninitialize();
        REQUIRE(ret == T_SUCCESS);
    }
}

TEST_CASE("test_softmaxint","[interface]")
{
    SECTION("model init")
    {
        int8_t *res;
        uint64_t res_len = 0;
        load_bin_file("./model/test_softmaxint/model.bin", &res, &res_len);  
        auto ret =tInitialize();
        REQUIRE(ret == T_SUCCESS);
        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        REQUIRE(ret == T_SUCCESS);

        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t size = memory_list[i].size_;
            if (memory_list[i].dptr_ == 0)
                memory_list[i].dptr_ = (uint64_t)malloc(sizeof(int8_t)*size);
        }

        tModelHandle model_hdl;   //typedef uint64_t
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);
        tUninitialize();
    }

    SECTION("test_softmaxint")
    {
        #define PSRAM_SIZE  (2*1024*1024)
        #define SHARE_SIZE  (640*1024)

        static int8_t g_psram_buf[PSRAM_SIZE];
        static int8_t g_share_buf[SHARE_SIZE];

        int32_t use_psram_size = 0;
        int32_t use_share_size = 0;

        memset(g_psram_buf, 0, PSRAM_SIZE);
        memset(g_share_buf, 0, SHARE_SIZE);

        int8_t *res;
        int8_t *input_data = NULL;
        int8_t *result = NULL;
        uint64_t res_len = 0;
        uint64_t input_size = 0;
        uint64_t result_size = 0;
        load_bin_file("./model/test_softmaxint/model.bin", &res, &res_len);
        load_bin_file("./model/test_softmaxint/input.bin", &input_data, &input_size);
        load_bin_file("./model/test_softmaxint/output.bin", &result, &result_size);

        tStatus ret = T_SUCCESS;
        ret = tInitialize();
        REQUIRE(ret == T_SUCCESS);

        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t mem_size = memory_list[i].size_;
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
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tExecHandle hdl;
        ret = tCreateExecutor(model_hdl, &hdl, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tData input; 
        input.dptr_ = (int8_t*)input_data;
        input.dtype_ = Float32;
        input.scale_ = 1.0f;
        input.shape_.ndim_ = 4;
        input.shape_.dims_[0] = 1;
        input.shape_.dims_[1] = 10;
        input.shape_.dims_[2] = 10;
        input.shape_.dims_[3] = 10;
        ret = tSetInput(hdl, 0, &input);
        REQUIRE(ret == T_SUCCESS);

        ret = tForward(hdl);
        REQUIRE(ret == T_SUCCESS);

        tData output;
        ret = tGetOutput(hdl, 0, &output);
        REQUIRE(ret == T_SUCCESS);
        uint32_t shape_dim = output.shape_.ndim_;
        uint32_t *shape = output.shape_.dims_;
        uint32_t size = 1;
        for (uint32_t j = 0; j < shape_dim; ++j) {
            size *= shape[j];
        }
        const float *output_data = (float *)output.dptr_;
        const float *result_data = (float *)result;
        REQUIRE(size * 4 == result_size);
        for (uint32_t j = 0; j < size; j++) 
        {
            REQUIRE(output_data[j] == result_data[j]);
        }

        ret = tReleaseExecutor(hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tModelFini(model_hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tUninitialize();
        REQUIRE(ret == T_SUCCESS);
    }
}

TEST_CASE("test_logsoftmax","[interface]")
{
    SECTION("model init")
    {
        int8_t *res;
        uint64_t res_len = 0;
        load_bin_file("./model/test_logsoftmax/model.bin", &res, &res_len);  
        auto ret =tInitialize();
        REQUIRE(ret == T_SUCCESS);
        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        REQUIRE(ret == T_SUCCESS);

        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t size = memory_list[i].size_;
            if (memory_list[i].dptr_ == 0)
                memory_list[i].dptr_ = (uint64_t)malloc(sizeof(int8_t)*size);
        }

        tModelHandle model_hdl;   //typedef uint64_t
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);
        tUninitialize();
    }

    SECTION("test_logsoftmax")
    {
        #define PSRAM_SIZE  (2*1024*1024)
        #define SHARE_SIZE  (640*1024)

        static int8_t g_psram_buf[PSRAM_SIZE];
        static int8_t g_share_buf[SHARE_SIZE];

        int32_t use_psram_size = 0;
        int32_t use_share_size = 0;

        memset(g_psram_buf, 0, PSRAM_SIZE);
        memset(g_share_buf, 0, SHARE_SIZE);

        int8_t *res;
        int8_t *input_data = NULL;
        int8_t *result = NULL;
        uint64_t res_len = 0;
        uint64_t input_size = 0;
        uint64_t result_size = 0;
        load_bin_file("./model/test_logsoftmax/model.bin", &res, &res_len);
        load_bin_file("./model/test_logsoftmax/input.bin", &input_data, &input_size);
        load_bin_file("./model/test_logsoftmax/output.bin", &result, &result_size);

        tStatus ret = T_SUCCESS;
        ret = tInitialize();
        REQUIRE(ret == T_SUCCESS);

        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t mem_size = memory_list[i].size_;
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
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tExecHandle hdl;
        ret = tCreateExecutor(model_hdl, &hdl, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tData input; 
        input.dptr_ = (int8_t*)input_data;
        input.dtype_ = Float32;
        input.scale_ = 1.0f;
        input.shape_.ndim_ = 4;
        input.shape_.dims_[0] = 1;
        input.shape_.dims_[1] = 10;
        input.shape_.dims_[2] = 10;
        input.shape_.dims_[3] = 10;
        ret = tSetInput(hdl, 0, &input);
        REQUIRE(ret == T_SUCCESS);

        ret = tForward(hdl);
        REQUIRE(ret == T_SUCCESS);

        tData output;
        ret = tGetOutput(hdl, 0, &output);
        REQUIRE(ret == T_SUCCESS);
        uint32_t shape_dim = output.shape_.ndim_;
        uint32_t *shape = output.shape_.dims_;
        uint32_t size = 1;
        for (uint32_t j = 0; j < shape_dim; ++j) {
            size *= shape[j];
        }
        const float *output_data = (float *)output.dptr_;
        const float *result_data = (float *)result;
        REQUIRE(size * 4 == result_size);
        for (uint32_t j = 0; j < size; j++) 
        {
            REQUIRE(output_data[j] == result_data[j]);
        }

        ret = tReleaseExecutor(hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tModelFini(model_hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tUninitialize();
        REQUIRE(ret == T_SUCCESS);
    }
}

TEST_CASE("test_BabyCry","[interface]")
{
    SECTION("model init")
    {
        int8_t *res;
        uint64_t res_len = 0;
        load_bin_file("./model/test_BabyCry/model.bin", &res, &res_len);  
        auto ret =tInitialize();
        REQUIRE(ret == T_SUCCESS);
        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        REQUIRE(ret == T_SUCCESS);

        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t size = memory_list[i].size_;
            if (memory_list[i].dptr_ == 0)
                memory_list[i].dptr_ = (uint64_t)malloc(sizeof(int8_t)*size);
        }

        tModelHandle model_hdl;   //typedef uint64_t
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);
        tUninitialize();
    }

    SECTION("test_BabyCry")
    {
        #define PSRAM_SIZE  (2*1024*1024)
        #define SHARE_SIZE  (640*1024)

        static int8_t g_psram_buf[PSRAM_SIZE];
        static int8_t g_share_buf[SHARE_SIZE];

        int32_t use_psram_size = 0;
        int32_t use_share_size = 0;

        memset(g_psram_buf, 0, PSRAM_SIZE);
        memset(g_share_buf, 0, SHARE_SIZE);

        int8_t *res;
        int8_t *input_data = NULL;
        int8_t *result = NULL;
        uint64_t res_len = 0;
        uint64_t input_size = 0;
        uint64_t result_size = 0;
        load_bin_file("./model/test_BabyCry/model.bin", &res, &res_len);
        load_bin_file("./model/test_BabyCry/input.bin", &input_data, &input_size);
        load_bin_file("./model/test_BabyCry/output.bin", &result, &result_size);

        tStatus ret = T_SUCCESS;
        ret = tInitialize();
        REQUIRE(ret == T_SUCCESS);

        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t mem_size = memory_list[i].size_;
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
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tExecHandle hdl;
        ret = tCreateExecutor(model_hdl, &hdl, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tData input; 
        input.dptr_ = (int8_t*)input_data;
        input.dtype_ = Float32;
        input.scale_ = 1.0f;
        input.shape_.ndim_ = 3;
        input.shape_.dims_[0] = 1;
        input.shape_.dims_[1] = 80;
        input.shape_.dims_[2] = 40;
        ret = tSetInput(hdl, 0, &input);
        REQUIRE(ret == T_SUCCESS);

        ret = tForward(hdl);
        REQUIRE(ret == T_SUCCESS);

        tData output;
        ret = tGetOutput(hdl, 0, &output);
        REQUIRE(ret == T_SUCCESS);
        uint32_t shape_dim = output.shape_.ndim_;
        uint32_t *shape = output.shape_.dims_;
        uint32_t size = 1;
        for (uint32_t j = 0; j < shape_dim; ++j) {
            size *= shape[j];
        }
        const float *output_data = (float *)output.dptr_;
        const float *result_data = (float *)result;
        REQUIRE(size * 4 == result_size);
        for (uint32_t j = 0; j < size; j++) 
        {
            REQUIRE(output_data[j] == result_data[j]);
        }

        ret = tReleaseExecutor(hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tModelFini(model_hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tUninitialize();
        REQUIRE(ret == T_SUCCESS);
    }
}

TEST_CASE("test_iqsigmoid","[interface]")
{
    SECTION("model init")
    {
        int8_t *res;
        uint64_t res_len = 0;
        load_bin_file("./model/test_iqsigmoid/model.bin", &res, &res_len);  
        auto ret =tInitialize();
        REQUIRE(ret == T_SUCCESS);
        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        REQUIRE(ret == T_SUCCESS);

        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t size = memory_list[i].size_;
            if (memory_list[i].dptr_ == 0)
                memory_list[i].dptr_ = (uint64_t)malloc(sizeof(int8_t)*size);
        }

        tModelHandle model_hdl;   //typedef uint64_t
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);
        tUninitialize();
    }

    SECTION("test_iqsigmoid")
    {
        #define PSRAM_SIZE  (2*1024*1024)
        #define SHARE_SIZE  (640*1024)

        static int8_t g_psram_buf[PSRAM_SIZE];
        static int8_t g_share_buf[SHARE_SIZE];

        int32_t use_psram_size = 0;
        int32_t use_share_size = 0;

        memset(g_psram_buf, 0, PSRAM_SIZE);
        memset(g_share_buf, 0, SHARE_SIZE);

        int8_t *res;
        int8_t *input_data = NULL;
        int8_t *result = NULL;
        uint64_t res_len = 0;
        uint64_t input_size = 0;
        uint64_t result_size = 0;
        load_bin_file("./model/test_iqsigmoid/model.bin", &res, &res_len);
        load_bin_file("./model/test_iqsigmoid/input.bin", &input_data, &input_size);
        load_bin_file("./model/test_iqsigmoid/output.bin", &result, &result_size);

        tStatus ret = T_SUCCESS;
        ret = tInitialize();
        REQUIRE(ret == T_SUCCESS);

        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t mem_size = memory_list[i].size_;
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
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tExecHandle hdl;
        ret = tCreateExecutor(model_hdl, &hdl, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tData input; 
        input.dptr_ = (int8_t*)input_data;
        input.dtype_ = Float32;
        input.scale_ = 1.0f;
        input.shape_.ndim_ = 4;
        input.shape_.dims_[0] = 1;
        input.shape_.dims_[1] = 10;
        input.shape_.dims_[2] = 10;
        input.shape_.dims_[3] = 10;
        ret = tSetInput(hdl, 0, &input);
        REQUIRE(ret == T_SUCCESS);

        ret = tForward(hdl);
        REQUIRE(ret == T_SUCCESS);

        tData output;
        ret = tGetOutput(hdl, 0, &output);
        REQUIRE(ret == T_SUCCESS);
        uint32_t shape_dim = output.shape_.ndim_;
        uint32_t *shape = output.shape_.dims_;
        uint32_t size = 1;
        for (uint32_t j = 0; j < shape_dim; ++j) {
            size *= shape[j];
        }
        const float *output_data = (float *)output.dptr_;
        const float *result_data = (float *)result;
        REQUIRE(size * 4 == result_size);
        for (uint32_t j = 0; j < size; j++) 
        {
            REQUIRE(output_data[j] == result_data[j]);
        }

        ret = tReleaseExecutor(hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tModelFini(model_hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tUninitialize();
        REQUIRE(ret == T_SUCCESS);
    }
}

TEST_CASE("test_layernorm","[interface]")
{
    SECTION("model init")
    {
        int8_t *res;
        uint64_t res_len = 0;
        load_bin_file("./model/test_layernorm/model.bin", &res, &res_len);  
        auto ret =tInitialize();
        REQUIRE(ret == T_SUCCESS);
        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        REQUIRE(ret == T_SUCCESS);

        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t size = memory_list[i].size_;
            if (memory_list[i].dptr_ == 0)
                memory_list[i].dptr_ = (uint64_t)malloc(sizeof(int8_t)*size);
        }

        tModelHandle model_hdl;   //typedef uint64_t
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);
        tUninitialize();
    }

    SECTION("test_layernorm")
    {
        #define PSRAM_SIZE  (2*1024*1024)
        #define SHARE_SIZE  (640*1024)

        static int8_t g_psram_buf[PSRAM_SIZE];
        static int8_t g_share_buf[SHARE_SIZE];

        int32_t use_psram_size = 0;
        int32_t use_share_size = 0;

        memset(g_psram_buf, 0, PSRAM_SIZE);
        memset(g_share_buf, 0, SHARE_SIZE);

        int8_t *res;
        int8_t *input_data = NULL;
        int8_t *result = NULL;
        uint64_t res_len = 0;
        uint64_t input_size = 0;
        uint64_t result_size = 0;
        load_bin_file("./model/test_layernorm/model.bin", &res, &res_len);
        load_bin_file("./model/test_layernorm/input.bin", &input_data, &input_size);
        load_bin_file("./model/test_layernorm/output.bin", &result, &result_size);

        tStatus ret = T_SUCCESS;
        ret = tInitialize();
        REQUIRE(ret == T_SUCCESS);

        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t mem_size = memory_list[i].size_;
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
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tExecHandle hdl;
        ret = tCreateExecutor(model_hdl, &hdl, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tData input; 
        input.dptr_ = (int8_t*)input_data;
        input.dtype_ = Float32;
        input.scale_ = 1.0f;
        input.shape_.ndim_ = 4;
        input.shape_.dims_[0] = 1;
        input.shape_.dims_[1] = 10;
        input.shape_.dims_[2] = 10;
        input.shape_.dims_[3] = 10;
        ret = tSetInput(hdl, 0, &input);
        REQUIRE(ret == T_SUCCESS);

        ret = tForward(hdl);
        REQUIRE(ret == T_SUCCESS);

        tData output;
        ret = tGetOutput(hdl, 0, &output);
        REQUIRE(ret == T_SUCCESS);
        uint32_t shape_dim = output.shape_.ndim_;
        uint32_t *shape = output.shape_.dims_;
        uint32_t size = 1;
        for (uint32_t j = 0; j < shape_dim; ++j) {
            size *= shape[j];
        }
        const float *output_data = (float *)output.dptr_;
        const float *result_data = (float *)result;
        REQUIRE(size * 4 == result_size);
        for (uint32_t j = 0; j < size; j++) 
        {
            REQUIRE(output_data[j] == result_data[j]);
        }

        ret = tReleaseExecutor(hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tModelFini(model_hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tUninitialize();
        REQUIRE(ret == T_SUCCESS);
    }
}

TEST_CASE("test_shufflechannel","[interface]")
{
    SECTION("model init")
    {
        int8_t *res;
        uint64_t res_len = 0;
        load_bin_file("./model/test_shufflechannel/model.bin", &res, &res_len);  
        auto ret =tInitialize();
        REQUIRE(ret == T_SUCCESS);
        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        REQUIRE(ret == T_SUCCESS);

        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t size = memory_list[i].size_;
            if (memory_list[i].dptr_ == 0)
                memory_list[i].dptr_ = (uint64_t)malloc(sizeof(int8_t)*size);
        }

        tModelHandle model_hdl;   //typedef uint64_t
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);
        tUninitialize();
    }

    SECTION("test_shufflechannel")
    {
        #define PSRAM_SIZE  (2*1024*1024)
        #define SHARE_SIZE  (640*1024)

        static int8_t g_psram_buf[PSRAM_SIZE];
        static int8_t g_share_buf[SHARE_SIZE];

        int32_t use_psram_size = 0;
        int32_t use_share_size = 0;

        memset(g_psram_buf, 0, PSRAM_SIZE);
        memset(g_share_buf, 0, SHARE_SIZE);

        int8_t *res;
        int8_t *input_data = NULL;
        int8_t *result = NULL;
        uint64_t res_len = 0;
        uint64_t input_size = 0;
        uint64_t result_size = 0;
        load_bin_file("./model/test_shufflechannel/model.bin", &res, &res_len);
        load_bin_file("./model/test_shufflechannel/input.bin", &input_data, &input_size);
        load_bin_file("./model/test_shufflechannel/output.bin", &result, &result_size);

        tStatus ret = T_SUCCESS;
        ret = tInitialize();
        REQUIRE(ret == T_SUCCESS);

        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t mem_size = memory_list[i].size_;
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
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tExecHandle hdl;
        ret = tCreateExecutor(model_hdl, &hdl, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tData input; 
        input.dptr_ = (int8_t*)input_data;
        input.dtype_ = Float32;
        input.scale_ = 1.0f;
        input.shape_.ndim_ = 4;
        input.shape_.dims_[0] = 1;
        input.shape_.dims_[1] = 10;
        input.shape_.dims_[2] = 10;
        input.shape_.dims_[3] = 10;
        ret = tSetInput(hdl, 0, &input);
        REQUIRE(ret == T_SUCCESS);

        ret = tForward(hdl);
        REQUIRE(ret == T_SUCCESS);

        tData output;
        ret = tGetOutput(hdl, 0, &output);
        REQUIRE(ret == T_SUCCESS);
        uint32_t shape_dim = output.shape_.ndim_;
        uint32_t *shape = output.shape_.dims_;
        uint32_t size = 1;
        for (uint32_t j = 0; j < shape_dim; ++j) {
            size *= shape[j];
        }
        const float *output_data = (float *)output.dptr_;
        const float *result_data = (float *)result;
        REQUIRE(size * 4 == result_size);
        for (uint32_t j = 0; j < size; j++) 
        {
            REQUIRE(output_data[j] == result_data[j]);
        }

        ret = tReleaseExecutor(hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tModelFini(model_hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tUninitialize();
        REQUIRE(ret == T_SUCCESS);
    }
}

TEST_CASE("test_gru","[interface]")
{
    SECTION("model init")
    {
        int8_t *res;
        uint64_t res_len = 0;
        load_bin_file("./model/test_gru/model.bin", &res, &res_len);  
        auto ret =tInitialize();
        REQUIRE(ret == T_SUCCESS);
        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        REQUIRE(ret == T_SUCCESS);

        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t size = memory_list[i].size_;
            if (memory_list[i].dptr_ == 0)
                memory_list[i].dptr_ = (uint64_t)malloc(sizeof(int8_t)*size);
        }

        tModelHandle model_hdl;   //typedef uint64_t
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);
        tUninitialize();
    }

    SECTION("test_gru")
    {
        #define PSRAM_SIZE  (2*1024*1024)
        #define SHARE_SIZE  (640*1024)

        static int8_t g_psram_buf[PSRAM_SIZE];
        static int8_t g_share_buf[SHARE_SIZE];

        int32_t use_psram_size = 0;
        int32_t use_share_size = 0;

        memset(g_psram_buf, 0, PSRAM_SIZE);
        memset(g_share_buf, 0, SHARE_SIZE);

        int8_t *res;
        int8_t *input_data = NULL;
        int8_t *result = NULL;
        uint64_t res_len = 0;
        uint64_t input_size = 0;
        uint64_t result_size = 0;
        load_bin_file("./model/test_gru/model.bin", &res, &res_len);
        load_bin_file("./model/test_gru/input.bin", &input_data, &input_size);
        load_bin_file("./model/test_gru/output.bin", &result, &result_size);

        tStatus ret = T_SUCCESS;
        ret = tInitialize();
        REQUIRE(ret == T_SUCCESS);

        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t mem_size = memory_list[i].size_;
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
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tExecHandle hdl;
        ret = tCreateExecutor(model_hdl, &hdl, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tData input; 
        input.dptr_ = (int8_t*)input_data;
        input.dtype_ = Float32;
        input.scale_ = 1.0f;
        input.shape_.ndim_ = 3;
        input.shape_.dims_[0] = 1;
        input.shape_.dims_[1] = 5;
        input.shape_.dims_[2] = 10;
        ret = tSetInput(hdl, 0, &input);
        REQUIRE(ret == T_SUCCESS);

        ret = tForward(hdl);
        REQUIRE(ret == T_SUCCESS);

        tData output;
        ret = tGetOutput(hdl, 0, &output);
        REQUIRE(ret == T_SUCCESS);
        uint32_t shape_dim = output.shape_.ndim_;
        uint32_t *shape = output.shape_.dims_;
        uint32_t size = 1;
        for (uint32_t j = 0; j < shape_dim; ++j) {
            size *= shape[j];
        }
        const float *output_data = (float *)output.dptr_;
        const float *result_data = (float *)result;
        REQUIRE(size * 4 == result_size);
        for (uint32_t j = 0; j < size; j++) 
        {
            REQUIRE(output_data[j] == result_data[j]);
        }

        ret = tReleaseExecutor(hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tModelFini(model_hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tUninitialize();
        REQUIRE(ret == T_SUCCESS);
    }
}

TEST_CASE("test_OCR","[interface]")
{
    SECTION("model init")
    {
        int8_t *res;
        uint64_t res_len = 0;
        load_bin_file("./model/test_OCR/model.bin", &res, &res_len);  
        auto ret =tInitialize();
        REQUIRE(ret == T_SUCCESS);
        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        REQUIRE(ret == T_SUCCESS);

        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t size = memory_list[i].size_;
            if (memory_list[i].dptr_ == 0)
                memory_list[i].dptr_ = (uint64_t)malloc(sizeof(int8_t)*size);
        }

        tModelHandle model_hdl;   //typedef uint64_t
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);
        tUninitialize();
    }

    SECTION("test_OCR")
    {
        #define PSRAM_SIZE  (20*1024*1024)
        #define SHARE_SIZE  (640*1024)

        static int8_t g_psram_buf[PSRAM_SIZE];
        static int8_t g_share_buf[SHARE_SIZE];

        int32_t use_psram_size = 0;
        int32_t use_share_size = 0;

        memset(g_psram_buf, 0, PSRAM_SIZE);
        memset(g_share_buf, 0, SHARE_SIZE);

        int8_t *res;
        int8_t *input_data = NULL;
        int8_t *result = NULL;
        uint64_t res_len = 0;
        uint64_t input_size =  0;
        uint64_t result_size = 0;
        load_bin_file("./model/test_OCR/model.bin", &res, &res_len);
        load_bin_file("./model/test_OCR/input.bin", &input_data, &input_size);
        load_bin_file("./model/test_OCR/output.bin", &result, &result_size);

        tStatus ret = T_SUCCESS;
        ret = tInitialize();
        REQUIRE(ret == T_SUCCESS);

        int32_t num_memory = 0;
        tMemory memory_list[5];
        ret = tGetMemoryPlan((tMemory *)memory_list, &num_memory, (int8_t*)res, res_len);
        for(int32_t i = 0; i < num_memory; i++)
        {
            int32_t mem_size = memory_list[i].size_;
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
        ret = tModelInit(&model_hdl, (int8_t*)res, res_len, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tExecHandle hdl;
        ret = tCreateExecutor(model_hdl, &hdl, memory_list, num_memory);
        REQUIRE(ret == T_SUCCESS);

        tData input; 
        input.dptr_ = (int8_t*)input_data;
        input.dtype_ = Int8;
        input.scale_ = 7.0f;
        input.shape_.ndim_ = 4;
        input.shape_.dims_[0] = 1;
        input.shape_.dims_[1] = 1;
        input.shape_.dims_[2] = 32;
        input.shape_.dims_[3] = 512;
        ret = tSetInput(hdl, 0, &input);
        REQUIRE(ret == T_SUCCESS);

        ret = tForward(hdl);
        REQUIRE(ret == T_SUCCESS);

        tData output;
        ret = tGetOutput(hdl, 0, &output);
        REQUIRE(ret == T_SUCCESS);
        uint32_t shape_dim = output.shape_.ndim_;
        uint32_t *shape = output.shape_.dims_;
        uint32_t size = 1;
        for (uint32_t j = 0; j < shape_dim; ++j) {
            size *= shape[j];
        }
        const int8_t *output_data = (int8_t *)output.dptr_;
        const int8_t *result_data = (int8_t *)result;
        REQUIRE(size == result_size);
        for (uint32_t j = 0; j < size; j++) 
        {
            REQUIRE(output_data[j] == result_data[j]);
        }

        ret = tReleaseExecutor(hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tModelFini(model_hdl);
        REQUIRE(ret == T_SUCCESS);
        ret = tUninitialize();
        REQUIRE(ret == T_SUCCESS);
    }
}

