
# thinker API 使用文档

## tGetVersion
### 功能说明
* 获取当前luna库或thinker推理框架版本
### 声明定义
*   const char *tGetVersion(const int8_t idx)
### 参数说明
* index  ： 0 表示获取thinker版本，1表示获取luna库版本
* 函数返回版本字符，共三个字段VRM

## tInitialize
### 功能说明
* 全局初始化，op注册
### 声明定义
*   tStatus tInitialize()
### 参数说明
* 函数返回执行结果状态，返回IVW_SUCCESS代表 API 执行成功，其余为错误码

## tUninitialize
### 功能说明
* 全局逆初始化接口
### 声明定义
*   tStatus tUninitialize()
### 参数说明
* 函数返回执行结果状态，返回IVW_SUCCESS代表 API 执行成功，其余为错误码

## tGetMemoryPlan
### 功能说明
* 解析资源，获取运行内存大小
### 声明定义
*   tStatus tGetMemoryPlan(tMemory * memory, int32_t *num_memory, const int8_t *res, const uint64_t size)
### 参数说明
* tMemory *memory[out]        ： 返回运行内存块
* int32_t *num_memory[out]    ： 返回内存块个数
* const int8_t *res[in]       ： 传入的序列化资源
* const uint64_t size[in]     ： 传入的资源大小
* 函数返回执行结果状态，返回IVW_SUCCESS代表 API 执行成功，其余为错误码

## tModelInit
### 功能说明
* 解析资源，创建模型句柄
### 声明定义
*   tStatus tModelInit(tModelHandle *model, const int8_t *res, const uint64_t size, const tMemory *memory, const int32_t num_memory)
### 参数说明
* tModelHandle *model[out]        ： 返回的模型句柄
* const int8_t *res[in]           ： 传入的序列化资源
* uint64_t size[in]               ： 传入的资源大小
* const tMemory *memory[in]       ： 传入的内存块地址
* const int32_t num_memory[in]    ： 传入的内存块个数
* 函数返回执行结果状态，返回IVW_SUCCESS代表 API 执行成功，其余为错误码

## tModelFini
### 功能说明
* 销毁模型资源
### 声明定义
*   tStatus tModelFini(tModelHandle hdl)
### 参数说明
* tModelHandle hdl[in]       ： 传入模型句柄
* 函数返回执行结果状态，返回IVW_SUCCESS代表 API 执行成功，其余为错误码

## tGetInputCount
### 功能说明
* 获取模型输入的个数
### 声明定义
*   int32_t tGetInputCount(const tModelHandle hdl)
### 参数说明
* const tModelHandle hdl[in]  ： 传入模型句柄
* 函数返回模型输入的个数

## tGetInputName
### 功能说明
* 获取指定模型输入的名称
### 声明定义
*   const char *tGetInputName(const tModelHandle hdl, const int32_t idx)
### 参数说明
* const tModelHandle hdl[in]  ： 传入模型句柄
* const int32_t idx[in]       :  传入输入索引
* 函数返回指定输入的名称字符串

## tGetInputDataType
### 功能说明
* 获取指定模型输入的数据类型
### 声明定义
*   tDType tGetInputDataType(const tModelHandle model, const int32_t idx)
### 参数说明
* const tModelHandle hdl[in]  ： 传入模型句柄
* const int32_t idx[in]       :  传入输入索引
* 函数返回指定输入的数据类型

## tGetInputShape
### 功能说明
* 获取指定模型输入的数据类型
### 声明定义
*   tShape tGetInputShape(const tModelHandle model, const int32_t idx)
### 参数说明
* const tModelHandle hdl[in]  ： 传入模型句柄
* const int32_t idx[in]       :  传入输入索引
* 函数返回指定输入的数据shape信息

## tGetInputInfo
### 功能说明
* 获取指定模型输入的名称
### 声明定义
*   tStatus tGetInputInfo(const tExecHandle hdl, const int32_t idx, tData *input)
### 参数说明
* const tExecHandle hdl[in]  ： 传入执行器句柄
* const int32_t idx[in]      :  传入输入索引
* tData *input[out]          :  返回输入数据信息，包括tData的所有字段
* 函数返回执行结果状态，返回IVW_SUCCESS代表 API 执行成功，其余为错误码

## tGetOutputCount
### 功能说明
* 获取模型输出的个数
### 声明定义
*   int32_t tGetOutputCount(const tModelHandle hdl)
### 参数说明
* const tModelHandle hdl[in]  ： 传入模型句柄
* 函数返回模型输出的个数

## tGetOutputName
### 功能说明
* 获取指定模型输出的名称
### 声明定义
*   const char *tGetOutputName(const tModelHandle hdl, const int32_t idx)
### 参数说明
* const tModelHandle hdl[in]  ： 传入模型句柄
* const int32_t idx[in]       :  传入输出索引
* 函数返回指定输出的名称字符串

## tGetOutputDataType
### 功能说明
* 获取指定模型输出的数据类型
### 声明定义
*   tDType tGetOutputDataType(const tModelHandle model, const int32_t idx)
### 参数说明
* const tModelHandle hdl[in]  ： 传入模型句柄
* const int32_t idx[in]       :  传入输出索引
* 函数返回指定输出的数据类型

## tGetOutputShape
### 功能说明
* 获取指定模型输入的数据类型
### 声明定义
*   tShape tGetOutputShape(const tModelHandle model, const int32_t idx)
### 参数说明
* const tModelHandle hdl[in]       ： 传入模型句柄
* const int32_t idx[in]            :  传入输出索引
* 函数返回指定输出的数据shape信息

## tCreateExecutor
### 功能说明
* 创建执行器
### 声明定义
*   tStatus tCreateExecutor(const tModelHandle model, tExecHandle *hdl, const tMemory *memory_list, const int32_t num_memory)
### 参数说明
* const tModelHandle model[in]     ： 传入的模型句柄
* const tExecHandle *hdl[out]      ： 返回执行器句柄
* const tMemory *memory_list[in]   :  传入可用内存块
* const int32_t num_memory[in]     :  传入可用内存块的个数
* 函数返回执行结果状态，返回IVW_SUCCESS代表 API 执行成功，其余为错误码

## tReleaseExecutor
### 功能说明
* 销毁执行器
### 声明定义
*   tStatus tReleaseExecutor(tExecHandle hdl)
### 参数说明
* const tExecHandle hdl[in]        ： 传入执行器句柄
* 函数返回执行结果状态，返回IVW_SUCCESS代表 API 执行成功，其余为错误码
  
## tSetInput
### 功能说明
* 设置模型输入
### 声明定义
*   tStatus tSetInput(const tExecHandle hdl, const int32_t idx, const tData *input)
### 参数说明
* const tExecHandle hdl[in]         ： 传入执行器句柄
* const int32_t idx[in]             :  传入输入索引
* const tData *input[in]            ： 传入输入数据
* 函数返回执行结果状态，返回IVW_SUCCESS代表 API 执行成功，其余为错误码

## tSetInputByName
### 功能说明
* 设置模型输入
### 声明定义
*   tStatus tSetInputByName(const tExecHandle hdl, const char *name, const tData *input)
### 参数说明
* const tExecHandle hdl[in]         ： 传入执行器句柄
* const char *name[in]              :  传入输入名称
* const tData *input[in]            ： 传入输入数据
* 函数返回执行结果状态，返回IVW_SUCCESS代表 API 执行成功，其余为错误码

## tForward
### 功能说明
* 模型推理执行
### 声明定义
*   tStatus tForward(const tExecHandle hdl)
### 参数说明
* const tExecHandle hdl[in]          ： 传入执行器句柄
* 函数返回执行结果状态，返回IVW_SUCCESS代表 API 执行成功，其余为错误码

## tGetOutput
### 功能说明
* 根据索引获取模型输出
### 声明定义
*   tStatus tGetOutput(const tExecHandle hdl, const int32_t idx, tData *output)
### 参数说明
* const tExecHandle hdl[in]           ： 传入执行器句柄
* const int32_t idx[in]               :  传入输出的索引
* tData *output[out]                  ： 返回输出数据
* 函数返回执行结果状态，返回IVW_SUCCESS代表 API 执行成功，其余为错误码

## tGetOutputByName
### 功能说明
* 根据名称获取模型输出
### 声明定义
*   tStatus tGetOutputByName(const tExecHandle hdl, const char *name, tData *output)
### 参数说明
* const tExecHandle hdl[in]           ： 传入执行器句柄
* const char *name[in]                :  传入输出的名称
* tData *output[out]                  ： 返回输出数据
* 函数返回执行结果状态，返回IVW_SUCCESS代表 API 执行成功，其余为错误码

## tExecutorStart
### 功能说明
* 模型推理执行开始
### 声明定义
*   tStatus tExecutorStart(const tExecHandle hdl)
### 参数说明
* const tExecHandle hdl[in]           ： 传入执行器句柄
* 函数返回执行结果状态，返回IVW_SUCCESS代表 API 执行成功，其余为错误码
* 注意该接口为特殊情况下使用，与tExecutorStop配合使用，是tExecutorStop的逆操作

## tExecutorStop
### 功能说明
* 模型推理执行停止
### 声明定义
*   tStatus tExecutorStop(const tExecHandle hdl)
### 参数说明
* const tExecHandle hdl[in]           ： 传入执行器句柄
* 函数返回执行结果状态，返回IVW_SUCCESS代表 API 执行成功，其余为错误码
* 注意该接口为特殊情况下使用，与tExecutorStart配合使用。调用tForward后，可在另一个任务调用tExecutorStop立即停止推理并返回