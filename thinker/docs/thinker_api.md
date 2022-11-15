
# thinker API 使用文档

## tInitialize
### thinker全局初始化接口
### 参数说明

## tUninitialize
### thinker全局逆初始化接口
### 参数说明

## tModelInit
### 创建模型资源
### 参数说明
* tModelHandle *model[out]        ： 返回的模型句柄
* const int8_t *res[in]           ： 传入的资源内存
* uint64_t size[in]               ： 传入的资源尺寸
* int32_t dev_id[in]              ： 设备ID，默认为0
* const char* plugin_lib_path[in] ：第三方so路径，缺省null

## tModelFini
### 销毁模型资源
### 参数说明
* tModelHandle model[in]  ： 模型句柄

## tCreateExecutor
### 创建执行器
### 参数说明
* tModelHandle model[in]          ： 输入的模型句柄
* tExecHandle *hdl[out]           ： 返回执行器句柄

## tReleaseExecutor
### 销毁执行器
### 参数说明
* tExecHandle hdl[in]           ： 输入执行器句柄
  
## tSetInput
### 输入
### 参数说明
* tExecHandle hdl[in]  ： 执行器句柄
* int idx[in]          :  输入的索引
* tData *input[in]         ：输入数据

## tForward
### 执行
### 参数说明
* tExecHandle hdl[in]  ： 执行器句柄

## tGetOutput
### 获取输出
### 参数说明
* tExecHandle hdl[in]  ： 执行器句柄
* int idx[in]          :  输出的索引
* tData *input[out]    ：返回输出数据