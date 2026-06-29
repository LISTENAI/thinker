# Thinker 仿真运行示例指南

Thinker 提供了多个示例程序，用于在 x86/Linux 仿真平台上验证资源文件正确性、推理流程和结果一致性。

> 说明
>
> - 本文档中的运行方式默认基于 x86/Linux 仿真平台，目的是在 PC 侧模拟目标平台的资源加载、算子调度和结果验证流程，并不等同于芯片固件镜像直接运行。
> - Thinker 的上层接口、资源格式和主体调用流程在仿真平台与真实目标平台之间保持一致。
> - 部署到芯片侧时，通常只需要替换目标平台对应的底层 `luna` 库和固件/BSP 库，并将示例中的文件读写、内存分配等逻辑接入实际 SDK 工程。

## 1. 可用示例程序

当前仓库中的常用示例包括：

- `test_thinker`：单资源、静态输入的基础推理示例
- `test_dynamic`：支持动态 shape 输入的示例
- `test_multi_resource`：支持多个资源顺序执行和资源间串联的示例
- `test_stream_graph`：支持同一计算图按 step 流式重复执行的示例

这些示例的调用流程都可以作为芯片侧集成参考。

## 2. 使用前准备

在运行示例前，通常需要先完成：

1. 环境搭建：参考 `./thinker_environment.md`
2. Python 工具安装：参考 `./thinker_build.md`
3. 仿真平台编译：参考 `./thinker_compile.md`
4. 模型资源打包：参考 `./thinker_packer.md`

默认情况下，示例程序位于根目录 `bin` 下。

## 3. 基础示例：`test_thinker`

### 3.1 命令格式

在 Thinker 根目录下运行：

```bash
./bin/test_thinker {resource.bin} {input1.bin} ... {output1.bin} ...
```

### 3.2 参数说明

- `{resource.bin}`：模型资源文件路径，即 `tpacker` 的输出结果
- `{input1.bin} ...`：模型输入文件路径，个数必须与模型实际输入个数一致
- `{output1.bin} ...`：模型输出文件路径，可选；如果提供，个数需与模型实际输出个数一致

### 3.3 适用场景

适用于：

- 单个模型资源验证
- 静态 shape 模型推理
- 基础结果正确性检查

### 3.4 注意事项

- 输入文件必须是原始二进制数据，且大小满足模型输入需求。
- 输出文件如果已存在，示例程序会以追加方式写入，建议重新运行前先删除旧输出文件。

## 4. 动态输入示例：`test_dynamic`

### 4.1 命令格式

```bash
./bin/test_dynamic {resource.bin} {num_input_files} {num_dynamic_axis} {input1.bin} ... {dynamic_axis_name:value} ... {output1.bin} ...
```

### 4.2 参数说明

- `{resource.bin}`：模型资源文件路径
- `{num_input_files}`：输入文件个数，必须与模型实际输入个数一致
- `{num_dynamic_axis}`：需要显式设置的动态轴数量
- `{input1.bin} ...`：输入文件列表
- `{dynamic_axis_name:value}`：动态轴配置，格式为 `轴名称:实际大小`
- `{output1.bin} ...`：输出文件路径，可选

### 4.3 示例

```bash
./bin/test_dynamic decoder.bin 1 1 tokens.bin seq_len:128 output.bin
```

### 4.4 注意事项

- 动态轴名称必须与计算图中的轴名称完全一致。
- 该示例适用于打包阶段已声明动态 shape 的模型资源；如果资源本身不是动态模型，`tUpdateShape()` 可能会失败。

## 5. 多资源调用示例：`test_multi_resource`

多资源示例适用于一个应用中需要顺序执行多个模型资源，并将上游模型输出作为下游模型输入的场景，例如：

- encoder / decoder
- 检测 / 分类级联
- 多阶段流水线推理

### 5.1 命令格式

```bash
./bin/test_multi_resource demo/test_multi_resource/multi_resource.yaml
```

### 5.2 YAML 配置示例

```yaml
resources:
  - name: encoder
    model: encoder.bin
    inputs:
      - file: encoder_input.bin
    outputs:
      - file: encoder_output.bin

  - name: decoder
    model: decoder.bin
    outputs:
      - file: decoder_output.bin

links:
  - from: encoder:0
    to: decoder:0
```

### 5.3 字段说明

- `resources`：资源列表，书写顺序即执行顺序
- `name`：资源名，用于在 `links` 中引用
- `model`：资源文件路径
- `inputs`：从文件加载的输入列表
- `outputs`：需要保存到文件的输出列表
- `links`：资源间连接关系
- `from`：连接源，格式为 `{source_resource}:{output_index}`
- `to`：连接目标，格式为 `{target_resource}:{input_index}`

### 5.4 行为说明

如果某个输入已经通过 `links` 连接到上游输出，则该输入不会再从文件读取，而是直接使用上游 `tGetOutput()` 获取到的地址作为输入。

## 6. 无文件系统场景的多资源配置思路

如果目标环境没有文件系统，建议将 YAML 仅作为离线描述形式，在构建阶段将其转换为 C 静态配置表。运行时不再解析 YAML，也不从文件读取模型，而是直接从数组、固定 Flash 地址或平台资源区加载。

可参考如下结构：

```c
typedef struct {
    const char *name;
    const int8_t *model_data;
    uint64_t model_size;
} tResourceDesc;

typedef struct {
    uint32_t src_resource;
    uint32_t src_output;
    uint32_t dst_resource;
    uint32_t dst_input;
} tResourceLink;
```

这种方式更适合 MCU / SoC 固件工程集成。

## 7. 流式计算图示例：`test_stream_graph`

流式示例适用于同一个模型资源需要按 step 重复执行的场景，例如：

- 大模型 decoder token-by-token 推理
- 流式语音处理
- 状态循环网络

### 7.1 命令格式

```bash
./bin/test_stream_graph {resource.bin} {step_count} {num_input_files} {num_kv_pairs} {input_idx:input_file}... {kv_input_idx:kv_output_idx}... [output_prefix]
```

### 7.2 示例

```bash
./bin/test_stream_graph decoder.bin 16 1 2 0:tokens.bin 1:3 2:4 logits
```

含义如下：

- 执行同一个计算图 16 个 step
- 普通输入 `input 0` 从 `tokens.bin` 读取
- `input 1` 与 `input 2` 作为 KV cache 的 past 输入
- `output 3` 与 `output 4` 作为 present 输出回写到 KV cache
- 非 KV 输出保存为以 `logits` 为前缀的文件

### 7.3 参数说明

- `{resource.bin}`：模型资源文件路径
- `{step_count}`：流式执行的 step 数量
- `{num_input_files}`：普通输入文件个数，不包含 KV cache 输入
- `{num_kv_pairs}`：KV cache 输入输出映射个数
- `{input_idx:input_file}`：普通输入配置，格式如 `0:tokens.bin`
- `{kv_input_idx:kv_output_idx}`：KV cache 映射，格式如 `1:3`
- `[output_prefix]`：可选输出前缀，配置后会保存非 KV 输出

### 7.4 输入组织规则

普通输入文件支持两种方式：

- 如果文件大小大于等于 `单 step 输入大小 × step_count`，示例会按 step 切片读取
- 如果文件只满足单 step 输入大小，则所有 step 复用同一份输入数据

### 7.5 KV cache 说明

示例中的 KV cache 由应用侧管理，因此在实际芯片工程中可以很方便地替换为：

- 静态内存
- PSRAM
- 环形缓存
- 平台专用内存池

## 8. 通用注意事项

- 仿真平台用于在 PC 侧复现目标平台推理流程，便于调试和结果对齐；迁移到芯片侧时，Thinker 的主要调用逻辑通常可以保持不变。
- 芯片侧集成时，通常只需要替换目标平台对应的底层 `luna` 库和固件/BSP 库，并将文件 IO 改为实际资源加载方式。
- 基础示例和动态示例的资源不可混用，否则可能出现 shape 或资源解析错误。
- 多资源示例中，生产者资源必须配置在消费者资源之前。
- 已通过 `links` 连接的输入不需要再配置输入文件；未连接的输入必须通过 `inputs` 提供。
- 动态模型运行时，动态轴名称与实际输入 shape 必须保持一致。
- 流式示例中，`kv_input_idx:kv_output_idx` 只描述缓存数据的映射关系，shape 与 dtype 需要在模型打包阶段保证匹配。
