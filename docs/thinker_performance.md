# Thinker 运行期调试与性能辅助指南

本文档介绍 Thinker 在运行期常用的三个调试开关：算子耗时打印、中间结果 dump 和 CRC 校验输出。

> 说明
>
> - 本文档描述的是运行期调试能力，重点用于定位数值问题、性能热点和仿真/板端差异，不同于 `./thinker_profile.md` 中的离线 cycle 估算工具。
> - x86/Linux 侧运行仍然属于目标平台推理行为仿真，但 Thinker 上层接口和执行流程与真实芯片平台保持通用；切换到目标平台时通常只需要替换底层 `luna` 库和固件/BSP 库。
> - 建议在调试阶段按需开启这些开关，发布版本或正式性能测试时再关闭，避免额外开销影响结论。

## 1. 可用调试开关

| 开关 | 默认值 | 作用 |
| --- | --- | --- |
| `THINKER_PROFILE` | `OFF` | 打印每个算子的运行耗时和部分 shape 信息 |
| `THINKER_RESULT_DUMP` | `OFF` | 将每个算子输出张量写入文本文件 |
| `DTHINKER_RESULT_CRC_PRINT` | `OFF` | 打印每个输出张量的 CRC32 和采样值，用于快速一致性比对 |

这三个开关都定义在工程顶层 `CMakeLists.txt` 中，可通过编译参数控制。

## 2. 如何开启

### 2.1 修改默认编译脚本

可直接编辑 `scripts/x86_linux.sh` 中对应的 CMake 选项，例如：

```bash
-DTHINKER_PROFILE=ON \
-DTHINKER_RESULT_DUMP=ON \
-DTHINKER_RESULT_CRC_PRINT=ON \
```

然后重新执行：

```bash
sh scripts/x86_linux.sh
```

### 2.2 手动使用 CMake

如果你希望临时切换开关，推荐使用手动构建方式：

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug \
  -DTHINKER_SHARED_LIB=ON \
  -DTHINKER_PROFILE=ON \
  -DTHINKER_RESULT_DUMP=ON \
  -DTHINKER_RESULT_CRC_PRINT=ON \
  -DTHINKER_TARGET_PLATFORM=ARCS \
  ..
make -j$(nproc)
```

调试完成后，建议重新关闭这些开关并完整编译一次，避免残留配置影响后续验证。

## 3. `THINKER_PROFILE`：算子耗时打印

开启后，运行示例程序时会在控制台输出每个算子的执行耗时。不同算子打印字段略有差异，但通常会包含：

- 算子名称
- 单层执行耗时
- 输入和输出张量 shape
- 某些算子的关键参数，例如卷积核、padding、stride

这个能力适合：

- 快速确认热点算子是否符合预期；
- 判断某次图优化是否让关键算子变快；
- 在不导出任何中间文件的情况下粗看推理路径。

说明：

- `THINKER_PROFILE` 更适合做运行期热点排查，不替代 `tprofile` 的离线全图估算。
- 开启后会引入额外打印开销，因此不建议直接把该输出作为正式性能数据。

## 4. `THINKER_RESULT_DUMP`：中间结果导出

开启后，运行时会把每个算子输出张量写入文本文件。默认文件路径格式为：

```text
workspace/data/<tensor_name>##_<shape>.txt
```

其中：

- 张量名中的 `/` 会自动替换为 `_`；
- 文件内容为逐元素文本输出；
- 输出目录基于当前工作目录创建。

这个能力适合：

- 在 x86/Linux 仿真环境下手工查看某层输出；
- 配合脚本或文本对比工具排查特定张量；
- 作为 `tvalidator` 之外的低层调试手段。

注意事项：

- 文件 dump 主要适合 Linux/Windows 这类具备文件系统的仿真环境。
- 如果更换模型后重复运行，建议先清理旧的 `workspace/data/`，避免不同模型文件混在一起造成误判。
- 当你想自动化完成训练侧与仿真侧的逐层比较时，更推荐直接使用 `./thinker_validator.md`。

## 5. `DTHINKER_RESULT_CRC_PRINT`：CRC 一致性比对

开启后，运行时会在每个输出张量完成后打印：

- 当前张量的 CRC32
- 张量数据的几个采样值
- 张量名称

该方式不依赖文件系统，尤其适合：

- 仿真平台与真实芯片平台之间做快速层级对齐；
- 串口日志场景下定位首个差异层；
- 板端环境不方便导出完整张量时做轻量验证。

推荐用法：

1. 在 x86/Linux 仿真环境使用固定输入跑一遍，记录 CRC 输出。
2. 在板端使用相同输入、相同资源文件再次运行并记录 CRC。
3. 从首个 CRC 不一致的张量开始，继续结合 dump 或 `tvalidator` 做深入定位。

## 6. 推荐调试路径

实际项目中，建议按以下顺序使用这些能力：

1. 先用 `tvalidator` 验证训练端与仿真端一致性。
2. 若仿真端与板端不一致，优先开启 `DTHINKER_RESULT_CRC_PRINT` 比较首个差异层。
3. 需要看具体数据时，再在仿真端开启 `THINKER_RESULT_DUMP` 查看中间张量。
4. 需要分析运行热点时，开启 `THINKER_PROFILE`；如果要做离线总体 cycle 评估，再使用 `tprofile`。

## 7. 注意事项

- 这些调试开关都会引入额外运行时开销，正式性能测试前请关闭。
- `THINKER_RESULT_DUMP` 和 `DTHINKER_RESULT_CRC_PRINT` 都依赖模型执行过程中实际经过的张量路径，因此输入数据应尽量固定，便于多环境对齐。
- 板端与仿真端共享 Thinker 的上层执行逻辑；若两侧差异只出现在底层实现，通常可优先检查目标平台对应的 `luna` 库、固件库和 BSP 集成方式。
