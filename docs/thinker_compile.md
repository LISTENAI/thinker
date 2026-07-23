# Thinker 仿真平台编译指南

本文档介绍如何在 x86_64 Linux 环境下编译 Thinker 仿真平台，并说明常用编译参数及典型使用方式。

> 说明
>
> - 当前章节构建的是 x86/Linux 下的目标平台仿真版本，用于验证资源打包结果、推理流程和结果一致性。
> - 仿真平台并不直接生成芯片固件镜像。
> - Thinker 上层执行代码在仿真平台与真实目标平台之间保持通用；迁移到芯片工程时，通常只需要替换目标平台对应的底层 `luna` 库和固件/BSP 库。

## 1. 编译前准备

建议先完成开发环境搭建，再进行源码编译：

- 操作系统：x86_64 Linux
- 编译工具：建议使用 CMake 3.20.1 及以上版本、GCC 4.8.5 及以上版本
- Python 环境：建议参考 `./thinker_environment.md` 中的本地环境或 Docker 环境说明
- 当前目录：在 Thinker 仓库根目录执行编译命令

## 2. 快速编译

仓库中已提供默认编译脚本，可直接在根目录运行：

```bash
sh scripts/x86_linux.sh
```

默认脚本会：

- 创建并进入 `build` 目录
- 使用 CMake 生成工程
- 编译仿真平台所需的运行库和示例程序

编译完成后，产物默认输出到根目录 `bin` 下。

## 3. 手动编译示例

如需调整平台、构建类型或调试开关，可手动执行 CMake：

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug \
  -DTHINKER_SHARED_LIB=ON \
  -DTHINKER_PROFILE=OFF \
  -DTHINKER_RESULT_DUMP=ON \
  -DTHINKER_RESULT_CRC_PRINT=OFF \
  -DTHINKER_RESOUCR_CRC_CHECK=OFF \
  -DTHINKER_TARGET_PLATFORM=ARCS \
  -DTHINKER_TARGET_CHECK=ON \
  -DTHINKER_USE_NNBLAS=OFF \
  ..
make -j$(nproc)
```

如果需要切换目标平台，只需修改 `DTHINKER_TARGET_PLATFORM` 的取值即可。

## 4. 常用编译参数

### 4.1 基本参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `CMAKE_BUILD_TYPE` | 字符串 | `Debug` | 构建类型，可选 `Debug` 或 `Release` |

说明：

- `Debug` 适合开发与问题定位。
- `Release` 适合性能测试与正式交付验证。

### 4.2 平台与资源校验参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `DTHINKER_TARGET_PLATFORM` | 字符串 | `VENUS` | 选择目标平台，可选 `VENUS`、`ARCS`、`VENUSA` |
| `DTHINKER_TARGET_CHECK` | 布尔值 | `ON` | 是否启用平台资源匹配检查，开启后会校验资源文件与目标平台是否一致 |

说明：

- `DTHINKER_TARGET_PLATFORM` 在一次编译中只能选择一个目标平台。
- 推荐保留 `DTHINKER_TARGET_CHECK=ON`，以便尽早发现资源与平台不匹配的问题。

### 4.3 功能开关参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `THINKER_SHARED_LIB` | 布尔值 | `ON` | `ON` 生成动态库，`OFF` 生成静态库 |
| `THINKER_PROFILE` | 布尔值 | `OFF` | 是否开启性能统计相关能力 |
| `THINKER_RESULT_DUMP` | 布尔值 | `OFF` | 是否导出中间结果，便于调试数值问题 |
| `DTHINKER_RESULT_CRC_PRINT` | 布尔值 | `OFF` | 是否打印中间结果 CRC，用于一致性对比 |
| `DTHINKER_RESOUCR_CRC_CHECK` | 布尔值 | `ON` | 是否校验资源文件 CRC |

说明：

- `THINKER_RESULT_DUMP` 更适合 x86/Linux 仿真环境使用，因为该功能依赖文件系统输出。
- 在真实芯片平台上，通常更推荐通过 `DTHINKER_RESULT_CRC_PRINT` 比对中间结果一致性。

## 5. 常见编译场景

### 5.1 切换仿真目标平台

例如，将脚本中的：

```bash
-DTHINKER_TARGET_PLATFORM=ARCS
```

替换为：

```bash
-DTHINKER_TARGET_PLATFORM=VENUS
```

或：

```bash
-DTHINKER_TARGET_PLATFORM=VENUSA
```

即可切换为对应平台的仿真版本。

### 5.2 切换为 Release 构建

```bash
-DCMAKE_BUILD_TYPE=Release
```

适用于性能验证或发布前测试。

### 5.3 开启中间结果一致性检查

```bash
-DTHINKER_RESULT_CRC_PRINT=ON
```

适用于仿真平台与芯片平台的结果对比。

## 6. 编译输出

默认情况下，编译产物位于根目录 `bin` 下，常见文件包括：

- `bin/libthinker.so`：Thinker 动态库
- `bin/test_thinker`：基础示例程序
- `bin/test_dynamic`：动态 shape 示例程序

## 7. 注意事项

- 仿真平台当前主要在 x86_64 Linux 环境下完成验证测试。
- 默认脚本使用 `make -j16`，如本机核心数较少，可按需调整并行编译数。
- 如果修改了编译选项或目标平台，建议先清理 `build` 目录后再重新编译，避免缓存配置干扰结果。
