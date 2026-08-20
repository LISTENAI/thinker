# Thinker 仿真平台编译指南

本文档介绍如何使用仓库中的编译脚本构建 Thinker x86 仿真平台，并说明平台切换、检查开关和 MOSS 模型资源编译方式。

> 说明
>
> - x86 仿真版本用于验证资源打包结果、推理流程和结果一致性，不直接生成芯片固件镜像。
> - Thinker 上层执行代码在仿真平台与真实目标平台之间保持通用。迁移到芯片工程时，通常只需替换目标平台对应的底层 `luna` 库和固件/BSP 库。

## 1. 编译前准备

- Linux：x86_64 Linux、CMake 3.20.1 及以上版本。
- 环境搭建：参考 [thinker_environment.md](./thinker_environment.md)。
- 执行目录：除非特别说明，命令均在 Thinker 仓库根目录执行。

## 2. Linux 快速编译

执行默认脚本：

```bash
sh scripts/x86_linux.sh
```

脚本默认执行以下操作：

- 删除并重新创建根目录下的 `build` 构建目录。
- 使用 `Debug` 构建类型和 16 个并行任务。
- 构建 `VENUSA` 平台的 x86 仿真版本。
- 生成动态库，并开启中间结果 dump。
- 开启目标平台资源匹配检查，关闭资源 CRC、参数和运行时检查。
- 关闭 MOSS 模型适配。
- 使用 `cmake -S <仓库根目录> -B <构建目录>` 配置工程，再通过 `cmake --build` 编译。

编译产物输出到仓库根目录的 `bin` 目录。

## 3. 脚本配置

Linux 脚本通过环境变量调整配置，无需直接修改脚本。

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `BUILD_DIR` | `<仓库根目录>/build` | CMake 构建目录；脚本执行时会先删除该目录 |
| `BUILD_TYPE` | `Debug` | 构建类型，常用值为 `Debug` 或 `Release` |
| `BUILD_JOBS` | `16` | 并行编译任务数 |
| `THINKER_TARGET_PLATFORM` | `VENUSA` | 仿真目标平台，可选 `VENUS`、`ARCS`、`VENUSA`，大小写不敏感 |
| `THINKER_PARAM_CHECK` | `OFF` | 是否启用算子输入、属性、数据类型等参数检查 |
| `THINKER_RUNTIME_CHECK` | `OFF` | 是否启用地址、空间大小和运行状态等运行时检查 |
| `THINKER_USE_MOSS` | `OFF` | 是否启用 MOSS 顶层模型适配 |
| `MOSS_RES_DIR` | `<仓库根目录>/moss_res` | MOSS 生成资源的根目录 |
| `MOSS_MODELS` | `anyreid;face_keypoint` | MOSS 模型名列表，模型之间使用分号分隔 |
| `THINKER_MOSS_HOST_SOURCES` | 根据 `MOSS_RES_DIR` 和 `MOSS_MODELS` 生成 | MOSS `host.c` 文件列表，使用分号分隔 |
| `THINKER_MOSS_MODEL_GETTERS` | 根据 `MOSS_MODELS` 生成 | MOSS 模型 getter 符号列表，使用分号分隔 |
| `THINKER_MOSS_MODEL_NAMES` | 根据 `MOSS_MODELS` 生成 | 注册到 Thinker 的模型名列表，使用分号分隔 |

常用示例：

```bash
# 编译 ARCS Release 仿真版本，并按本机 CPU 数设置并行任务数
BUILD_TYPE=Release \
BUILD_JOBS="$(nproc)" \
THINKER_TARGET_PLATFORM=ARCS \
sh scripts/x86_linux.sh
```

```bash
# 启用参数检查和运行时检查
THINKER_PARAM_CHECK=ON \
THINKER_RUNTIME_CHECK=ON \
sh scripts/x86_linux.sh
```

```bash
# 使用独立构建目录，避免覆盖默认 build 目录
BUILD_DIR="$PWD/build-venus" \
THINKER_TARGET_PLATFORM=VENUS \
sh scripts/x86_linux.sh
```

## 4. MOSS 模型编译

启用 MOSS 后，脚本会把每个模型对应的生成代码编译进 `libthinker.so`，并链接目标平台目录下的 `libmossruntime.so` 和 `libnnblaslinux.a`。脚本会同时设置 `THINKER_USE_NNBLAS=OFF`，避免普通 Thinker NNBLAS 链接路径与 MOSS 依赖冲突。

### 4.1 默认目录约定

假设配置如下：

```bash
MOSS_RES_DIR="$PWD/moss_res"
MOSS_MODELS="anyreid;face_keypoint"
```

脚本会生成以下 CMake 参数：

```text
THINKER_MOSS_HOST_SOURCES=<仓库根目录>/moss_res/anyreid/anyreid_host.c;<仓库根目录>/moss_res/face_keypoint/face_keypoint_host.c
THINKER_MOSS_MODEL_GETTERS=mGetModel_anyreid;mGetModel_face_keypoint
THINKER_MOSS_MODEL_NAMES=anyreid;face_keypoint
```

因此，每个模型默认需要满足以下约定：

- 生成代码路径为 `<MOSS_RES_DIR>/<模型名>/<模型名>_host.c`。
- 生成代码导出的 getter 符号为 `mGetModel_<模型名>`。
- `host.c`、getter 和模型名三个列表的元素数量及顺序必须一致。

### 4.2 按模型名编译

```bash
THINKER_USE_MOSS=ON \
MOSS_RES_DIR="$PWD/moss_res" \
MOSS_MODELS="anyreid;face_keypoint" \
sh scripts/x86_linux.sh
```

### 4.3 显式指定生成资源

如果文件名或 getter 符号不符合默认约定，可直接覆盖三个列表：

```bash
THINKER_USE_MOSS=ON \
THINKER_MOSS_HOST_SOURCES="$PWD/generated/model_a_host.c;$PWD/generated/model_b_host.c" \
THINKER_MOSS_MODEL_GETTERS="mGetModelA;mGetModelB" \
THINKER_MOSS_MODEL_NAMES="model_a;model_b" \
sh scripts/x86_linux.sh
```

> Shell 中的分号必须放在引号内，否则会被解释为命令分隔符。

## 5. 等价的手动编译

不启用 MOSS 时，默认脚本等价于以下命令：

```bash
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTHINKER_SHARED_LIB=ON \
  -DTHINKER_PROFILE=OFF \
  -DTHINKER_RESULT_DUMP=ON \
  -DTHINKER_RESULT_CRC_PRINT=OFF \
  -DTHINKER_RESOUCR_CRC_CHECK=OFF \
  -DTHINKER_TARGET_PLATFORM=VENUSA \
  -DTHINKER_TARGET_CHECK=ON \
  -DTHINKER_PARAM_CHECK=OFF \
  -DTHINKER_RUNTIME_CHECK=OFF \
  -DTHINKER_USE_MOSS=OFF
cmake --build build -j 16
```

推荐优先使用脚本及环境变量，以便后续脚本参数变更时保持一致。

## 6. CMake 编译参数

| 参数 | 脚本设置值 | 说明 |
| --- | --- | --- |
| `CMAKE_BUILD_TYPE` | `Debug` | 构建类型；可通过 `BUILD_TYPE` 覆盖 |
| `THINKER_SHARED_LIB` | `ON` | `ON` 生成动态库，`OFF` 生成静态库 |
| `THINKER_PROFILE` | `OFF` | 是否开启逐层性能统计 |
| `THINKER_RESULT_DUMP` | `ON` | 是否输出中间层结果；适合 x86 仿真调试 |
| `THINKER_RESULT_CRC_PRINT` | `OFF` | 是否打印中间层结果 CRC |
| `THINKER_RESOUCR_CRC_CHECK` | `OFF` | 是否校验模型资源 CRC；参数名沿用工程现有拼写 |
| `THINKER_TARGET_PLATFORM` | `VENUSA` | 目标平台，可选 `VENUS`、`ARCS`、`VENUSA` |
| `THINKER_TARGET_CHECK` | `ON` | 是否检查模型资源与目标平台是否匹配 |
| `THINKER_PARAM_CHECK` | `OFF` | 是否启用参数检查；由同名环境变量控制 |
| `THINKER_RUNTIME_CHECK` | `OFF` | 是否启用运行时检查；由同名环境变量控制 |
| `THINKER_USE_MOSS` | `OFF` | 是否启用 MOSS 模型适配 |

`THINKER_TARGET_PLATFORM` 一次只能选择一个平台。建议保持 `THINKER_TARGET_CHECK=ON`，以便尽早发现模型资源与仿真平台不匹配的问题。

## 7. Windows 工程生成

Windows 下可在仓库根目录运行：

```bat
scripts\x86_win.bat
```

该脚本会删除并重建 `build_win`，然后使用 `Visual Studio 14 2015 Win64` 生成 `Debug`、`VENUS` 平台工程。脚本仅执行 CMake 工程生成，不调用 Visual Studio 编译；生成完成后需在 `build_win` 中打开工程或使用 Visual Studio 构建工具继续编译。

当前 Windows 脚本采用固定参数，不支持 Linux 脚本中的环境变量和 MOSS 资源列表配置。

## 8. 编译输出

默认产物位于仓库根目录的 `bin` 下，主要包括：

- `bin/libthinker.so`：Linux Thinker 动态库。
- `bin/test_thinker`：基础推理示例。
- `bin/test_dynamic`：动态 shape 示例。

若将 `THINKER_SHARED_LIB` 改为 `OFF`，库产物为静态库。Windows 下的文件名和配置子目录由 Visual Studio 生成器决定。

## 9. 注意事项

- Linux 脚本每次都会递归删除 `BUILD_DIR`，请勿将其设置为需要保留的目录。
- 切换平台时，必须确保 `executor/libs/<平台>/linux64` 下存在对应的仿真依赖库。
- MOSS 模式要求目标平台库目录中存在 `libmossruntime.so` 和 `libnnblaslinux.a`，缺失时 CMake 会终止配置。
- `THINKER_RESULT_DUMP` 依赖文件系统输出，主要用于 x86 仿真环境；芯片侧结果对比通常使用 `THINKER_RESULT_CRC_PRINT`。
- 参数检查和运行时检查有助于定位资源或算子调用问题，但可能增加运行开销，性能测试时可保持关闭。
