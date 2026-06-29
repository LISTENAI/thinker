# Thinker 一致性验证工具指南

本文档介绍如何使用 `tvalidator` 对比 Linger 训练侧与 Thinker 仿真侧的中间结果，快速定位第一处数值不一致的张量。

> 说明
>
> - `tvalidator` 在 x86/Linux 上执行的是目标平台推理流程仿真与结果验证，不是芯片固件镜像直接运行。
> - Thinker 的上层执行流程、资源格式和主体代码在仿真平台与真实目标平台之间保持通用；迁移到芯片工程时，通常只需要替换底层 `luna` 库和固件/BSP 库。
> - 工具会自动创建 `workspace/<model_name>/` 目录，并将输入文件、Linger dump 与 Thinker dump 集中保存到该目录下。

## 1. 工具作用

`tvalidator` 的核心目标是验证训练端和推理端的一致性。它会自动完成以下流程：

1. 根据 ONNX 图生成输入数据，或加载用户提供的 `.npy` 输入。
2. 调用 Linger 的 `OnnxRunner` 执行参考推理，并导出中间张量 dump。
3. 调用 Thinker 加载资源并执行仿真推理，导出中间张量 dump。
4. 按张量名前缀对两侧 dump 逐个比对，输出第一处不一致的位置。

当定位到不一致张量后，工具会打印张量名称、shape 以及前 16 个差异元素；如果本机安装了 VS Code 命令行工具 `code`，还会尝试自动拉起文件差异对比界面。

## 2. 使用前准备

建议先完成以下准备工作：

- 已按 `./thinker_build.md` 安装 `tvalidator` 命令行工具。
- Python 环境中已安装 Linger，并能够正常导入 `linger.checker.OnnxRunner`。
- 如需复用已有仿真动态库，请先完成 `./thinker_compile.md` 中的编译流程，得到 `bin/libthinker.so`。
- 如需手动指定输入，请提前准备 `.npy` 文件，且顺序与 ONNX 模型输入顺序一致。

如果不传 `-l/--lib_path`，工具会在当前工作目录下删除并重建 `build` 目录，然后自动编译 `bin/libthinker.so`。因此推荐在 Thinker 仓库根目录执行该命令。

## 3. 命令格式

```bash
tvalidator [options]
```

### 3.1 参数说明

| 参数 | 是否必选 | 说明 |
| --- | --- | --- |
| `-g`, `--onnx_path` | 是 | ONNX 模型路径 |
| `-r`, `--res_path` | 否 | 已打包好的 Thinker 资源文件路径，通常为 `.bin` |
| `-l`, `--lib_path` | 否 | `libthinker.so` 路径；如果省略，工具会尝试在当前目录自动编译 |
| `-i`, `--input_path` | 否 | 一个或多个 `.npy` 输入文件路径，顺序需与模型输入一致 |
| `--cfg` | 否 | 动态 shape 配置，格式为 `symbol=min:max:step[,symbol2=min:max:step]` |

### 3.2 参数行为说明

- `--res_path` 省略时，`tvalidator` 会自动调用 `tpacker`，在当前目录生成临时资源 `data.ignore/test.bin`。
- `--lib_path` 省略时，`tvalidator` 会根据 ONNX 图中的 `platform` 属性自动选择 `ARCS`、`VENUS` 或 `VENUSA` 重新编译仿真动态库。
- `--input_path` 省略时，工具会按照模型输入信息自动生成随机输入，同时保存一份 Linger 侧 `.npy` 输入和一份 Thinker 侧原始二进制输入。
- `--cfg` 仅用于动态 shape 图，配置名必须与 ONNX 中的动态维度符号一致，例如 `seq_len`、`yinsu_len`。

## 4. 典型用法

### 4.1 使用 ONNX 和现有动态库进行验证

适用于资源文件由工具自动打包、动态库已提前编译好的场景：

```bash
tvalidator \
  -g model/track_id/model.onnx \
  -l bin/libthinker.so
```

### 4.2 使用手动打包的资源文件进行验证

适用于你已经通过 `tpacker` 生成了固定资源文件的场景：

```bash
tvalidator \
  -g model/track_id/model.onnx \
  -r model/track_id/model.bin \
  -l bin/libthinker.so
```

### 4.3 使用固定输入进行可复现对比

当你希望复现实验结果，或需要与板端输入完全一致时，建议明确传入输入文件：

```bash
tvalidator \
  -g model/track_id/model.onnx \
  -r model/track_id/model.bin \
  -l bin/libthinker.so \
  -i data/input_0.npy data/input_1.npy
```

### 4.4 验证动态 shape 图

动态 shape 配置使用 `min:max:step` 形式：

```bash
tvalidator \
  -g model/track_id/model.onnx \
  -r model/track_id/model.bin \
  -l bin/libthinker.so \
  --cfg seq_len=32:384:32,yinsu_len=1:80:1
```

### 4.5 在仓库根目录自动编译动态库

如果当前目录就是 Thinker 仓库根目录，也可以省略 `-l`：

```bash
tvalidator -g model/track_id/model.onnx
```

此模式会重建 `./build` 并编译 `bin/libthinker.so`，首次运行耗时通常会更长。

## 5. 输出目录和中间文件

`tvalidator` 每次运行都会以 ONNX 文件名为单位创建工作区，例如：

```text
workspace/model/
├── dump_linger/
├── dump_thinker/
├── input_0_linger.npy
└── input_0_thinker.bin
```

其中：

- `dump_linger/` 保存 Linger 侧中间结果。
- `dump_thinker/` 保存 Thinker 侧中间结果。
- `*_linger.npy` 是传给 Linger 的输入副本。
- `*_thinker.bin` 是传给 Thinker 的原始输入副本。

如果 ONNX 输入名中包含 `/`、空格等字符，工具会自动转换为适合文件名的格式。

## 6. 结果说明

### 6.1 验证通过

当所有可匹配张量都一致时，命令行会输出验证通过提示，说明训练侧参考结果与 Thinker 仿真结果对齐。

### 6.2 验证失败

当出现不一致时，工具会输出：

- 第一处不一致张量名称
- 张量 shape
- 前 16 个不一致元素的位置和值
- 对应的 Linger dump 文件和 Thinker dump 文件路径

如果本机可用 `code --diff`，工具还会尝试直接打开这两个文件进行差异对比。

### 6.3 没有可比较的文件

如果提示没有找到可对比的对应文件，通常说明：

- Thinker 未成功生成 dump 文件；
- Linger 与 Thinker 使用的图或资源不一致；
- 资源文件与当前 ONNX 图不匹配。

这类问题建议优先检查 `--res_path` 是否来自当前 ONNX 图，以及构建选项是否与目标平台一致。

## 7. 调试建议

- 建议优先固定输入文件，避免随机输入造成问题难以复现。
- 如果想进一步与真实芯片平台对齐，可在板端使用相同输入并开启 `DTHINKER_RESULT_CRC_PRINT=ON`，通过 CRC 快速定位差异层。
- 若模型包含动态 shape，建议先用一组较小但稳定的 shape 组合完成首轮验证，再逐步扩大验证范围。
- 当你只是想做自动化一致性检查时，优先使用 `tvalidator`；当你需要对比板端与仿真端时，再结合 `./thinker_performance.md` 中的 CRC 调试开关使用。
