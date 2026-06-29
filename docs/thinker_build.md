# Thinker 源码安装指南

本文档介绍如何从源码安装 Thinker 的 Python 工具链组件，并说明与仿真平台编译之间的关系。

> 说明
>
> - `tools/install.sh` 安装的是 Thinker 的 Python 侧离线工具包，包含 `tpacker`、`tvalidator`、`tprofile` 等命令行工具。
> - 如果你还需要编译 x86/Linux 仿真运行库和示例程序，请继续参考 `./thinker_compile.md`。
> - 仿真平台上的 Thinker 上层代码与真实目标平台保持通用，迁移到芯片工程时通常只需要替换底层 `luna` 库和固件/BSP 库。

## 1. 安装前准备

建议先完成开发环境搭建：

- Python / Conda 环境：参考 `./thinker_environment.md`
- 源码编译环境：如需仿真平台编译，参考 `./thinker_compile.md`

推荐在独立虚拟环境中安装 Thinker 工具链，避免与系统 Python 或其他项目依赖冲突。

## 2. 获取源码

可从公开仓库拉取源码：

```bash
git clone https://github.com/LISTENAI/thinker.git
cd thinker
```

如果团队内部使用镜像仓库，也可以替换为对应的内部 Git 地址。

## 3. 安装 Python 工具链

进入 `tools` 目录后执行安装脚本：

```bash
cd tools
sh install.sh
```

当前安装脚本会自动完成以下步骤：

- 卸载已安装的旧版本 `pythinker`
- 清理 `dist` 目录中的旧打包产物
- 使用 `setup.py sdist` 重新生成源码分发包
- 通过 `pip` 安装新生成的 `pythinker` 包

安装完成后，`tpacker`、`tvalidator`、`tprofile` 等命令会注册到当前 Python 环境中。

## 4. 验证安装结果

### 4.1 验证命令行工具

```bash
tpacker -h
tvalidator -h
tprofile -h
```

如果命令能够正常显示帮助信息，说明命令行入口已安装成功。

### 4.2 验证 Python 导入

```bash
python
```

```python
>>> import tpacker
>>> import tvalidator
>>> import tprofile
```

如果没有报错，即可认为 Python 工具链安装成功。

## 5. 常见使用场景

### 5.1 仅安装离线工具链

如果当前只需要：

- 进行 ONNX 图分析与资源打包
- 做结果一致性验证
- 做离线性能评估

那么执行 `tools/install.sh` 即可，无需先编译 x86 仿真运行库。

### 5.2 安装工具链后继续编译仿真平台

如果后续还需要在 PC 上运行仿真推理或执行回归测试，可在仓库根目录继续执行：

```bash
sh scripts/x86_linux.sh
```

或参考 `./thinker_compile.md` 中的手动编译方式进行构建。

## 6. 升级与重装

当源码更新后，建议重新执行安装脚本：

```bash
cd tools
sh install.sh
```

适用场景包括：

- 更新了 `tools/tpacker`、`tools/tvalidator`、`tools/tprofile` 下的 Python 代码
- 调整了 `tools/setup.py` 中的版本或入口配置
- 需要重新安装当前源码版本覆盖旧环境

## 7. 注意事项

- 安装脚本依赖当前 Python 环境中的 `pip` 和 `setuptools`，建议先确保环境可用。
- 当前包名为 `pythinker`，安装脚本会先执行卸载再安装，属于覆盖式更新。
- 若执行 `sh install.sh` 时提示依赖缺失，请先按 `./thinker_environment.md` 补齐基础环境与 `requirements.txt` 依赖。
- 如果只更新了执行器 C/C++ 代码，除了重新安装 Python 工具链外，通常还需要重新编译仿真平台产物。
