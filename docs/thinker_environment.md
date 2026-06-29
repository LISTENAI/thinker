# Linger + Thinker 开发环境搭建（本地 / Docker）

本文档用于搭建 Linger + Thinker 统一开发环境，适用于本地手动安装和 Docker 快速部署两种方式。

> 说明
>
> - 当前工具链在 x86/Linux 上的推理运行，主要用于目标平台仿真验证，不直接生成芯片固件镜像。
> - Thinker 上层代码、资源格式和主体流程在仿真环境与真实芯片平台之间通用。
> - 迁移到真实芯片平台时，通常只需要替换底层 `luna` 库和固件/BSP 库。

## 1. 选择安装方式

在开始之前，建议先根据使用场景选择环境搭建方案：

| 方式 | 适用场景 | 特点 |
| --- | --- | --- |
| 本地手动搭建 | 需要长期开发、可精细控制 CUDA / GCC / Python 版本 | 灵活度高，适合深度调试与定制 |
| Docker 镜像 | 希望快速获得可用环境，或需要隔离不同版本依赖 | 上手快、环境一致性好，适合团队协作 |

## 2. 前置检查

### 2.1 检查显卡驱动

使用以下命令查看显卡驱动信息，并确认当前驱动支持的最高 CUDA 版本：

```bash
nvidia-smi
```

示例输出：

![NVIDIA_INFO](images/nvidia_info.png)

注意：

- 如需使用更高版本的 CUDA，请先将显卡驱动升级到支持该版本的驱动。
- 如果没有输出类似信息，通常说明当前机器未安装 NVIDIA 驱动、正在使用集成显卡，或使用的是非 NVIDIA 显卡。
- 在这种情况下，仍可使用纯 CPU 环境体验工具链流程，但不建议进行实际量化训练或大规模验证，效率会明显偏低。

### 2.2 检查当前 CUDA 安装情况

```bash
nvcc -V
```

如果尚未安装 CUDA Toolkit，通常会看到类似下图的提示：

![NVCC](images/nvcc_none.png)

## 3. 方案一：本地环境（手动搭建）

### 3.1 安装 CUDA 与 cuDNN

完成显卡驱动检查后，可根据驱动支持的最高 CUDA 版本，选择合适的 CUDA Toolkit 与 cuDNN 组合。

安装建议：

- 从 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) 下载并安装与驱动兼容的 CUDA 版本。
- cuDNN 版本需与 CUDA 版本匹配。
- 如使用 Linux，建议在安装后立即完成版本校验。

安装完成后，可使用以下命令验证 CUDA：

```bash
nvcc -V
```

验证 cuDNN（Linux）：

```bash
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

如果能正常看到版本信息，则说明 cuDNN 已安装成功。

### 3.2 安装 GCC、CMake 与 Conda

除 CUDA 环境外，还需要准备基础编译与 Python 运行环境。

建议如下：

- GCC 版本需与 CUDA 版本匹配，可参考下图进行选择。

  ![cuda_gcc](images/cuda_gcc.png)

- CMake 建议使用 4.2.1 或更高版本。
- 建议安装 Anaconda 或 Miniconda，并为项目创建独立虚拟环境，避免与系统 Python 或其他项目依赖冲突。

### 3.3 创建 Python 虚拟环境

以 CUDA 12.2 环境为例，推荐使用 Python 3.10：

```bash
conda create -n linger_thinker_3x python=3.10
conda activate linger_thinker_3x
pip install -U pip
pip install -r requirements.txt
```

说明：

- 当前项目默认依赖配置更偏向 CUDA 12.2 对应的软件栈。
- 如果使用其他 CUDA 版本，请同步确认 PyTorch、TorchVision、TorchAudio 与 CUDA 的版本兼容关系。

## 4. 方案二：Docker 环境（快速部署）

Docker 方式适合快速获得隔离、稳定、可复用的 Linger + Thinker 开发环境。

说明：

- 镜像中通常已预装基础开发依赖和 Python 依赖。
- 镜像默认不包含 `linger` 与 `thinker` 源码。
- 建议将源码、模型和测试数据通过挂载目录映射到容器中，而不是直接修改镜像内部内容。

### 4.1 安装 Docker 并检查权限

如果当前机器尚未安装 Docker，可参考官方文档：

- [Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [Debian](https://docs.docker.com/engine/install/debian/)
- [CentOS](https://docs.docker.com/engine/install/centos/)
- [其他 Linux 发行版](https://docs.docker.com/engine/install/binaries/)
- [Windows](https://docs.docker.com/desktop/install/windows-install/)

建议安装 Docker 19.03 及以上版本。安装完成后，先检查版本与权限：

```bash
docker version
```

如果出现 `Got permission denied`，说明当前用户没有直接访问 Docker 的权限。建议将用户加入 `docker` 用户组，而不是长期使用 `root`：

```bash
sudo groupadd docker
sudo gpasswd -a $USER docker
newgrp docker
docker ps
```

再次执行 `docker version`，若不再出现权限错误，即可继续后续步骤。

如需加速镜像拉取，可通过修改 `/etc/docker/daemon.json` 配置镜像加速源。

### 4.2 启动 Docker 服务

```bash
sudo systemctl start docker
```

### 4.3 拉取基础镜像

基础环境通常基于 Ubuntu 20.04 LTS 提供，并按 CPU 或不同 CUDA 版本提供对应镜像。请根据显卡驱动和 CUDA 需求选择合适版本。

常见镜像示例：

```bash
docker pull listenai/linger_thinker_cpu:1.0.0
```

```bash
docker pull listenai/linger_thinker_cu118:1.0.0
```

```bash
docker pull listenai/linger_thinker_cu122:1.0.0
```

拉取成功后会看到类似输出：

```text
451b59821fee: Pull complete
aac4069a8048: Pull complete
c6ff8e4994ee: Pull complete
Digest: sha256:88211fd30fd0146c6189a5d2fecd0293f3f89b26409ba80a314255b02d435df2
Status: Downloaded newer image for listenai/linger_thinker_cu118:v1.0.0
docker.io/listenai/linger_thinker_cu118:v1.0.0
```

### 4.4 启动容器并挂载工作目录

以下示例以 `linger_thinker_cu118` 为例，通过 `-v` 将宿主机目录 `./workspace` 挂载到容器内的 `/workspace`。这样可以在不修改镜像本身的前提下，同步管理源码、模型和测试数据。

```bash
docker run -it --name linger_thinker_cu118 --gpus all \
  -v ./workspace:/workspace \
  listenai/linger_thinker_cu118:v1.0.0 bash
```

进入容器后，通常会看到类似提示：

```text
(linger_thinker_env) root@d5964a88f77b:/#
```

其中 `d5964a88f77b` 为容器 ID，表示当前已经进入容器 Shell。

### 4.5 使用预置 Conda 环境并验证依赖

镜像中默认已创建统一环境 `linger_thinker_env`，并预装对应 CUDA 版本的常用依赖。

激活环境：

```bash
conda activate linger_thinker_env
```

验证 NVIDIA 驱动、CUDA 与 PyTorch：

```bash
nvidia-smi
nvcc -V
python -c "import torch; print('Torch版本:', torch.__version__); print('CUDA是否可用:', torch.cuda.is_available())"
```

## 5. 安装 Linger 与 Thinker

无论使用本地环境还是 Docker 环境，后续安装流程基本一致。

### 5.1 拉取源码

进入工作目录后，从 GitHub 拉取最新代码：

```bash
git clone https://github.com/LISTENAI/linger.git
git clone https://github.com/LISTENAI/thinker.git
```

### 5.2 激活虚拟环境

本地环境示例：

```bash
conda activate linger_thinker_3x
```

Docker 环境示例：

```bash
conda activate linger_thinker_env
```

### 5.3 安装 Linger

```bash
cd linger
sh install.sh
```

### 5.4 安装 Thinker

```bash
cd thinker/tools
sh install.sh
```

### 5.5 验证安装结果

```bash
python
```

```python
>>> import linger
>>> import tpacker
>>> import tvalidator
>>> import tprofile
```

如果没有报错，即可认为 Linger 与 Thinker 已安装成功。

## 6. 附录

### 6.1 Conda 环境管理常用命令

查看已创建环境：

```bash
conda info --env
```

查看当前环境中的已安装包：

```bash
conda list
```

删除不再需要的环境：

```bash
conda activate base
conda remove -n <env_name> --all
```

### 6.2 常用 Docker 指令

查看容器（加 `-a` 可显示已停止容器）：

```bash
docker ps -a
```

从宿主机拷贝文件到容器：

```bash
docker cp model 66d80f4aaf1e:/workspace/
```

从容器拷贝文件到宿主机：

```bash
docker cp 66d80f4aaf1e:/models /opt
```

停止容器：

```bash
docker stop <container_id>
```

删除已停止容器：

```bash
docker rm <container_id>
```

容器退出方式对比：

| 方式 | 结果 | 再次进入方式 |
| --- | --- | --- |
| `exit` | 退出容器，容器停止但不销毁 | `docker start <容器名或容器ID>` |
| `Ctrl + D` | 退出容器，容器停止但不销毁 | `docker start <容器名或容器ID>` |
| `Ctrl + P`，再按 `Ctrl + Q` | 退出当前终端，容器继续后台运行 | `docker exec -it <容器名或容器ID> bash` |

## 7. 下一步

完成开发环境搭建后，可根据后续工作继续参考以下文档：

- 如果你要安装或升级 Thinker Python 工具链：参考 `./thinker_build.md`
- 如果你要编译 x86/Linux 仿真平台并运行示例：参考 `./thinker_compile.md`
- 如果你要继续进行模型打包、仿真运行和结果验证：可从 `README.md` / `README_EN.md` 的 Quick Start 流程继续阅读
