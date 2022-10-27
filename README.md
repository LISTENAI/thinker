![thinker_logo](thinker/docs/images/Thinker_logo.png)
--------------------------------------------------------------------------------
#### [English](./README_EN.md) | 简体中文

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/thinker.svg)](https://pypi.org/project/thinker)
[![PyPI](https://badge.fury.io/py/thinker.svg)](https://badge.fury.io/py/thinker)
[![LICENSE](https://img.shields.io/github/license/LISTENAI/thinker.svg?style=flat-square)](https://github.com/LISTENAI/thinker/blob/main/LICENSE)
[![linux](https://github.com/LISTENAI/thinker/actions/workflows/linux_x86.yml/badge.svg)](https://github.com/LISTENAI/thinker/actions/workflows/linux_x86.yml)

Thinker是聆思科技开发的轻量级神经网络推理框架，结合另一个聆思开源的量化训练工具[linger](https://github.com/LISTENAI/linger)可实现产业级深度学习平台，
集深度学习量化训练和引擎推理、LUNA器件库和丰富的工具组件于一体。聆思生态工具链（linger+thinker）是专为聆思AIOT芯片（目前只支持CSK60xx）研发，
其中推理引擎框架Thinker助力开发者在聆思VENUS芯片上快速上线AI业务，帮助越来越多嵌入式尤其是AIOT产品实现AI赋能，助力产业智能化升级。
目前linger+thinker工具链已支持聆思芯片在计算机视觉、语音唤醒、语音识别、离线翻译等领域的10多个AI场景中应用落地。
***
## 框架特点
![thinker/docs/images/struct.png](thinker/docs/images/struct-CH.png)
### 1. 超轻量
如上述框架示意图所示，Thinker框架中包含两个部分：离线分析工具和引擎执行器
离线分析工具中包含了大部分的计算图预处理部分，包括图融合、图优化和图适配等功能，模拟执行器功能，提前分配好内存，将执行器中非计算部分的功能尽量剥离。
引擎执行器主要负责计算部分以及其它辅助调试功能（可选），代码精简，纯C语言实现，无任何依赖，使用调研示例demo，基本不用修改就可方便地部署到CSKXX设备中。

### 2. 通用性
对于常规的CV模型，经过linger的量化训练导出计算图后，一键打包部署。支持多输入多输出计算图，支持动态输入（输入大小可变），支持CV模型中32个常见的量化算子，[详见算子支持列表](./thinker/docs/support_quant_ops.md)。

### 3. 高性能
引擎执行器专门针对CSK60XX的VENUS架构进行了适配，集成核心运算的LUNA库，通过手写自定义指令码方式充分发挥LUNA的算力，单线程下运行常见CV模型能接近设备算力峰值。
thinker+linger工具链支持全低精度计算(int8/int16)以提升推理性能，并对相关指令进行了适配。相对于浮点模型，量化模型能减少50%-75%的参数量，加快了数据存取速度和提升运算效率。
***

## 快速开始
聆思工具链中包括Linger和Thinker，两者相互衔接，必须联合使用。Thinker依赖于Linger的计算图导出，两者使用同一个算子标准库。
整个工具链的使用贯穿模型落地的整个生命周期，大致可以分为六个阶段：
### 1. 工具链安装
- [pip安装方式](./thinker/docs/thinker_environment.md)
- [源码编译安装方式](./thinker/docs/thinker_build.md)
- [docker镜像](./thinker/docs/thinker_docker.md)(包含了linger和thinker)
### 2. 模型设计阶段
  算法研究人员在完成模型结构设计后，使用随机初始化参数，过一遍linger+thinker工具链，工具链会对该模型的参数可适配性、内存占用和运行效率进行评估，避免后期不满求应用需求而设计返工。
  
### 3. 模型量化训练和导出
  [linger](https://github.com/LISTENAI/linger)作为pytorch的插件，一键导入。从浮点训练阶段就开始对模型参数进行规范处理，浮点模型训练完成后，添加少量代码即可进入量化训练阶段。[Linger](https://github.com/LISTENAI/linger)采用QAT量化方式，对于CV模型能做到完全无损或基本无损。
  量化训练完成后，使用自带的工具，一键导出。
  [模型量化训练和导出示例](./thinker/docs/linger.md)

### 4. 模型分析和打包
  使用Thinker离线工具tpacker对计算图的参数检查、计算图优化和内存分析检查。最后将计算图序列化成引擎执行器所需要的格式，并对运行内存进行预分配。
  [打包示例](./thinker/docs/thinker_packer.md)

### 5. 推理执行
  直接加载离线工具序列化的资源。在少量修改甚至零修改的情况下，实现计算图在VENUS芯片上的落地应用。
  [运行示例](./thinker/docs/thinker_run.md)

### 6. 辅助功能
  查看算子性能统计和中间结果数据
  [辅助工具](./thinker/docs/thinker_performance.md)

## 能力展示
* [thinker API](./thinker/docs/thinker_api.md)
* [支持量化OP列表及限制说明](./thinker/docs/support_quant_ops.md)
***  

## 交流与反馈
- 欢迎您通过 Github Issues 来提交 BUG 与建议
- 技术交流微信群
***

## 引用
- [ONNX](https://github.com/onnx/onnx)
- [MNN](https://github.com/alibaba/MNN)
- [NCNN](https://github.com/Tencent/ncnn)
- [TNN](https://github.com/Tencent/TNN)


## 版权和许可证
[Apache-2.0 license](LICENSE)
***
