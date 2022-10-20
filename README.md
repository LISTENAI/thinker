![thinker_logo](thinker/docs/images/Thinker_logo.png)
#### [English](./README_EH.md) | 简体中文

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/thinker.svg)](https://pypi.org/project/thinker)
[![PyPI](https://badge.fury.io/py/thinker.svg)](https://badge.fury.io/py/thinker)
[![Downloads](https://pepy.tech/badge/thinker)](https://pepy.tech/project/thinker)
[![DockerHub](https://img.shields.io/docker/pulls/thinker/thinker-cpu.svg)](https://hub.docker.com/r/thinker/thinker-cpu)
[![LICENSE](https://img.shields.io/github/license/thinker-ai/thinker.svg?style=flat-square)](https://github.com/LISTENAI/thinker/blob/main/LICENSE)

# 欢迎使用聆思科技芯片专用推理引擎框架Thinker
* Thinker是基于聆思科技自主研发的AIOT芯片CSK60xx，专门定制的轻量级神经网络推理框架，结合另一个开源量化训练工具Linger可实现产业级深度学习平台，集深度学习量化训练和引擎推理、LUNA器件库和丰富的工具组件于一体。
* Thinker助力开发者在聆思VENUS芯片上快速上线AI业务，帮助越来越多AIOT产品实现AI赋能，实现产业智能化升级。目前linger+thinker工具链已支持聆思芯片在计算机视觉、语音唤醒、语音识别、离线翻译等领域的10多个AI应用场景的使用。
***
## 一、框架特点：
![thinker/docs/images/struct.png](thinker/docs/images/struct-CH.png)
### 1. 超轻量
离线打包工具对计算图进行针对性优化，对计算图执行进行内存分析和预分配，剥离执行器的非在线功能。引擎执行器无任何依赖，代码精简，基本不用修改就可方便地部署到终端设备中。配合量化训练工具linger，支持模型进行Int8压缩与量化，落地到芯片可减少模型50-75%的体积。

### 2. 通用性
支持多输入多输出，支持动态输入（输入大小可变），CV模型中常见的算子都支持。

### 3. 高性能
对聆思的VENUS芯片架构进行了适配，编写自定义指令码以实现核心运算，充分发挥LUNA的算力，单线程下运行常见CV模型接近设备算力峰值支持低精度计算(int8/int16)以提升推理性能。并对相关指令进行了适配。
***

## 二、安装：
- [pip安装方式](./thinker/docs/thinker_environment.md)
- [源码编译安装方式](./thinker/docs/thinker_build.md)
- [docker镜像](./thinker/docs/thinker_docker.md)
***
## 三、快速开始
聆思专用芯片工具链包括Linger+Thinker，两者无缝衔接。Thinker依赖于Linger的计算图导出整个工具链的使用贯穿模型落地的整个生命周期，大致可以分为三个阶段：
#### 模型设计
  开发者在模型结构设计初期，可以使用工具链对模型结构的可适配性进行评估，避免后期由于底层硬件的限制导致模型无法落地而设计返工。
* [示例教程1](./thinker/docs/thinker_build.md)
  
#### 模型训练
  linger作为pytorch的插件，一键导入。从浮点训练阶段就开始对模型进行保驾护航，浮点模型出来后，添加少量代码即可进入量化训练阶段。
* [示例教程2](./thinker/docs/thinker_build.md)

#### 模型转换
 使用linger导出量化后的计算图。
* [示例教程3](./thinker/docs/thinker_build.md)
  
使用Thinker离线工具一键将计算图处理成引擎执行器所需要的格式，并对执行器的运行内存进行预分配。

* [示例教程4](./thinker/docs/thinker_build.md)
  
#### 推理执行
  直接加载离线工具序列化的资源。在少量修改甚至零修改的情况下，实现计算图在VENUS芯片上的落地。
* [示例教程5](./thinker/docs/thinker_build.md)
***
#### How To
  * [打包运行](./thinker/docs/thinker_packer.md)
  * [辅助工具](./thinker/docs/thinker_performance.md)
#### 参考资料
  * [API说明](./thinker/docs/thinker_api.md)
#### 自动化测试
  * [添加测试用例](./thinker/docs/thinker_auto_test.md)
***  

## 四、能力展示
* [支持的算子 算子支持列表](./thinker/docs/support_quant_ops.md)
* [支持的网络  E-D范式](./thinker/docs/support_quant_ops.md)  
***  


## 五、版权和许可证
Thinker由Apache 2.0 license提供
版本说明请参阅[RELEASE](./RELEASE.md)
***

## 六、联系我们
* 欢迎大家参与，协同共建，打造高性能推理框架
* 技术交流群（二维码）
***
