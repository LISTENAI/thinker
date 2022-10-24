![logo](thinker/docs/images/Thinker_logo.png)

#### English | [简体中文](./README.md)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/thinker.svg)](https://pypi.org/project/thinker)
[![PyPI](https://badge.fury.io/py/thinker.svg)](https://badge.fury.io/py/thinker)
[![Downloads](https://pepy.tech/badge/thinker)](https://pepy.tech/project/thinker)
[![DockerHub](https://img.shields.io/docker/pulls/thinker/thinker-cpu.svg)](https://hub.docker.com/r/thinker/thinker-cpu)
[![LICENSE](https://img.shields.io/github/license/thinker-ai/thinker.svg?style=flat-square)](https://github.com/LISTENAI/thinker/blob/main/LICENSE)

# Welcome to the Thinker GitHub
* Thinker is based on the AIOT chip CSK60XX independently developed by LISTENAI Technology, combined with another open source quantitative training tool, Linger, can implement an industrial-grade deep learning platform, integrating a deep learning quantitative training and inference framework, LUNA device library, and rich tool components.
* Thinker helps developers quickly launch AI services on the VENUS chip, helping more and more AIOT products to achieve AI empowerment and realize industrial intelligent upgrades. At present, the linger+thinker tool chain has supported more than 10 AI applications of linger chips in the fields of computer vision, voice wake-up, speech recognition, offline translation, etc.
***

## Frame Features:
![thinker/docs/images/struct.png](thinker/docs/images/struct.png)
### 1. Ultra-lightweight
As shown in the above framework diagram, the Thinker framework contains two parts: offline analysis tool and engine executor The offline analysis tool contains most of the computational graph pre-processing part, including graph fusion, graph optimization and graph adaptation functions, simulating the executor functions, allocating memory in advance, and stripping the non-computational part of the executor functions as much as possible. The engine executor is mainly responsible for the computational part and other auxiliary debugging functions (optional). The code is streamlined and implemented in pure C language without any dependencies, and it can be easily deployed to CSKXX devices with basically no modifications using the research sample demos.

### 2. Generality
For regular CV models, after linger's quantization training to export computational graphs, one key package is deployed. It supports multiple input and multiple output computation graphs, dynamic input (variable input size), and 32 common quantization operators in CV models, see the operator [support list](./thinker/docs/support_quant_ops.md) for details.

### 3. High Performance
The engine actuator is specially adapted for the VENUS architecture of CSK60XX, integrating the LUNA library for core computing, giving full play to the arithmetic power of LUNA by handwriting custom instruction codes, and running common CV models under single thread can approach the peak arithmetic power of the device. The thinker+linger tool chain supports full low-precision computation (int8/int16) to improve inference performance, and adapts the relevant instructions. Compared to floating-point models, quantized models can reduce the number of parameters by 50%-75%, speeding up data access and improving computing efficiency.
***

## Quick Start
The Linger tool chain includes Linger and Thinker, which are interlinked and must be used jointly; Thinker relies on Linger's computational graph export, and both use the same standard library of operators.
The entire tool chain is used throughout the life cycle of the model landing and can be roughly divided into six phases.
### 1. Tool chain installation
- [pip installation](./thinker/docs/thinker_environment.md)
- [Source code compilation and installation](./thinker/docs/thinker_build.md)
- [docker ( linger + thinker ) ](./thinker/docs/thinker_docker.md)

### 2. Model design
After finishing the model structure design, algorithm researchers use random initialization parameters to go through the linger+thinker tool chain, which evaluates the model's parameter adaptability, memory consumption and running efficiency to avoid design rework later on when the application needs are not met.

* [Example Tutorial 2](./thinker/docs/thinker_docker.md)

### 3. Quantitative training and export of models
Linger is a plug-in for pytorch and can be imported with one click. Linger uses QAT quantization, which is completely or basically lossless for CV models. After the quantization training is completed, the model can be exported with a single click using its own tools.
* [Example Tutorial 3](./thinker/docs/thinker_docker.md)

### 4. Model analysis and packaging
Parameter checking of the computational graph, computational graph optimization and memory analysis checking using Thinker's offline tool tpacker. Finally, the computational graph is serialized into the format required by the engine executor and the runtime memory is pre-allocated.
* [Example Tutorial 4](./thinker/docs/thinker_packer.md)

### 5. Inference execution
Directly load resources serialized by offline tools. Implement computational graphs on VENUS chips with few or even zero modifications.
* [Example Tutorial 5](./thinker/docs/thinker_docker.md)

### 6. Auxiliary Functions
View operator performance statistics and intermediate result data
  * [pack and run]
  * [Example Tutorial 6](./thinker/docs/thinker_performance.md)

***
## Ability Demonstration
  * [API Interface](./thinker/docs/thinker_api.md)
  * [Support quantization OP list and restriction description](./thinker/docs/support_quant_ops.md)
***  

## Communication and Feedback
* Welcome to submit bugs and suggestions through Github Issues
* Technical Exchange WeChat Group  
***  

## Citation
- [ONNX](https://github.com/onnx/onnx)
- [MNN](https://github.com/alibaba/MNN)
- [NCNN](https://github.com/Tencent/ncnn)
- [TNN](https://github.com/Tencent/TNN)
***

## Copyright and License
[Apache-2.0 license](LICENSE)
***

