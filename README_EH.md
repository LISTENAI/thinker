![logo](thinker/docs/images/Thinker_logo.png)

#### English | [简体中文](./README.md)

# Welcome to the Thinker GitHub
* Thinker is based on the AIOT chip CSK60XX independently developed by LISTENAI Technology, combined with another open source quantitative training tool, Linger, can implement an industrial-grade deep learning platform, integrating a deep learning quantitative training and inference framework, LUNA device library, and rich tool components.
* Thinker helps developers quickly launch AI services on the VENUS chip, helping more and more AIOT products to achieve AI empowerment and realize industrial intelligent upgrades. At present, the linger+thinker tool chain has supported more than 10 AI applications of linger chips in the fields of computer vision, voice wake-up, speech recognition, offline translation, etc.
***

## 1. Frame Features:
![thinker/docs/images/struct.png](thinker/docs/images/struct.png)
#### Super lightweight
The offline packaging tool performs targeted optimization of the computational graph, performs memory analysis and pre-allocation for the execution of the computational graph, and strips the off-line functions of the executor. The engine executor has no dependencies, the code is simplified, and it can be easily deployed to the terminal device without modification. With the quantization training tool linger, it supports Int8 compression and quantization of the model, and landing on the chip can reduce the volume of the model by 50-75%.

#### generality
Supports multiple inputs and multiple outputs, supports dynamic input (variable input size), and supports common operators in CV models.

#### high performance
Adapted to the VENUS chip architecture of LISTENAI, and wrote custom instruction codes to realize core computing, give full play to the computing power of LUNA, and run common CV models under a single thread close to the peak computing power of the device Support low-precision computation (int8/int16) to improve inference performance. and adapted the relevant instructions.
***

## 2. Install
- [pip installation](./thinker/docs/thinker_environment.md)
- [Source code compilation and installation](./thinker/docs/thinker_build.md)
- [docker](./thinker/docs/thinker_docker.md)
***
## 3. Quick Start
Linger's dedicated chip tool chain includes Linger+Thinker, and the two are seamlessly connected. Thinker relies on Linger's computational graph export
The use of the entire tool chain runs through the entire life cycle of the model landing, which can be roughly divided into three stages:
####  Model design
In the early stage of model structure design, developers can use the tool chain to evaluate the adaptability of the model structure to avoid design rework due to the limitation of the underlying hardware that causes the model to fail to land.

* [Example Tutorial 1](./thinker/docs/thinker_docker.md)

#### Model training
As a plugin of pytorch, linger can be imported with one click. The model is escorted from the floating-point training stage. After the floating-point model is released, a small amount of code can be added to enter the quantitative training stage. 
* [Example Tutorial 2](./thinker/docs/thinker_docker.md)
#### Model conversion
Use linger to export the quantized computation graph.
* [Example Tutorial 3](./thinker/docs/thinker_docker.md)

Use the Thinker offline tool to process the computational graph into the format required by the engine executor with one click, and pre-allocate the executor's running memory.
* [Example Tutorial 4](./thinker/docs/thinker_docker.md)
#### Inference execution
Directly load resources serialized by offline tools. In the case of a small amount of modification or even zero modification, the implementation of the calculation graph on the VENUS chip is realized. 
* [Example Tutorial 5](./thinker/docs/thinker_docker.md)
***
#### How To
  * [pack and run](./thinker/docs/thinker_packer.md)
  * [auxiliary](./thinker/docs/thinker_performance.md)
#### Acknowledgement
  * [API Interface](./thinker/docs/thinker_api.md)
#### Automated Test
  * [add more test demo](./thinker/docs/thinker_auto_test.md)
***  

## 4. Ability Demonstration
* [Supported operators](./thinker/docs/support_quant_ops.md)
* [Supported network E-D paradigm](./thinker/docs/support_quant_ops.md)  
***  

## 5. Copyright and License
Thinker is provided by Apache 2.0 license
See the release notes [RELEASE](./RELEASE.md)
***

## 6. Join Us
* Everyone is welcome to participate to build the best inference framework in the industry.
* Technical Discussion（二维码）
***