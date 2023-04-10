![logo](thinker/docs/images/thinker_logo.png)
----------------------------------------------------------------------------
#### English | [Chinese](./README.md)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/thinker.svg)](https://pypi.org/project/thinker)
[![PyPI](https://badge.fury.io/py/thinker.svg)](https://badge.fury.io/py/thinker)
[![Downloads](https://pepy.tech/badge/thinker)](https://pepy.tech/project/thinker)
[![DockerHub](https://img.shields.io/docker/pulls/thinker/thinker-cpu.svg)](https://hub.docker.com/r/thinker/thinker-cpu)
[![LICENSE](https://img.shields.io/github/license/thinker-ai/thinker.svg?style=flat-square)](https://github.com/LISTENAI/thinker/blob/main/LICENSE)

# Welcome to the thinker GitHub
thinker is based on the AIOT chip CSK60XX independently developed by LISTENAI Technology, combined with another open source quantitative training tool, Linger, can implement an industrial-grade deep learning platform, integrating a deep learning quantitative training and inference framework, LUNA device library, and rich tool components.thinker helps developers quickly launch AI services on the VENUS chip, helping more and more AIOT products to achieve AI empowerment and realize industrial intelligent upgrades. At present, the linger+thinker tool chain has supported more than 10 AI applications of linger chips in the fields of computer vision, voice wake-up, speech recognition, offline translation, etc.

## Frame Features:
thinker strips out the non-core functions in the engine executor as much as possible and puts them in offline tools to complete  
![thinker/docs/images/struct.png](thinker/docs/images/struct.png)

## Technical Highlights
### 1. Ultra-lightweight
* The optimizer and executor are separated, only the executor needs to be deployed
* The actuator architecture is based on C language, which can realize fast and efficient deployment on embedded devices
* The embedded version has no third-party library dependencies, and the compiled library is less than 200k
* Compilation is optional for operators, which can be further lightweight

### 2. Closed-loop quantitative ecology
* Cooperate with linger to realize the closed loop of model training-quantization landing
* Support onnx quantization extension
* Support full int8 quantization
* Seamlessly connect to NPU to ensure binary consistency between NPU results and quantized training results

### 3. Efficient development
* Easy to use, supports various one-click operations, and realizes training, conversion, and streaming execution in one package
* The interface is simple and easy to use, providing various calling examples, and the model deployment can be completed with only a few changes
* Unified cross-platform model, unified calling interface, unified simulation code and chip code

## Quick Start
1. [Installation](thinker/docs/tutorial/install.md)：support pip, source code, docker and other installation methods
2. [Resource Packer](thinker/docs/tutorial/thinker_packer.md): Specifying a calculation graph can automatically complete graph analysis and resource serialization
3. [Speculative Executor](thinker/docs/tutorial/thinker_run.md)：Specify the resource location, given the input and output path to complete the execution of the engine
4. [Auxiliary Tools](thinker/docs/tutorial/thinker_performance.md)：View model memory usage, efficiency evaluation and print intermediate results

## Demo
The implementation of AI algorithms basically covers six stages: model specification check, floating-point training, quantization training, model packaging, simulation engine execution, firmware burning and chip operation. The firmware programming and chip operation need to be completed on the development board of Lenses. If necessary, please contact us, and no further introduction will be made here. The flow chart of the other five stages is as follows:  
![lnn_flow_path](thinker/docs/images/lnn_flow_path.png)   
Among them, the function of model regularity check is interspersed in quantization training and model packaging.  
We first assume that the model structure is fully compatible with the underlying hardware, introduce each stage in the process, and then introduce the specific implementation of the model convention check (in the actual development process, the convention check should be carried out initially on the model structure to avoid rework in subsequent work).
### 1. Floating-point training
We are based on [pythoch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100) for function demonstration
First of all, Make sure that in the current environment, the floating-point model training can run based on pytorch.  
```Shell
python train.py -net resnet50 -gpu
```
It is recommended to use two-stage quantization training to restrict the range of floating-point training data, and only need to [add a small amount of code](thinker/docs/tutorial/resnet_modify1.md).  
To avoid conflicts, turn tesnorboard[function off](thinker/docs/tutorial/resnet_modify2.md). Start the training with the same command, and after running several epochs, a **.pth file is generated in the checkpoint/resnet50 folder

### 2. Quantization training and Export
Load the floating-point model **.pth saved in step 1, and [modify the constraint code](thinker/docs/images/linger_set2.png) to replace the floating-point operator with a quantized operator. The same command starts quantization training. After several epochs are trained, a **.pth file is also generated in the checkpoint/resnet50 folder.  
Use linger's model conversion tool to [convert the model into an onnx calculation graph](thinker/docs/images/onnx_export.png).

### 3. Model analysis and packaging
Use the thinker offline tool tpacker to pack the onnx calculation graph generated in step 2   
```Shell
tpacker -g demo/resnet28/resnet18-12-regular.onnx -d Ture -o demo/resnet28/model.bin
```

### 4. Engine Execution
Use the sample project test_thinker to run the simulation code by specifying the input data, resource file and output file name.  
```Shell
chmod +x ./bin/test_thinker
./bin/test_thinker demo/resnet28/input.bin demo/resnet28/model.bin demo/resnet28/output.bin 3 32 32 6
```
Simplify the overall processing process here, with the engine input being a normalized 3x32x32 image and the output taking max_ The ID corresponding to value is used as the classification result. The processing of input images can refer to the [Image Processing Script](tools/image_process.py), or the processed test set images can be taken from Pytorch cifar100 for testing.

Additionally, you can [view the results of operator performance and intermediate data by modifying the compilation script](think/docs/tutorial/thinker_performance.md)

### 5. Conventional check
At this stage, we do not pay attention to the effect of the model, but only pay attention to whether the structure of the model is compatible with the underlying hardware, and the function realization runs through steps 1~4
* In step 1, the model file can be exported by initializing the model parameters or training a few epochs without model convergence.
* Load the model file of step 1 in step 2. When performing quantitative training, the compliance of operator parameters will be checked. If there are any settings that do not meet the requirements, an error will be reported and exit
[error example](thinker/docs/images/resnet50_linger_err.png). The user modifies the layer parameters according to the error message and returns to step 1 until step 2 is passed.
* Load the calculation graph of step 2 in step 3, the tool will check the tensor size of the node, [if the tensor size exceeds the limit, an error will be reported and exit](thinker/docs/images/Resnet50_err.png). Otherwise, enter the memory analysis stage, and generate a [memory analysis report](thinker/docs/images/Resnet50_Mem1.png) in the root directory, and prompt the overall flash /psram/share-memory occupied. For errors that exceed the hardware limit, users can combine the error information and [Memory Analysis Report](thinker/docs/images/Resnet50_Mem2.png) to locate the calculation graph The overrun operator returns to step 1 to adjust the model structure until [through the packaging process of step 3](thinker/docs/images/Resnet50_sucess.png ).  
So far, the model compliance check has been completed, ensuring that the model can run on the chip. Model efficiency evaluation currently only supports deployment and operation on chips, please contact us for specific needs.

## Data Search
* [API Interface](thinker/docs/thinker_api.md)
* [Support quantization OP list](thinker/docs/support_quant_ops.md) and [restriction description](thinker/docs/tutorial/restrain_of_model.md)

## Communication and Feedback
- Welcome to submit bugs and suggestions through Github Issues
- Technical Exchange WeChat Group  
![concat us](thinker/docs/images/contact_me_qr.png)

## Citation
- [ONNX](https://github.com/onnx/onnx)
- [pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100)

## Copyright and License
[Apache-2.0 license](LICENSE)

