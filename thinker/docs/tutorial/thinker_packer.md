模型分析打包指令：
```Shell
tpacker -g xx.onnx [-p venus] [-d True] [-m memory] [-o model.bin]
```
* -g ：输入ONNX模型的路径，必须配置，目前仅支持从linger导出的计算图
* -p : 目标平台，目前只支持venus，选填项，默认为venus
* -d : 中间计算图导出开关，选填项，默认为False
* -m : 模型参数在venus上存放的位置，选填项，默认为psram，可选项有flash、psram
* -o : 输出的二进制模型资源路径，选填项，默认为./model.pkg

打包工具会对计算图进行图优化、模拟引擎执行以规划内存占用并将分析结果序列化到资源文件中
