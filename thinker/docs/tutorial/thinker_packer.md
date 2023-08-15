模型分析打包指令：
```Shell
tpacker -g xx.onnx [-p venus] [-d True] [-m psram] [-o model.bin]
```
* -g ：输入ONNX模型的路径，必须配置，目前仅支持从linger导出的计算图
* -p : 目标平台，目前只支持venus，选填项，默认为venus
* -d : 中间计算图和内存分析报告导出开关，选填项，默认为False
* -m : 指定模型参数或者激活数据的存放位置，选填项，模型参数默认存储在psram，激活数据默认存储在share-mem
       参数位置可选项有flash、psram，例如：param:flash；激活数据位置可选项有psram、share-mem。例如: entry1:psram；
* -o : 输出的二进制模型资源路径，选填项，默认为./model.pkg

打包工具会对计算图进行图优化、模拟引擎执行以规划内存占用并将分析结果序列化到资源文件中。
通过-d选项可以看到打包处理的中间图和最终计算图对应的内存分析报告。结合两者可以指定中间节点存储位置，适用于解决运行内存超限情况。
