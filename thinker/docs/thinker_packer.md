thinker包括离线打包工具和引擎执行器两部分：
离线打包工具提供onnx计算图的加载解析、图优化、Layout转换、op拆分等功能，并对优化后的计算图进行内存分析，将分析结果和图信息序列化，最终生成资源文件（默认存放在根目录下model.pkg）;
引擎执行器解析资源文件，实现全静态内存分配


# Step1-模型打包
```Shell
tpacker -g xx.onnx [-p venus] [-d True] [-m memory] [-o model.bin]
```
* -g ：输入ONNX模型的路径，必须配置
* -p : 目标平台，目前只支持venus，选填项，默认为venus
* -d : 中间计算图导出开关，选填项，默认为False
* -m : 模型参数在venus上存放的位置，选填项，默认为psram，可选项有flash、psram
* -o : 输出的二进制模型资源路径，选填项，默认为./model.pkg

# Step2-引擎执行
执行编译后，会对应测试工程test_thinker，执行脚本如下
* 输入数据格式为 input.bin model.bin result.bin  c h w
  * input.bin : 输入的模型二进制数据
  * model.bin : 打包生成优化后的模型二进制数据
  * result.bin: thinker引擎最终生成的二进制结果数据
  * shape : c(通道),h(高度),w(宽度)

* 输入
```Shell
./test_thinker input.bin model.bin result.bin 8 64 128
```
