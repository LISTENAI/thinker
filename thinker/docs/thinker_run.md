thinker包括离线打包工具和引擎执行器两部分：
离线打包工具提供onnx计算图的加载解析、图优化、Layout转换、op拆分等功能;
引擎执行器解析资源文件，实现全静态内存分配.

# Step2-引擎执行
执行编译后，会对应测试工程test_thinker，执行脚本如下

* 输入数据格式为 input.bin model.bin result.bin  c h w

  * input.bin : 输入的模型二进制数据
  * model.bin : 打包生成优化后的模型二进制数据
  * result.bin: thinker引擎最终生成的二进制结果数据
  * shape : c(通道),h(高度),w(宽度)


以 Resnet50 模型为例，输入:
```Shell
./bin/test_thinker demo/resnet50/input.bin demo/resnet50/model.bin demo/resnet50/result.bin 1 32 32
```