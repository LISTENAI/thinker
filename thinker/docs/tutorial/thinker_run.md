
使用demo中的调用示例test_thinker
给定输入数据input.bin、序列化资源model.bin、输出数据名称xxx.bin和输入图像的尺寸

格式为 test_thinker input.bin model.bin result.bin  c h w

  * input.bin : 输入的模型二进制数据
  * model.bin : 打包生成优化后的模型二进制数据
  * result.bin: thinker引擎最终生成的二进制结果数据
  * shape : c(通道),h(高度),w(宽度)

如：
```Shell
./bin/test_thinker demo/resnet50/input.bin demo/resnet50/model.bin demo/resnet50/result.bin 1 32 32
```

可根据自己的需要，修改test_thinker.c来调整模型的输入维度