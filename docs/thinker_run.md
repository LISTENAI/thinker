thinker 框架提供了两个测试工程，用于验证模型资源正确性及比对计算结果一致性
- 1.test_thinker：适用于单个模型资源导入，静态输入的测试。
- 2.test_dynamic：支持动态形状输入的测试用例，可指定实际输入大小。

所有示例工程的调用逻辑均可作为实际应用引擎的参考

## 基础示例
在thinker的根目录下运行
```Shell
./bin/test_thinker {resource.bin} {input1.bin} ... {output1.bin} ...
```
## 参数说明
* {resource.bin}：表示模型资源文件路径，是tpacker打包的输出结果
* {input1.bin} ... : 表示模型的输入文件路径，确保文件路径个数与计算图中的实际输入个数保持一致
* {output1.bin} ...: 表示模型的输出文件路径，确保文件路径个数与计算图中的实际输出个数保持一致

## 动态输入示例
thinker的根目录下运行
```Shell
./bin/test_dynamic {resource.bin} {num_input} {num_dynamic_axis} {input1.bin} ... {dynamic_axis_name:value} ... {output1.bin} ...
```
## 参数说明
* 相对于基础示例，增加了{num_input}参数，用于表示模型有几个输入
* 相对于基础示例，增加了{num_dynamic_axis}参数，用于表示模型输入中几个动态维度需要设置实际的size（如果实际size和打包时最大尺寸一致也可不设置）；
* {input1.bin}...的个数要与num_input保持一致;
* 相对于基础示例，增加了{dynamic_axis_name:value}参数，用于设置计算图输入中动态轴的名称和对应的实际大小，字段的个数要求与num_dynamic_axis保持一致

## 注意事项
* 确保所有输入文件的大小不小于模型的实际需求；
* 模型的输入和输出数量需与实际配置一致；
* 基础示例和动态示例资源不可互用，否则会报错；
* 动态输入的size调整是根据轴的名称匹配，所以命令行中轴名称一定要与计算图中的名称保持一致；
* 对于动态输入的情况，引擎集成时可简化操作，一种是将tUpdateShape()的第二个参数和第三个参数固定（适合特定计算图，不通用），另一种是直接操作包含动态轴的输入，最后统一调用tUpdateShape()接口进行shape更新（参考注释代码173~192）。
