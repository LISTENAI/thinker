
# thinker分析工具

## 性能分析
修改编译脚本scripts中的-Dthinker_PROFILE=ON， 打开性能分析，会打印每个op的执行时间

## 一致性分析
* 修改编译脚本scripts中的-Dthinker_DUMP=ON，会dump每个op的处理结果，以txt保存，以tensor的名称命名
* 调用onnxinfer输出每个op的结果，与thinker的结果进行对比分析
