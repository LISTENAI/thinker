
# thinker辅助工具

## 内存分析报告和计算量统计
在使用离线工具分析打包时，指定配置参数"-d True"，打包完成后会在根目录下生成{model_name}_memory_report.txt文件并在打印信息中提示内存分析结果

## 计算效率统计报告
修改编译脚本scripts中的-DTHINKER_PROFILE=ON， 打开性能分析，会打印每个op的执行时间

## 中间计算结果打印
* 修改编译脚本scripts中的-DTHINKER_DUMP=ON，会dump每个op的计算结果，以txt格式保存，以tensor的名称命名（注意在根目录下新建data文件夹）
* 调用onnxinfer输出每个op的结果，与thinker的结果进行对比分析
