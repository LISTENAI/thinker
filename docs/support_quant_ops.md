# 算子支持列表及说明

本文档汇总 Thinker 当前版本支持的所有 OP 及对应的命名规则、常用量化术语以及各 OP 的输入、输出、中间计算精度、属性信息和类型约束。
不同硬件平台的精度和参数限制可能不同。

## 术语说明

### 量化方式

linger3.x之前版本导出的onnx计算图中部分 OP 属性中包含 `platform_quant`，用于标识平台相关的量化方法。
linger3.x及以后版本导出的onnx计算图中使用  `quant_mode` 属性标识平台相关的量化方法。

- `luna_quant` 或 `floor_add` ：浮点到定点的 round 采用 `(x + 0.5).floor()`。
- `floor` ：浮点到定点的 round 采用 `x.floor()`。

### Scale 说明

硬件友好型量化，scale都取2的幂次方

- `scale_i`：Input 的 scale，`scale_x`、`scale_1`、`scale_2`、`scale_y` 同理；`bits` 可取 8、16 等。

$$\frac{2^{bits-1}-1}{running\_i}$$

- `scale_w`：Weight 的 scale，`scale_iw`、`scale_hw` 同理；`bits` 可取 8、16 等。

$$\frac{2^{bits-1}-1}{weight.abs().max()}$$

- `scale_o`：Output 的 scale；`bits` 可取 8、16 等。

$$\frac{2^{bits-1}-1}{running\_o}$$

### ONNX 类型值

本节仅列出 ONNX 类型枚举值，不表示 Thinker runtime 支持对应 tensor 计算；除 `Quant` 和 `Dequant` 的模型边界转换外，当前不支持 Float32 tensor 计算。

#### 类型说明

| Group | Types | Description |
| --- | --- | --- |
| Floating Point Types | FLOAT16, FLOAT32, FLOAT64 | Values adhering to the IEEE 754-2008 standard representation of floating-point data. |
| Signed Integer Types | INT8, INT16, INT32, INT64 | Signed integers are supported for 8-64 bit widths. |
| Unsigned Integer Types | UINT8, UINT16 | Unsigned integers of 8 or 16 bits are supported. |
| Complex Types | COMPLEX64, COMPLEX128 | A complex number with either 32- or 64-bit real and imaginary parts. |
| Other | STRING | Strings represent textual data. All strings are encoded using UTF-8. |
| Other | BOOL | Boolean values represent data with only two values, typically true and false. |

#### 类型和值

| 类型 | 值 |
| --- | --- |
| UNDEFINED | 0 |
| FLOAT32 | 1 |
| UINT8 | 2 |
| INT8 | 3 |
| UINT16 | 4 |
| INT16 | 5 |
| INT32 | 6 |
| INT64 | 7 |
| STR | 8 |
| BOOL | 9 |
| FLOAT16 | 10 |
| UINT32 | 12 |
| UINT64 | 13 |
| COMPLEX64 | 14 |
| COMPLEX128 | 15 |
| BFLOAT16 | 16 |

## 算子说明

### 快速导航

|  |  |  |  |
| --- | --- | --- | --- |
| [ArgMax](#argmax) | [AvgPool2dInt](#avgpool2dint) | [BatchNorm1dInt](#batchnorm1dint) | [BatchNorm2dInt](#batchnorm2dint) |
| [BmmInt](#bmmint) | [Cast](#cast) | [Clip](#clip) | [Concat](#concat) |
| [Conv1dInt](#conv1dint) | [Conv2dInt](#conv2dint) | [ConvTranspose2dInt](#convtranspose2dint) | [Expand](#expand) |
| [FFNInt](#ffnint) | [Flatten](#flatten) | [Gather](#gather) | [QGelu](#qgelu) |
| [GluInt](#gluint) | [GRUInt](#gruint) | [iqAdd](#iqadd) | [iqDiv](#iqdiv) |
| [iqMul](#iqmul) | [iqPad](#iqpad) | [iqSigmoid](#iqsigmoid) | [iqSub](#iqsub) |
| [iqSum](#iqsum) | [iqTanh](#iqtanh) | [iqVar](#iqvar) | [LayerNormInt](#layernormint) |
| [LinearInt](#linearint) | [LogSoftmaxInt](#logsoftmaxint) | [LSTMInt](#lstmint) | [MaxPool](#maxpool) |
| [MultiHeadAttentionInt](#multiheadattentionint) | [PReLU](#prelu) | [ReLU](#relu) | [ReLUx](#relux) |
| [Quant](#quant) | [Dequant](#dequant) | [Requant](#requant) | [Reshape](#reshape) |
| [Resize](#resize) | [Shape](#shape) | [ShuffleChannel](#shufflechannel) | [Slice](#slice) |
| [SoftmaxInt](#softmaxint) | [SparifyFFNInt](#sparifyffnint) | [Split](#split) | [Squeeze](#squeeze) |
| [QSwish](#qswish) | [Tile](#tile) | [topN](#topn) | [topN2](#topn2) |
| [Transpose](#transpose) | [Unsqueeze](#unsqueeze) |  |  |

### 通用阅读规则

- 本文的“有效支持”取离线分析器、共享执行 wrapper、目标平台 backend 三者的交集。设备清单、历史 workbook 或单个 kernel 中出现名称，不等于模型可以被正常打包并执行。
- 除 `Quant` 和 `Dequant` 的模型边界转换外，当前不支持 Float32 tensor 输入、输出或计算；浮点 scale 属性和量化舍入说明仅属于元数据与配置。
- 平台矩阵中的“支持”表示当前正常 tpacker 路径可达；“runtime-only”表示存在专用 wrapper/backend，但缺少正常离线分析器，普通 ONNX 打包不可达；`-` 表示当前有效交集不支持。
- ONNX 中 `scale_x`、`scale_w`、`scale_o` 等属性是正的 2 的幂；离线分析将其转换为指数 `q = log2(scale)`，运行时 tensor 的 `scale_` 保存 `q`。文中 shift 均是指数差，不是浮点 scale 相减。
- `SM` 表示 ShareMem，`PSRAM` 表示外部内存。平台 kernel 不能直接处理 PSRAM 时，tpacker 会分配 SM workspace 做 DMA、转置、分块或输出暂存；实际容量以离线分析结果为准。
- 除非单节另有说明，量化算子只支持 zero point 0，shape 维度必须为正，workspace 为 SM 中的 Int8 字节缓冲区。
- 名称按注册表规范化：`BatchNormInt` 归入 `BatchNorm2dInt`，`DeConv2dInt` 归入 `ConvTranspose2dInt`，workbook 中的 `TopN` 对应注册名 `topN`。LSTM/GRU 的可选状态或序列长度输入是同一算子的输入变体，不是独立算子。
- 源文件名区分大小写；特别注意共享 wrapper `conv2dInt.c`、`conv1dInt.c`、`gluInt.c` 与平台头文件的小写名称。

### ArgMax

#### 功能说明

沿最后一维查找最大值，输出把值和索引编码在首维为 2 的 Int32 tensor 中；不支持 `select_last_index=1`。

#### 接口定义

- 输入：一个 rank >= 1 tensor；属性 `axis`。
- 输出：Int32，归约维变为 1，首维变为 2。

#### 平台支持矩阵

| 平台 | 输入 dtype | 状态 |
| --- | --- | --- |
| Venus | Int8 | 支持 |
| ARCS | Int8 / Int32 | 支持 |
| VenusA | Int8 / Int16 / Int32 | 支持 |

#### 参数、Shape 与内存约束

- 仅最后一维；rank > 1 时要求 `shape[0] == 1`，归约长度大于 0。
- 固定申请 8 字节 SM workspace。

#### 源码依据

- `executor/core/ops/argmax.c`；`executor/core/ops/{venus,arcs,venusA}/argmax.h`
- `tools/tpacker/graph_analysis/ops/ArgMax.py`

### AvgPool2dInt

#### 功能说明

对 NCHW Int8 feature map 执行二维平均池化并按 `scale_o` 重标定。

#### 接口定义

- 输入/输出：`x[N,C,H,W] -> y[N,C,Ho,Wo]`，均为 Int8。
- 重要属性：`kernel_shape`、`strides`、`pads`、`ceil_mode=0`、`scale_x`、`scale_o`。

#### 平台支持矩阵

| 平台 | 输入/输出 | 状态 |
| --- | --- | --- |
| Venus | Int8 -> Int8 | 支持 |
| ARCS | Int8 -> Int8 | 支持 |
| VenusA | Int8 -> Int8 | 支持 |

#### 参数、Shape 与内存约束

- 三平台均要求 NCHW、batch=1、非 ceil 模式。非全局池化 stride 各维为 1/2/4，kernel >= stride，pad < kernel。
- Venus 非全局 kernel <= 5、pad <= 4，且输入输出必须在 SM；2 的幂 kernel 面积要求 `scale_x == scale_o`。
- ARCS/VenusA 非全局 kernel <= 7、pad <= 11。ARCS 与 VenusA 根据通道和空间尺寸分块，workspace 含 Int32 sum；非 2 的幂除法还需额外缓冲。
- 平均值累加为 Int32；shift 范围由 kernel 面积和平台除法路径共同校验，最大右移 63。

#### 源码依据

- `executor/core/ops/avgpool2dint.c`；`executor/core/ops/{venus,arcs,venusA}/avgpool2dint.h`
- `tools/tpacker/graph_analysis/ops/Pool.py`

### BatchNorm1dInt

#### 功能说明

对 rank-3 序列执行逐通道仿射变换 `y = x * weight + bias`，是 BatchNorm2dInt 的一维版本。

#### 接口定义

- 输入：Int8 `x[N,C,L]`、Int8 `weight[C]`、Int32 `bias[C]`；输出同 shape Int8。
- 属性：`scale_x`、`scale_w`、`scale_o`。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | 支持 |
| ARCS | 支持 |
| VenusA | 支持 |

#### 参数、Shape 与内存约束

- weight/bias 元素数等于 C，bias 和乘加中间值为 Int32；`0 <= qx+qw-qo <= 63`。
- 输入输出必须在 SM；workspace 为 `ALIGN4(4*L)` 字节，VenusA 为 `ALIGN4(6*L)`。

#### 源码依据

- `executor/core/ops/batchnorm1dint.c`；`executor/core/ops/{venus,arcs,venusA}/batchnorm1dint.h`
- `tools/tpacker/graph_analysis/ops/BatchNorm1dInt.py`

### BatchNorm2dInt

#### 功能说明

执行逐通道仿射变换 `y = x * weight + bias`。历史名称 `BatchNormInt` 不作为独立算子，统一为 `BatchNorm2dInt`。

#### 接口定义

- 输入：Int8 `x[N,C,H,W]`、Int8 `weight[C]`、Int32 `bias[C]`。
- 输出：同 shape Int8；属性 `scale_x`、`scale_w`、`scale_o`。

#### 平台支持矩阵

| 平台 | 权重 / bias / 输出 | 状态 |
| --- | --- | --- |
| Venus | Int8 / Int32 / Int8 | 支持 |
| ARCS | Int8 / Int32 / Int8 | 支持 |
| VenusA | Int8 / Int32 / Int8 | 支持 |

#### 参数、Shape 与内存约束

- `weight`、`bias` 元素数必须等于 C；bias scale 固定为 `qx + qw`，乘加使用 Int32 中间值。
- `0 <= qx + qw - qo <= 63`。输入输出必须在 SM；workspace 为每个空间平面约 `4*H*W` 字节，VenusA 为 `6*H*W` 字节并做 4 字节对齐。

#### 源码依据

- `executor/core/ops/batchnorm2dint.c`；`executor/core/ops/{venus,arcs,venusA}/batchnorm2dint.h`
- `tools/tpacker/graph_analysis/ops/BatchNorm2dInt.py`

### BmmInt

#### 功能说明

量化 rank-2/rank-3 矩阵乘法，逻辑 shape 为 `[...,M,K] x [...,K,N] -> [...,M,N]`，乘积累加使用 Int32。

#### 接口定义

- 输入：两个同 dtype tensor；输出由 `o_bits` 指定。
- 重要属性：`scale_x`、`scale_y`、`scale_o`、`o_bits`。

#### 平台支持矩阵

| 平台 | 输入 | 输出 | 状态 |
| --- | --- | --- | --- |
| Venus | Int8 | Int8 | 支持 |
| ARCS | 同 dtype Int8 / Int32 | Int8 / Int32 | 支持 |
| VenusA | 同 dtype Int8 / Int16 / Int32 | Int8 / Int16 / Int32 | 支持 |

#### 参数、Shape 与内存约束

- 两输入 rank 相同且为 2 或 3；rank 3 的 batch 相同；K 必须一致。
- `0 <= qx + qy - qo <= 63`，不支持左移。Venus 按 M/N 分块并对 64 KiB/32 KiB 矩阵容量做检查；PSRAM 输入或输出需要 SM 暂存。ARCS/VenusA 的 PSRAM 输出需要 `M*N*输出字节数` workspace。

#### 源码依据

- `executor/core/ops/bmmint.c`；`executor/core/ops/{venus,arcs,venusA}/bmmint.h`
- `tools/tpacker/graph_analysis/ops/BmmInt.py`

### Cast

#### 功能说明

共享执行器逐元素转换 dtype，不做量化 scale 换算。

#### 接口定义

- 一个输入/同 shape 输出；属性 `to`。输入和目标可为 Int8/16/32/64、UInt8/16/32/64，且仅支持整数 dtype 之间转换。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | 支持 |
| ARCS | 支持 |
| VenusA | 支持 |

#### 参数、Shape 与内存约束

- 输入输出 shape 和 scale 相同但 dtype 按 `to` 改变；不能原地执行，元素数必须可装入 Int32。共享 CPU 循环要求输入输出地址可直接访问，不分配平台 workspace。

#### 源码依据

- `executor/core/ops/cast.c`、`executor/core/comm/type_switch.h`
- `tools/tpacker/graph_analysis/ops/Cast.py`

### Concat

#### 功能说明

沿指定轴拼接同 dtype tensor。量化导出中的 `iqCat` 可在拼接时把各输入重标定到输出 scale；标准 `Concat` 与 `iqCat` 共用 `iqCatAttrs` 和平台 backend。

#### 接口定义

- 输入：至少两个同 rank tensor；输出：拼接 tensor；属性 `axis`。

#### 平台支持矩阵

| 平台 | 输入/输出 dtype | 状态 |
| --- | --- | --- |
| Venus | Int8 / Int16 / Int32，同 dtype | 支持 |
| ARCS | Int8 / Int32，同 dtype | 支持 |
| VenusA | Int8 / Int16 / Int32，同 dtype | 支持 |

#### 参数、Shape 与内存约束

- 除拼接轴外所有维必须一致；输入和输出 dtype 必须一致。负 axis 会规范化。
- Venus Int8 的 `qo-qi` 为 `[-63,6]`，Int16/Int32 要求输入输出 scale 相同；ARCS Int8 的 `qi-qo` 为 `[-6,63]`，Int32 要求 scale 相同；VenusA 按 Int8/Int16/Int32 分别允许最多 6/14/30 位左移和 63 位右移。
- 后端按连续块拷贝或重标定；PSRAM 输入/输出和需要重标定的分段可能使用 SM workspace，具体容量由图分析分配。

#### 源码依据

- `executor/core/ops/concat.c`；`executor/core/ops/{venus,arcs,venusA}/concat.h`
- `tools/tpacker/graph_analysis/ops/Concat.py`、`tools/tpacker/graph_analysis/ops/iqCat.py`

### Conv1dInt

#### 功能说明

一维量化卷积。Int8 激活与存储为 Int8 的 Int4/Int8 权重相乘，Int32 累加，可选 Int32 bias。

#### 接口定义

- 输入：`x[N,C,W]`、`weight[M,C/group,K]`、可选 `bias[M]`；输出：`[N,M,Wo]`。
- 属性：kernel/stride/pad/dilation/group、三组 scale、`data_bits`、`parameter_bits`、`o_bits`、`quant_mode`。

#### 平台支持矩阵

| 平台 | 权重位宽 | 输出 | 状态 |
| --- | --- | --- | --- |
| Venus | 4 / 8 | Int8 / Int16 / Int32 | 支持 |
| ARCS | 4 / 8 | Int8 / Int32 | 支持 |
| VenusA | 4 / 8 | Int8 / Int16 / Int32 | 支持 |

#### 参数、Shape 与内存约束

- dilation 固定 1，stride 为 1/2/4，kernel >= stride。Venus kernel <= 5、pad <= 4；ARCS/VenusA kernel <= 12、pad <= 11，且 batch=1。
- `shift = qx + qw - qo`。Venus 要求 `[0,63]`；ARCS/VenusA 仅 Int32 输出可做最多 30 位左移。
- 权重在 pack 时按平台重排/压缩；bias 为 Int32，累加器为 Int32；PSRAM 和大矩阵分块所需 SM 由平台分析模块计算。

#### 源码依据

- `executor/core/ops/conv1dInt.c`；`executor/core/ops/{venus,arcs,venusA}/conv1dint.h`
- `tools/tpacker/graph_analysis/ops/Conv1dInt.py` 及其 `venus/`、`arcs/`、`venusA/` 平台模块

### Conv2dInt

#### 功能说明

NCHW 二维量化卷积，Int8 激活/权重、Int32 累加，可选 Int16/Int32 bias；离线阶段会把 bias 统一到 Int32 并重排权重。

#### 接口定义

- 输入：`x[1,C,H,W]`、`weight[M,C/group,kH,kW]`、可选 `bias[M]`；输出 `[1,M,Ho,Wo]`。
- 属性同 Conv1dInt，另含二维 dilation。

#### 平台支持矩阵

| 平台 | 权重位宽 | 输出 | 状态 |
| --- | --- | --- | --- |
| Venus | 8 | Int8 | 支持 |
| ARCS | 4 / 8 | Int8 / Int32 | 支持 |
| VenusA | 4 / 8 | Int8 | 支持 |

#### 参数、Shape 与内存约束

- batch=1，stride 各维 1/2/4，kernel >= stride，pad < kernel。Venus dilation=1、kernel <= 5、pad <= 4；ARCS/VenusA dilation 可为 1/2/4/8，无 dilation 时 kernel <= 12，有 dilation 时 <= 5，pad <= 11。
- 通道、group、weight kernel shape 必须一致；`0 <= qx + qw - qo <= 63`。
- 权重实际计算精度由 `parameter_bits` 决定而存储 dtype 仍为 Int8；bias 和卷积累加均为 Int32。平台分析器负责卷积 tile、PSRAM DMA 和 SM workspace。

#### 源码依据

- `executor/core/ops/conv2dInt.c`；`executor/core/ops/{venus,arcs,venusA}/conv2dint.h`
- `tools/tpacker/graph_analysis/ops/Conv2dInt.py` 及三平台子模块

### ConvTranspose2dInt

#### 功能说明

二维转置卷积。历史 `DeConv2dInt` 仅是后端文件/旧表名称，注册和模型名称统一为 `ConvTranspose2dInt`。

#### 接口定义

- 输入：Int8 `x[1,C,H,W]`、Int8 `weight[C,M/group,kH,kW]`、可选 Int16/Int32 bias；输出量化 tensor。
- 属性：kernel/stride/pad/output_padding/dilation/group、scale、位宽和 `quant_mode`。

#### 平台支持矩阵

| 平台 | 权重位宽 | 输出 | 状态 |
| --- | --- | --- | --- |
| Venus | 8 | Int8 / Int16 / Int32 | 支持 |
| ARCS | 4 / 8 | Int8 | 支持 |
| VenusA | 4 / 8 | Int8 | 支持 |

#### 参数、Shape 与内存约束

- batch=1、group=1、不支持 depthwise，dilation=1，stride 为 1/2/4，`0 <= output_padding < stride`；不支持 `auto_pad` 和显式 `output_shape`。
- Venus kernel <= 5、pad <= 4；ARCS/VenusA kernel <= 12、pad <= 11。VenusA 对 stride=2 仅允许 kernel 2/3/4/5，对 stride=4 仅允许 4/5。
- `0 <= qx + qw - qo <= 63`；Int32 累加，bias pack 为 Int32。workspace 与权重重排由平台分析模块计算。

#### 源码依据

- `executor/core/ops/convtranspose2dint.c`；`executor/core/ops/{venus,arcs,venusA}/deconv2dint.h`
- `tools/tpacker/graph_analysis/ops/ConvTranspose2dInt.py` 及三平台子模块

### Expand

#### 功能说明

按 ONNX broadcasting 规则把 size=1 的维复制到目标 shape，保持 dtype/scale。

#### 接口定义

- 输入 data 和常量 Int32/Int64 一维 shape；输出 broadcast 后 tensor。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | 支持 |
| ARCS | 支持 |
| VenusA | 支持 |

#### 参数、Shape 与内存约束

- 输入/输出 rank <= 7，目标维均为正，维度必须相等或一方为 1；仅支持按整字节寻址的 dtype，输出不能与输入原地复用。
- 输出必须在 SM 或 PSRAM。ARCS/VenusA 对 PSRAM 使用平台 copy API；Venus 编译使用共享 `memcpy` 路径，不需要额外 workspace。

#### 源码依据

- `executor/core/ops/expand.c`
- `tools/tpacker/graph_analysis/ops/Expand.py`

### FFNInt

#### 功能说明

ARCS 两层全连接融合 runtime wrapper，包含中间激活；不是当前正常 packer 可达算子。

#### 接口定义

- runtime 输入顺序：input、weight1、bias1、weight2、bias2；一个输出及可选 workspace。具体精度由 ARCS 融合头文件约定。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | - |
| ARCS | runtime-only |
| VenusA | - |

#### 参数、Shape 与内存约束

- 共享 wrapper 和 ARCS backend 存在，但 `graph_analysis/ops` 无 `FFNInt` 分析器，设备支持清单也未开放，因此不能按普通 ONNX 节点打包。

#### 实现注意事项

- 不应根据 wrapper 返回成功推断其他平台支持；非 ARCS 编译下没有实际 kernel 调用。

#### 源码依据

- `executor/core/ops/ffnint.c`、`executor/core/ops/arcs/ffnint.h`
- `executor/core/operator_attrs.h`

### Flatten

#### 功能说明

按 `axis` 将输入折叠为二维 tensor，不改变元素顺序、dtype 或 scale。

#### 接口定义

- 输入一个 tensor，输出 `[prod(dims[:axis]), prod(dims[axis:])]`；属性 `axis`。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | 支持 |
| ARCS | 支持 |
| VenusA | 支持 |

#### 参数、Shape 与内存约束

- axis 必须在合法 rank 范围。可原地/别名时不搬运；内存区域不同或不能别名时按平台使用 SM 拷贝。

#### 源码依据

- `executor/core/ops/flatten.c`；`executor/core/ops/{venus,arcs,venusA}/flatten.h`
- `tools/tpacker/graph_analysis/ops/Flatten.py`

### Gather

#### 功能说明

按 Int32/Int64 indices 从指定轴抽取切片，输出 scale 必须与 data 相同。

#### 接口定义

- 输入：`data`、`indices`；输出 shape 为 `data[:axis] + indices.shape + data[axis+1:]`；属性 `axis`、可选 `parameter_bits`。

#### 平台支持矩阵

| 平台 | data 精度 | 状态 |
| --- | --- | --- |
| Venus | 非压缩数据 | 支持 |
| ARCS | Int4 常量或普通数据 | 支持 |
| VenusA | 非压缩数据 | 支持 |

#### 参数、Shape 与内存约束

- indices 仅 Int32/Int64；常量索引允许 `-1`，其余值必须小于轴长度。
- 仅 ARCS 支持 `parameter_bits=4`，要求 data 为常量 Int8 存储且每个 gather slice 元素数为偶数；Venus/VenusA 不支持 packed Int4。

#### 源码依据

- `executor/core/ops/gather.c`；`executor/core/ops/{venus,arcs,venusA}/gather.h`
- `tools/tpacker/graph_analysis/ops/Gather.py`

### QGelu

#### 功能说明

量化 GELU。离线类和注册名均为 `QGelu`，文件名 `gelu.c`/`Gelu.py` 不表示注册名是 `GELU`。

#### 接口定义

- 输入：Int8/Int16/Int32；输出：Int8 同 shape。
- 属性：`scale_x`、`scale_o`、`x_bits`、`o_bits=8`、`quant_mode=FLOOR_ADD`。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | - |
| ARCS | - |
| VenusA | 支持 |

#### 参数、Shape 与内存约束

- 输入 scale 指数 `[-3,90]`，输出指数 `[-36,57]`，zero point 0；输入必须在 SM。Int8 输入 workspace 为 `ALIGN4(2*n)+4*n` 字节，Int16/Int32 为 `4*n`。

#### 源码依据

- `executor/core/ops/gelu.c`、`executor/core/ops/venusA/gelu.h`
- `tools/tpacker/graph_analysis/ops/Gelu.py`

### GluInt

#### 功能说明

沿轴等分输入为 `a,b` 并计算 `a * sigmoid(b)`。workbook/导出常写 `GLUInt`，但执行注册名是大小写敏感的 `GluInt`，二者不作为两个算子。

#### 接口定义

- 输入一个 Int8 tensor；输出将 split 轴减半；属性 `axis` 或 `dim`、`scale_x`、`scale_o`、`o_bits`。

#### 平台支持矩阵

| 平台 | 输出 | 状态 |
| --- | --- | --- |
| Venus | Int8 | 支持 |
| ARCS | Int8 | 支持 |
| VenusA | Int8 / Int16 / Int32 | 支持 |

#### 参数、Shape 与内存约束

- split 轴长度必须为正偶数。Venus/ARCS 仅最后轴；Venus 还要求前置元素乘积为 1。Venus/ARCS 输入输出必须在 SM；VenusA 允许 SM/PSRAM 并分配最多 64 KiB workspace。
- 各平台分别校验 sigmoid 输入和乘法输出 shift；不允许超过 Luna 63 位右移或标量 30 位左移。

#### 源码依据

- `executor/core/ops/gluInt.c`；`executor/core/ops/{venus,arcs,venusA}/gluint.h`
- `tools/tpacker/graph_analysis/ops/GLUInt.py`

### GRUInt

#### 功能说明

单层量化 GRU。可选初始 hidden 是同一注册算子的输入变体，不使用 `_Is8_Is64` 等别名。

#### 接口定义

- 输入：Int8 `x`、可选 Int8 hidden、Int8 `weight_ih[3H,D]`/`weight_hh[3H,H]`、两个 Int32 bias `[3H]`。
- 输出：Int8 sequence 和 Int8 hidden；属性含五组 scale、`batch_first`、`go_forward`、`input_size`、`hidden_size`。

#### 平台支持矩阵

| 平台 | 输入变体 | 状态 |
| --- | --- | --- |
| Venus | 可无/有 hidden，batch=1 | 支持 |
| ARCS | 必须有 hidden，time-major、batch=1 | 支持 |
| VenusA | 必须有 hidden，支持两种 layout | 支持 |

#### 参数、Shape 与内存约束

- 输入/权重 Int8，bias 与门累加为 Int32；输出 scale 必须等于 hidden scale。所有状态 zero point 为 0。
- Venus/ARCS/VenusA 的输入、输出和状态均要求 SM；常量参数可驻 SM 或由 DMA 搬运。workspace 以 `H*B` 的多个 Int32 门缓冲为主，VenusA batch-first 且 B>1 时另含转置缓冲。

#### 源码依据

- `executor/core/ops/gruint.c`；`executor/core/ops/{venus,arcs,venusA}/gruint.h`
- `tools/tpacker/graph_analysis/ops/GRUInt.py`

### iqAdd

#### 功能说明

对两个量化 tensor 做逐元素饱和加法，先按各自 scale 对齐到输出 scale；三平台均不支持普通 broadcasting。

#### 接口定义

- 输入：`x1`、`x2`；输出：`y`。两输入和输出的 shape/dtype 完全相同。
- 属性：`scale_x`、`scale_y`、`scale_o`、`quant_mode`；旧模型可用 `platform_quant=luna_quant`。

#### 平台支持矩阵

| 平台 | `x1` | `x2` | 输出 | 状态 |
| --- | --- | --- | --- | --- |
| Venus | Int8 | Int8 | Int8 | 支持 |
| ARCS | Int8 | Int8 | Int8 | 支持 |
| VenusA | Int8 / Int16 / Int32 | 同 dtype | 同 dtype | 支持 |

#### 参数、Shape 与内存约束

- 两输入和输出要求 shape/dtype 相同、zero point 0。`shift1=qx1-qo`、`shift2=qx2-qo`。
- Venus 两个 shift 均为 `[0,63]`；ARCS Int8 为 `[-6,63]`；VenusA Int8/Int16/Int32 分别为 `[-6,63]`、`[-14,63]`、`[-30,63]`。
- tensor 可在 SM/PSRAM；PSRAM 搬运、scale 转换或外部输出需要 SM workspace。ARCS/VenusA workspace 地址按 4 字节对齐。

#### 实现注意事项

- 负 shift 表示有限左移，正 shift 表示右移；最后按输出 dtype 饱和。

#### 源码依据

- `executor/core/ops/iqadd.c`；`executor/core/ops/{venus,arcs,venusA}/iqadd.h`
- `tools/tpacker/graph_analysis/ops/iqAdd.py`

### iqDiv

#### 功能说明

两个同 shape 量化 tensor 的逐元素除法并重标定。

#### 接口定义

- 输入 `x/y`，输出同 shape；属性 `scale_x`、`scale_y`、`scale_o`。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | 支持 |
| ARCS | 支持 |
| VenusA | 支持 |

#### 参数、Shape 与内存约束

- 输入输出必须在 SM。右输入可为标量或同 shape：vector 路径仅 Int32；标量路径要求同 dtype，Venus 支持 Int8/16/32，ARCS 支持 Int8/32，VenusA 仅 Int32。
- 除数不得为 0；常量标量除数还必须是正的 2 的幂。`qo-(qx-qy)` 在 `[0,63]`，标量折算后的左移受 dtype 的 6/14/30 位限制。

#### 源码依据

- `executor/core/ops/iqdiv.c`；`executor/core/ops/{venus,arcs,venusA}/iqdiv.h`
- `tools/tpacker/graph_analysis/ops/iqDiv.py`

### iqMul

#### 功能说明

两个量化 tensor、整数 tensor 与同 dtype 整数标量，或 NCHW 与 NC11 tensor 相乘；乘积以扩展精度计算后重标定并饱和。

#### 接口定义

- 输入 `x/y`，输出同 shape；属性 `scale_x`、`scale_y`、`scale_o`。

#### 平台支持矩阵

| 平台 | 有效精度 | 状态 |
| --- | --- | --- |
| Venus | 同 dtype Int8 / Int16 / Int32 | 支持 |
| ARCS | 同 dtype Int8 / Int32 | 支持 |
| VenusA | Int8 / Int16 / Int32 | 支持 |

#### 参数、Shape 与内存约束

- 三平台支持同 shape 或与左输入同 dtype 的右输入整数标量。还支持 NCHW x NC11：Venus 要求 N=1，ARCS 支持 N=1 或等于左输入且 `H*W<=16384`，VenusA 仅该 broadcast 路径限制为 Int8。
- `shift=qx+qy-qo` 必须在 `[0,63]`。Venus/ARCS 输出 dtype 等于输入；VenusA 的 Int16 输入可输出 Int8/Int16，Int8 和 Int32 分别只能输出同精度。PSRAM 路径由 SM workspace 分块搬运。

#### 源码依据

- `executor/core/ops/iqmul.c`；`executor/core/ops/{venus,arcs,venusA}/iqmul.h`
- `tools/tpacker/graph_analysis/ops/iqMul.py`

### iqPad

#### 功能说明

对尾部维度做 constant、replicate 或 reflect padding。

#### 接口定义

- 输入：Int8 data、常量 Int64 一维 pads、常量标量 fill；输出同 dtype/scale；属性 `mode`。

#### 平台支持矩阵

| 平台 | rank | 状态 |
| --- | --- | --- |
| Venus | 4，batch=1 | 支持 |
| ARCS | 3/4，4D batch=1 | 支持 |
| VenusA | 1-4，4D batch=1 | 支持 |

#### 参数、Shape 与内存约束

- pads 长度为 2/4/6/8，值非负；4D kernel 路径不能 pad batch/channel。reflect pad 必须小于原维度。
- VenusA fill 仅 0；其他平台 replicate/reflect 的未用 fill 也必须为 0。Venus 输入输出在 SM；ARCS kernel 不需额外 workspace；VenusA 可输出 PSRAM并按需暂存。

#### 源码依据

- `executor/core/ops/iqpad.c`；`executor/core/ops/{venus,arcs,venusA}/iqpad.h`
- `tools/tpacker/graph_analysis/ops/iqPad.py`

### iqSigmoid

#### 功能说明

定点查表/整数 Sigmoid，保持 shape 并按 `scale_o` 输出。

#### 接口定义

- 输入一个量化 tensor，输出同 shape；属性 `scale_x`、`scale_o`。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | 支持 |
| ARCS | 支持 |
| VenusA | 支持 |

#### 参数、Shape 与内存约束

- 输入 zero point 0；输入/输出位宽和合法 shift 由离线分析与平台查表格式共同限制。SM 可直接计算，PSRAM 路径需要分块 workspace。

#### 源码依据

- `executor/core/ops/iqsigmoid.c`；`executor/core/ops/{venus,arcs,venusA}/iqsigmoid.h`
- `tools/tpacker/graph_analysis/ops/iqSigmoid.py`

### iqSub

#### 功能说明

同 shape 量化 tensor 逐元素减法，按输出 scale 对齐并饱和。

#### 接口定义

- 两输入一输出；属性 `scale_x`、`scale_y`、`scale_o`。

#### 平台支持矩阵

| 平台 | 有效精度 | 状态 |
| --- | --- | --- |
| Venus | Int8 | 支持 |
| ARCS | Int8 | 支持 |
| VenusA | Int8 | 支持 |

#### 参数、Shape 与内存约束

- 不支持 broadcasting；shape/dtype 相同。Venus 与 VenusA 当前有效路径为 Int8 且要求两个输入指数不小于输出指数，差值在 `[0,63]`；ARCS 有效路径为 Int8。PSRAM 或 scale 不同需要 SM workspace。

#### 源码依据

- `executor/core/ops/iqsub.c`；`executor/core/ops/{venus,arcs,venusA}/iqsub.h`
- `tools/tpacker/graph_analysis/ops/iqSub.py`

### iqSum

#### 功能说明

沿最后一维求和并将该维保留为 1。

#### 接口定义

- 输入一个 tensor，输出同 dtype；属性 `dims`、`scale_x`、`scale_o`。

#### 平台支持矩阵

| 平台 | 输入/输出 | 状态 |
| --- | --- | --- |
| Venus | Int8 | 支持 |
| ARCS | Int8 | 支持 |
| VenusA | 同 dtype Int8 / Int16 / Int32 | 支持 |

#### 参数、Shape 与内存约束

- 仅最后轴且长度 > 0；`0 <= qx-qo <= 63`；输入输出必须在 SM。Venus/ARCS 使用 Int32 sum workspace，VenusA backend 可直接完成而无需离线 workspace。

#### 源码依据

- `executor/core/ops/iqsum.c`；`executor/core/ops/{venus,arcs,venusA}/iqsum.h`
- `tools/tpacker/graph_analysis/ops/iqSum.py`

### iqTanh

#### 功能说明

定点 Tanh 激活，保持 shape 并转换到输出 scale。

#### 接口定义

- 一个量化输入和一个输出；属性 `scale_x`、`scale_o`。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | 支持 |
| ARCS | 支持 |
| VenusA | 支持 |

#### 参数、Shape 与内存约束

- zero point 0；查表输入范围、输出位宽和 shift 由离线分析器校验。PSRAM 数据按平台分块到 SM。

#### 源码依据

- `executor/core/ops/iqtanh.c`；`executor/core/ops/{venus,arcs,venusA}/iqtanh.h`
- `tools/tpacker/graph_analysis/ops/iqTanh.py`

### iqVar

#### 功能说明

沿最后或倒数第二维计算定点方差，输出保持被归约维为 1。

#### 接口定义

- 一个 Int8 输入和一个 Int8 输出；属性 `dims`、`scale_x`、`scale_o`。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | 支持 |
| ARCS | 支持 |
| VenusA | 支持 |

#### 参数、Shape 与内存约束

- rank >= 3；ARCS/VenusA 只接受 rank 3，Venus 更高 rank 的前导维必须全为 1。输入输出必须在 SM。
- `shift=2*qx-qo`：Venus `[0,30]` 且归约长度 <= 23726566；ARCS/VenusA `[-30,30]` 且长度 <= 131071。平方/求和使用扩展整数累加，workspace 依轴和平台分配。

#### 源码依据

- `executor/core/ops/iqvar.c`；`executor/core/ops/{venus,arcs,venusA}/iqvar.h`
- `tools/tpacker/graph_analysis/ops/iqVar.py`

### LayerNormInt

#### 功能说明

对尾部一维或二维做量化 LayerNorm，随后应用 weight 和 Int32 bias；统计与仿射中间值使用 Int32。

#### 接口定义

- 输入：Int8 data、weight、Int32 bias；输出 Int8 同 shape。
- 属性：`scale_x`、`scale_w`、`scale_o`、`parameter_bits`；runtime 固定 `eps=1e-5`、keepdims。

#### 平台支持矩阵

| 平台 | weight | 状态 |
| --- | --- | --- |
| Venus | Int8 / Int16 | 支持 |
| ARCS | Int8 | 支持 |
| VenusA | Int8 | 支持 |

#### 参数、Shape 与内存约束

- weight/bias 大小为最后一维或最后两维乘积。ARCS rank >= 2、宽度 <= 32767；Venus/VenusA rank >= 3，前导多余维乘积须为 1，Venus 宽度 <= 133144。
- `0 <= qx <= 15`，`0 <= 15+qw-qo <= 63`。VenusA 输入输出必须在 SM；三平台 workspace 都包含统计、weight/bias 和中间 Int32 缓冲。

#### 源码依据

- `executor/core/ops/layernormint.c`；`executor/core/ops/{venus,arcs,venusA}/layernormint.h`
- `tools/tpacker/graph_analysis/ops/LayerNormInt.py`

### LinearInt

#### 功能说明

量化全连接，runtime 仅支持 `transB=1`。矩阵乘积采用扩展整数累加，可选 bias 在累加精度加入。

#### 接口定义

- 输入：rank 1-4 data、rank-2 weight `[N,K]`、可选 bias `[N]`；输出把末维 K 替换为 N。
- 属性：三组 scale、`data_bits`、`parameter_bits`、`o_bits`、`quant_mode`、`transB=1`。

#### 平台支持矩阵

| 平台 | 主要有效组合 | 状态 |
| --- | --- | --- |
| Venus | Int8 x Int8(8-bit) -> Int8/16/32 | 支持 |
| ARCS | Int8 x Int8(4/8-bit) -> Int8/32 | 支持 |
| VenusA | 代码列举的 Int8/16/32 组合 -> Int8/16/32 | 支持 |

#### 参数、Shape 与内存约束

- Venus rank <= 3；ARCS/VenusA rank <= 4。Int4 weight 要求 K 为偶数。ARCS/VenusA bias 必须 Int32；Venus bias 可 Int8/16/32，元素数均为 N。
- `shift=qx+qw-qo`：Venus `[0,63]`；ARCS Int8 输出不可左移、Int32 最多左移 30；VenusA Int8/16/32 最小 shift 分别 0/-14/-30。
- VenusA 的有效 dtype/位宽组合以分析器显式集合为准，不能做笛卡尔积推断。权重 pack、PSRAM DMA、矩阵转置与输出暂存均计入 SM workspace。

#### 源码依据

- `executor/core/ops/linearint.c`；`executor/core/ops/{venus,arcs,venusA}/linearint.h`
- `tools/tpacker/graph_analysis/ops/LinearInt.py`

### LogSoftmaxInt

#### 功能说明

最后一维量化 LogSoftmax，Int8 输入/输出。

#### 接口定义

- 一个输入/输出；属性 `axis`（Venus 兼容 `dim`）、`scale_x`、`scale_o`。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | 支持 |
| ARCS | 支持 |
| VenusA | 支持 |

#### 参数、Shape 与内存约束

- 非标量，仅最后轴，轴长 1..2048。Venus/ARCS 输入转换到 Q25，VenusA 输出路径使用 Q15；各 shift 受 63 位右移/30 位左移限制。
- Venus 和 VenusA 输入输出必须在 SM；ARCS workspace 为 stride 的 8 倍，Venus/VenusA 的工作区上限按 64 KiB 分块。

#### 源码依据

- `executor/core/ops/logsoftmaxint.c`；`executor/core/ops/{venus,arcs,venusA}/logsoftmaxint.h`
- `tools/tpacker/graph_analysis/ops/LogSoftmaxInt.py`

### LSTMInt

#### 功能说明

单层、单方向量化 LSTM。可选 sequence-length、hidden、cell 输入由输入数量区分，不作为 `_Is8_Is64...` 独立别名。

#### 接口定义

- 输入：Int8 `x`，可选状态/长度输入，Int8 `weight_ih[4H,D]`、`weight_hh[4H,H]`，两个 Int32 bias `[4H]`。
- 输出：Int8 sequence、Int8 hidden、Int32 Q15 cell；属性含 `batch_first`、`go_forward`、D/H 和各 scale。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | 支持 |
| ARCS | 支持 |
| VenusA | 支持 |

#### 参数、Shape 与内存约束

- rank-3、batch=1，支持 time-major/batch-first 和单个正向或反向；输入/权重 Int8，bias/门累加 Int32，cell 固定 Int32 Q15，输出 scale 等于 hidden scale。
- 三平台输入、输出和状态要求 SM。Venus 校验两个权重矩阵存在 <=32 KiB 的有效分块；ARCS/VenusA workspace 还包含两组权重和 bias 的搬运空间。

#### 源码依据

- `executor/core/ops/lstmint.c`；`executor/core/ops/{venus,arcs,venusA}/lstmint.h`
- `tools/tpacker/graph_analysis/ops/LstmInt.py`

### MaxPool

#### 功能说明

NCHW Int8 二维最大池化，不改变 scale。

#### 接口定义

- `x[N,C,H,W] -> y[N,C,Ho,Wo]`；属性 kernel、stride、pads、`ceil_mode=0`。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | 支持 |
| ARCS | 支持 |
| VenusA | 支持 |

#### 参数、Shape 与内存约束

- 仅 NCHW Int8、非 ceil；stride 为 1/2/4，kernel >= stride，pad < kernel。Venus 非全局 kernel <=5/pad<=4；ARCS/VenusA <=7/pad<=11。
- ARCS 输出必须在 SM；Venus 大输入只支持 H 分块；VenusA 单次输入容量按 32 KiB 检查，PSRAM 输出按通道分块并使用最多 64 KiB SM 暂存。

#### 源码依据

- `executor/core/ops/maxpool.c`；`executor/core/ops/{venus,arcs,venusA}/maxpool.h`
- `tools/tpacker/graph_analysis/ops/Pool.py`

### MultiHeadAttentionInt

#### 功能说明

ARCS 多头注意力融合 runtime。共享注册名实际为 `MultiheadAttention`，源码 profile 文本写 `MultiheadAttentionInt`；本文用功能名 `MultiHeadAttentionInt`，不把这些大小写形式拆成多个算子。

#### 接口定义

- runtime 接收 input，Q/K/V/投影四组 weight+bias，key/value embedding，一个输出和必需 workspace。精度由 ARCS 融合 kernel 定义。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | - |
| ARCS | runtime-only |
| VenusA | - |

#### 参数、Shape 与内存约束

- 当前无正常 graph-analysis 类和设备清单入口，普通 tpacker 模型不可达；wrapper 强制 tensor 数等于输入+输出+1，即必须提供 workspace。

#### 实现注意事项

- 这是预构造资源的运行时接口，不应写成正常 packer 支持。

#### 源码依据

- `executor/core/ops/multiheadattentionint.c`、`executor/core/ops/arcs/multiheadattentionint.h`
- `executor/core/operator_attrs.h`、`executor/core/operator_list.h`

### PReLU

#### 功能说明

使用整数 slope 和 post-shift 的参数化 ReLU。

#### 接口定义

- 一个整数输入/同 dtype 输出；属性 `slope`、`post_shift`。

#### 平台支持矩阵

| 平台 | dtype | 状态 |
| --- | --- | --- |
| Venus | Int8 / Int16 / Int32 | 支持 |
| ARCS | Int8 / Int32 | 支持 |
| VenusA | Int8 / Int16 / Int32 | 支持 |

#### 参数、Shape 与内存约束

- `slope`、`post_shift` 均在 `[0,63]` 且和不超过 63；输入输出 shape/dtype 相同并必须位于 SM。

#### 源码依据

- `executor/core/ops/prelu.c`；`executor/core/ops/{venus,arcs,venusA}/prelu.h`
- `tools/tpacker/graph_analysis/ops/activations.py`

### ReLU

#### 功能说明

计算 `max(0,x)`，可同时按输出 scale/位宽重标定。

#### 接口定义

- 一个整数输入/同 shape 输出；可选 `scale_x`、`scale_o`、`o_bits`。

#### 平台支持矩阵

| 平台 | 输入/输出精度 | 状态 |
| --- | --- | --- |
| Venus | Int8 / Int16 / Int32 | 支持 |
| ARCS | Int8 / Int32 | 支持 |
| VenusA | Int8 / Int16 / Int32 | 支持 |

#### 参数、Shape 与内存约束

- `0 <= qo-qx <= 63`。任一端在 PSRAM 时仅允许 Int8 -> Int8，并使用最多 64 KiB SM 分块 workspace。

#### 源码依据

- `executor/core/ops/relu.c`；`executor/core/ops/{venus,arcs,venusA}/relu.h`
- `tools/tpacker/graph_analysis/ops/activations.py`

### ReLUx

#### 功能说明

将 ReLU 上界限制为整数 threshold；注册名为 `Relux`，本文显示名 `ReLUx`。

#### 接口定义

- 一个输入、一个 Int8 输出；属性 `threshold`、`shift`。

#### 平台支持矩阵

| 平台 | 输入 | 状态 |
| --- | --- | --- |
| Venus | - | - |
| ARCS | Int8 / Int32 | 支持 |
| VenusA | Int8 | 支持 |

#### 参数、Shape 与内存约束

- threshold 必须装入 Int8，shift 在 `[0,63]`；输出 scale 为 `qx+shift`，输入输出必须在 SM。

#### 源码依据

- `executor/core/ops/relux.c`；`executor/core/ops/{arcs,venusA}/relux.h`
- `tools/tpacker/graph_analysis/ops/activations.py`

### Requant

#### 功能说明

在整数 dtype/scale 之间做重标定和饱和转换。

#### 接口定义

- 一个输入/同 shape 输出；属性指定输入、输出 scale 和输出位宽。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | 支持 |
| ARCS | 支持 |
| VenusA | 支持 |

#### 参数、Shape 与内存约束

- zero point 0，shift 必须落在对应输入/输出精度的 kernel 范围；PSRAM 数据需要 SM 分块 workspace。

#### 源码依据

- `executor/core/ops/requant.c`；`executor/core/ops/{venus,arcs,venusA}/requant.h`
- `tools/tpacker/graph_analysis/ops/Requant.py`

### Reshape

#### 功能说明

按常量 shape 重解释 tensor，不改变元素、dtype 或 scale。

#### 接口定义

- 输入 data 与 shape，输出目标 shape；遵循 ONNX `0`/`-1` 维规则的离线检查。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | 支持 |
| ARCS | 支持 |
| VenusA | 支持 |

#### 参数、Shape 与内存约束

- 输入输出元素数相同；shape 必须可在打包阶段解析。通常可别名，不可别名或内存区不同才发生拷贝。

#### 源码依据

- `executor/core/ops/reshape.c`；`executor/core/ops/{venus,arcs,venusA}/reshape.h`
- `tools/tpacker/graph_analysis/ops/Reshape.py`

### Resize

#### 功能说明

通用 C 实现的 ONNX Resize，当前离线入口仅开放 Venus。

#### 接口定义

- 输入 data、roi、scales、sizes，输出按 scales 或 sizes 推导 shape；属性含 coordinate transformation、插值 mode 和 nearest mode。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | 支持 |
| ARCS | - |
| VenusA | - |

#### 参数、Shape 与内存约束

- 离线分析器明确要求 `platform=venus`。支持 nearest/linear/cubic 及对应坐标变换枚举的范围以共享 `resize.c` 分支为准；输入参数必须能在打包时推导出确定输出 shape。

#### 源码依据

- `executor/core/ops/resize.c`
- `tools/tpacker/graph_analysis/ops/Resize.py`

### Shape

#### 功能说明

在运行时返回输入各维长度。

#### 接口定义

- 一个任意 dtype 输入；输出为长度等于 rank 的 Int64 一维 tensor。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | 支持 |
| ARCS | 支持 |
| VenusA | 支持 |

#### 参数、Shape 与内存约束

- 输出第 0 维必须等于输入 rank；共享 wrapper 直接写入 Int64 维度值，无平台 workspace。

#### 源码依据

- `executor/core/ops/shape.c`
- `tools/tpacker/graph_analysis/ops/Shape.py`

### ShuffleChannel

#### 功能说明

按 group 重排 channel。当前存在 Venus wrapper/backend 和一个未导入的分析器文件，但没有进入执行注册清单，因此没有正常有效支持。

#### 接口定义

- 一个输入/输出；属性 `group`。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | - |
| ARCS | - |
| VenusA | - |

#### 参数、Shape 与内存约束

- Venus 源实现要求 channel 可按 group 整分；但在 `operator_list.h` 和 `graph_analysis/ops/__init__.py` 补齐前，不应生成该节点。

#### 实现注意事项

- 本节的 `-` 是有效交集结论，不否认 `shufflechannel.c`/Venus header 存在。

#### 源码依据

- `executor/core/ops/shufflechannel.c`、`executor/core/ops/venus/shufflechannel.h`
- `tools/tpacker/graph_analysis/ops/ShuffleChannel.py`、`executor/core/operator_list.h`

### Slice

#### 功能说明

按 starts/ends/axes/steps 提取切片，保持 dtype/scale。

#### 接口定义

- 输入 data 及常量 Slice 参数 tensor；输出推导后的 shape。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | 支持 |
| ARCS | 支持 |
| VenusA | 支持 |

#### 参数、Shape 与内存约束

- 参数必须可在打包阶段解析；支持的负索引、step 和连续块形式以离线分析器为准。非连续/PSRAM 路径使用平台 copy kernel 和 SM workspace。

#### 源码依据

- `executor/core/ops/slice.c`；`executor/core/ops/{venus,arcs,venusA}/slice.h`
- `tools/tpacker/graph_analysis/ops/Slice.py`

### SoftmaxInt

#### 功能说明

最后一维定点 Softmax。

#### 接口定义

- 一个量化输入/输出；属性 `axis` 或 `dim`、`scale_x`、`scale_o`、`o_bits`。

#### 平台支持矩阵

| 平台 | 输入 | 输出 | 状态 |
| --- | --- | --- | --- |
| Venus | Int8 | Int8 | 支持 |
| ARCS | Int8 | Int8 / Int32 | 支持 |
| VenusA | Int8 / Int16 / Int32 | Int8 / Int16 / Int32 | 支持 |

#### 参数、Shape 与内存约束

- 仅最后轴，长度 1..2048，zero point 0。Venus 输入变换 `25-qx` 为 `[0,30]`、输出 `15-qo` 为 `[0,63]`；ARCS Int32 输出要求 `qo=15`；VenusA 允许有限左移。
- workspace 含指数/求和的 Int32 缓冲；VenusA stride <= 2048 并将 workspace 限于分块后的 64 KiB。

#### 源码依据

- `executor/core/ops/softmaxint.c`；`executor/core/ops/{venus,arcs,venusA}/softmaxint.h`
- `tools/tpacker/graph_analysis/ops/SoftmaxInt.py`

### SparifyFFNInt

#### 功能说明

ARCS 三权重/三 bias 的稀疏 FFN 融合 runtime，不是正常 packer 可达算子。

#### 接口定义

- runtime 输入 input、weight1/2/3、bias1/2/3；一个输出及可选 workspace。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | - |
| ARCS | runtime-only |
| VenusA | - |

#### 参数、Shape 与内存约束

- 缺少 graph-analysis 类及正常设备入口；精度、稀疏布局和 workspace 必须由预构造资源满足 ARCS header 约定。

#### 实现注意事项

- 不应把 wrapper/backend 的存在写成普通 ONNX packer 支持。

#### 源码依据

- `executor/core/ops/sparifyffnint.c`、`executor/core/ops/arcs/sparifyffnint.h`
- `executor/core/operator_attrs.h`

### Split

#### 功能说明

沿指定轴把 Int8/Int16/Int32 tensor 分成多个输出，不改变 dtype/scale。

#### 接口定义

- 一个 Int8/Int16/Int32 data 输入，多个同 dtype 输出；属性/常量输入指定 `axis` 和各段长度。

#### 平台支持矩阵

| 平台 | 输入/输出 dtype | 状态 |
| --- | --- | --- |
| Venus | Int8 / Int16 / Int32 | 支持 |
| ARCS | Int8 / Int16 / Int32 | 支持 |
| VenusA | Int8 / Int16 / Int32 | 支持 |

#### 参数、Shape 与内存约束

- split 长度之和等于轴长度，输出非 split 维与输入一致。连续 SM 段可直接复制；PSRAM 输出按段使用 SM/DMA。

#### 源码依据

- `executor/core/ops/split.c`；`executor/core/ops/{venus,arcs,venusA}/split.h`
- `tools/tpacker/graph_analysis/ops/Split.py`

### Squeeze

#### 功能说明

删除长度为 1 的维度，只改变 shape 元数据。

#### 接口定义

- 输入 data 和可选常量 Int64 axes；也可用 `axes` 属性，支持 `noop_with_empty_axes`。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | 支持 |
| ARCS | 支持 |
| VenusA | 支持 |

#### 参数、Shape 与内存约束

- 指定 axis 必须唯一、合法且原长度为 1；空 axes 可删除全部 singleton 维或按属性保持不变。runtime 强制输入输出数据指针相同且内存类型相同，无 workspace。

#### 源码依据

- `executor/core/ops/squeeze.c`
- `tools/tpacker/graph_analysis/ops/Squeeze.py`

### QSwish

#### 功能说明

量化 Swish/SiLU，计算 `x * sigmoid(x)`；注册名为 `QSwish`。

#### 接口定义

- Int8/Int16/Int32 输入，Int8 同 shape 输出；属性 `scale_x`、`scale_o`、`data_bits`/`x_bits`、`o_bits=8`、`quant_mode=FLOOR_ADD`。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | - |
| ARCS | - |
| VenusA | 支持 |

#### 参数、Shape 与内存约束

- 输入 scale 指数 `[-3,90]`，输出指数 `[-36,57]`；输入输出均在 SM。Int8 workspace 为 `ALIGN4(2*n)+4*n`，Int16/Int32 为 `4*n`。

#### 源码依据

- `executor/core/ops/swish.c`、`executor/core/ops/venusA/swish.h`
- `tools/tpacker/graph_analysis/ops/Swish.py`

### Tile

#### 功能说明

按常量 repeats 在各轴重复 Int8 tensor，保持 dtype/scale。

#### 接口定义

- 输入 Int8 data 和 Int64 repeats；输出为 Int8，各维为 `shape[i]*repeats[i]`。

#### 平台支持矩阵

| 平台 | 输入/输出 dtype | 状态 |
| --- | --- | --- |
| Venus | Int8 | 支持 |
| ARCS | Int8 | 支持 |
| VenusA | Int8 | 支持 |

#### 参数、Shape 与内存约束

- repeats 必须在打包阶段已知、长度等于 rank 且为非负整数。输出在 PSRAM 或重复块不能直接写入时使用 SM workspace。

#### 源码依据

- `executor/core/ops/tile.c`；`executor/core/ops/{venus,arcs,venusA}/tile.h`
- `tools/tpacker/graph_analysis/ops/Tile.py`

### topN

#### 功能说明

查找最后一维最大元素并输出值/索引对。历史 workbook 的 `TopN` 规范为大小写敏感的注册名 `topN`。

#### 接口定义

- 输入：Int8 data 和 Int64 index offset；属性 `dim`、`max_num`；输出首维为 2。

#### 平台支持矩阵

| 平台 | 输出存储 | 状态 |
| --- | --- | --- |
| Venus | Int16 值/索引对 | 支持 |
| ARCS | Int32 值/索引对 | 支持 |
| VenusA | Int32 值/索引对 | 支持 |

#### 参数、Shape 与内存约束

- 仅最后轴、`max_num=1`、`shape[0]=1`；offset 至少一个 Int64 元素。workspace：Venus 8 字节，ARCS/VenusA 16 字节。

#### 源码依据

- `executor/core/ops/topn.c`；`executor/core/ops/{venus,arcs,venusA}/topn.h`
- `tools/tpacker/graph_analysis/ops/topN.py`

### topN2

#### 功能说明

对已经编码为值/索引对的中间 tensor 再做一次 top-1 合并，通常用于离线拆分后的归并。

#### 接口定义

- 一个 `[2,rows,width]` 输入；属性 `dim`、`max_num=1`、`scale_x`、`scale_o`；输出 `[2,rows,1]`。

#### 平台支持矩阵

| 平台 | 输入/输出 | 状态 |
| --- | --- | --- |
| Venus | Int16 | 支持 |
| ARCS | Int32 | 支持 |
| VenusA | Int32 | 支持 |

#### 参数、Shape 与内存约束

- 仅 rank 3、首维 2、最后轴、输入输出 scale 完全相同。Venus workspace 8 字节，ARCS/VenusA 16 字节；三平台均要求值/索引输入输出和 workspace 位于 SM。

#### 源码依据

- `executor/core/ops/topn2.c`；`executor/core/ops/{venus,arcs,venusA}/topn2.h`
- `tools/tpacker/graph_analysis/ops/topN2.py`

### Transpose

#### 功能说明

按 `perm` 重排 Int8/Int16/Int32 tensor 的维度和数据，保持 dtype/scale。

#### 接口定义

- 一个 Int8/Int16/Int32 输入和同 dtype 输出；属性 `perm` 是 rank 的排列。

#### 平台支持矩阵

| 平台 | 输入/输出 dtype | 状态 |
| --- | --- | --- |
| Venus | Int8 / Int16 / Int32 | 支持 |
| ARCS | Int8 / Int32 | 支持 |
| VenusA | Int8 / Int16 / Int32 | 支持 |

#### 参数、Shape 与内存约束

- perm 必须覆盖每个轴一次；输出 shape 按 perm 重排。矩阵转置受平台对齐和单 tile 容量限制，大 tensor 由离线分析分块并使用 SM workspace。

#### 源码依据

- `executor/core/ops/transpose.c`；`executor/core/ops/{venus,arcs,venusA}/transpose.h`
- `tools/tpacker/graph_analysis/ops/Transpose.py`

### Unsqueeze

#### 功能说明

在指定位置插入长度为 1 的维度，只改变 shape 元数据。

#### 接口定义

- 输入 data 和可选常量 Int64 axes；也可用 `axes` 属性。

#### 平台支持矩阵

| 平台 | 状态 |
| --- | --- |
| Venus | 支持 |
| ARCS | 支持 |
| VenusA | 支持 |

#### 参数、Shape 与内存约束

- 输出 rank <= 7；axes 必须唯一且在输出 rank 范围内。runtime 强制输入输出数据指针相同且内存类型相同，不需要 workspace。

#### 源码依据

- `executor/core/ops/unsqueeze.c`
- `tools/tpacker/graph_analysis/ops/Unsqueeze.py`

### Quant

#### 功能说明

将模型边界的 Float32 tensor 量化为 Int8 tensor，用于连接浮点输入与后续定点算子。这是允许使用 Float32 tensor 的边界转换之一，不表示中间网络支持浮点计算。

#### 接口定义

- 输入：Float32 tensor。
- 输出：同 shape Int8 tensor。
- 属性：`data_bits=8`、目标 scale、zero point 和量化舍入模式。

#### 平台支持矩阵

| 平台 | 输入 | 输出 | 状态 |
| --- | --- | --- | --- |
| Venus | Float32 | Int8 | 支持 |
| ARCS | Float32 | Int8 | 支持 |
| VenusA | Float32 | Int8 | 支持 |

#### 参数、Shape 与内存约束

- 输入输出 shape 相同，zero point 为 0，scale 必须为正的 2 的幂。
- wrapper 虽可解析 8/16/32 位属性，当前有效执行路径的输出指针和离线分析均限定为 Int8，因此不声明 Int16/Int32 输出支持。
- 输入输出不能原地复用，数据地址必须能由共享执行器直接访问。

#### 实现注意事项

- `Quant` 由共享 C wrapper 执行，没有 `venus/quant.h`、`arcs/quant.h` 或 `venusA/quant.h` 平台 backend。

#### 源码依据

- `executor/core/ops/quant.c`、`executor/core/comm/utils.c`
- `tools/tpacker/graph_analysis/ops/Quant.py`

### Dequant

#### 功能说明

将定点 tensor 反量化为模型边界的 Float32 tensor，用于连接定点网络输出与浮点消费者。这是允许使用 Float32 tensor 的另一边界转换。

#### 接口定义

- 输入：Int8、UInt8 或 Int32 tensor。
- 输出：同 shape Float32 tensor。
- 属性：输入 scale；zero point 必须为 0。

#### 平台支持矩阵

| 平台 | 输入 | 输出 | 状态 |
| --- | --- | --- | --- |
| Venus | Int8 / UInt8 / Int32 | Float32 | 支持 |
| ARCS | Int8 / UInt8 / Int32 | Float32 | 支持 |
| VenusA | Int8 / UInt8 / Int32 | Float32 | 支持 |

#### 参数、Shape 与内存约束

- 输入输出 shape 相同，输入 scale 指数必须为整数且位于 `[0,30]`，zero point 为 0。
- 离线分析会校验属性 scale 与输入 tensor scale 一致；输入输出不能原地复用。

#### 实现注意事项

- `Dequant` 由共享 C wrapper 执行，没有平台专用 backend；其 Float32 输出应位于模型边界，而不是继续进入普通计算算子。

#### 源码依据

- `executor/core/ops/dequant.c`、`executor/core/comm/utils.c`
- `tools/tpacker/graph_analysis/ops/Dequant.py`

### Clip

#### 功能说明

共享执行器逐元素截断到 `[min,max]`；既支持属性边界，也支持 ONNX 三输入标量边界形式。

#### 接口定义

- 输入形式一：data，边界来自 `min`/`max` 属性；形式二：data、同 dtype 标量 min、同 dtype 标量 max。输出同 shape/dtype/scale。

#### 平台支持矩阵

| 平台 | dtype | 状态 |
| --- | --- | --- |
| Venus | Int8 / Int16 / Int32 | 支持 |
| ARCS | Int8 / Int16 / Int32 | 支持 |
| VenusA | Int8 / Int16 / Int32 | 支持 |

#### 参数、Shape 与内存约束

- min <= max，三输入形式的边界各为一个元素且 dtype 与 data 相同；输出 shape/dtype/scale 与输入完全相同。共享循环直接访问输入输出，不需要平台 workspace。

#### 实现注意事项

- `Clip` 不是旧文档中的 `iqClamp` 别名；当前注册/分析的有效算子是标准化后的 `Clip`。

#### 源码依据

- `executor/core/ops/clip.c`
- `tools/tpacker/graph_analysis/ops/Clip.py`
