thinker支持的算子列表

- [支持的标准onnx算子列表](#支持的标准onnx算子列表)
- [支持的quant算子列表](#支持的quant算子列表)
- [支持的OnnxInfer算子列表](#支持的onnxinfer算子列表)
- [支持的自定义算子列表](#支持的自定义算子列表)

# 支持的标准Onnx算子列表
|                        Operation                        | onnx version |
| :-----------------------------------------------------: | :----------: |
|                   [Add](#二元运算符)                     |      13      |
|                   [And](#二元运算符)                     |      7       |
|           [AveragePool](#averagepoolmaxpool)            |      11      |
|        [BatchNormalization](#batchnormalization)        |      9       |
|                    [Concat](#concat)                    |      13      |
|                      [Conv](#conv)                      |      11      |
|             [ConvTranspose](#convtranspose)             |      11      |
|                       [GRU](#gru)                       |      7       |
|                    [Gather](#gather)                    |      13      |
|                     [LSTM](#-lstm)                      |      7       |
|                 [LeakyRelu](#leakyrelu)                 |      6       |
|              [LogSoftmax](#单输入激活函数)               |      13      |
|                    [MatMul](#matmul)                    |      13      |
|                   [Max](#多元运算符)                     |      13      |
|             [MaxPool](#averagepoolmaxpool)              |      12      |
|                     [PRelu](#prelu)                     |      9       |
|               [ReduceL1](#reduce系列算子)                |      13      |
|               [ReduceL2](#reduce系列算子)                |      13      |
|                 [Relu](#单输入激活函数)                  |      14      |
|                   [Reshape](#reshape)                   |      13      |
|                    [Resize](#resize)                    |      13      |
|               [Sigmoid](#单输入激活函数)                 |      13      |
|                     [Slice](#slice)                     |      13      |
|               [Softmax](#单输入激活函数)                 |      13      |
|                     [Split](#split)                     |      11      |
|                   [Squeeze](#squeeze)                   |      13      |
|                   [Sub](#二元运算符)                     |      13      |
|                   [Sum](#多元运算符)                     |      13      |
|                 [Tanh](#三角函数运算符)                  |      9       |
|                 [Transpose](#transpose)                 |      13      |
|                [Unsqueeze](#-unsqueeze)                 |      13      |

-------------------------------------------------
# 一元运算符
- defined in [Unary_ElementWise_Ops](../offline_tool/ops/math/Unary_ElementWise_Ops.py)

### Inputs(1)
- **X**: T, required

### Outputs(1)
- **Y**: T, required

| Operation  | onnx version |         Description         | Type Constraints |
| :--------: | :----------: | :-------------------------: | :--------------: |
|    Abs     |      13      |         y = abs(x)          |       all        |
|    Ceil    |      13      |         y = ceil(x)         |  tensor(float)   |
|    Erf     |      13      |         y = erf(x)          |       all        |
|    Exp     |      13      |         y = exp(x)          |  tensor(float)   |
|   Floor    |      13      |        y = floor(x)         |  tensor(float)   |
|   IsNaN    |      13      |         y = x==NaN          |  tensor(float)   |
|  Identity  |      13      |            y = x            |       all        |
|    Log     |      13      |         y = log(x)          |  tensor(float)   |
|    Neg     |      13      |           y = -x            |      signed      |
|    Not     |      1       |           y = !x            |   tensor(int8)   |
| Reciprocal |      13      |           y = 1/x           |  tensor(float)   |
|   Round    |      11      |        y = round(x)         |  tensor(float)   |
|    Sign    |      13      | y = (x>0)? 1:(x==0)? 0 : -1 |  tensor(float)   |
|    Sqrt    |      13      |         y = sqrt(x)         |  tensor(float)   |

- **all**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)
- **signed**: tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)
  
---------
## 三角函数运算符
- defined in [Unary_ElementWise_Ops](../offline_tool/ops/math/Unary_ElementWise_Ops.py)
  
### Inputs(1)
- **X**: T, required

### Outputs(1)
- **Y**: T, required

### Type Constraints
- **T**: tensor(float)

| Operation | onnx version |      Description      |
| :-------: | :----------: | :-------------------: |
|   Acos    |      7       |       arccosine       |
|   Acosh   |      9       | hyperbolic arccosine  |
|   Asin    |      7       |        arcsine        |
|   Asinh   |      9       |  hyperbolic arcsine   |
|   Atan    |      7       |      arctangent       |
|   Atanh   |      9       | hyperbolic arctangent |
|    Cos    |      7       |        cosine         |
|   Cosh    |      9       |   hyperbolic cosine   |
|    Sin    |      7       |         sine          |
|   Sinh    |      9       |    hyperbolic sine    |
|    Tan    |      7       |        tangent        |
|   Tanh    |      1       |  hyperbolic tangent   |

---------
## IsInf
- onnx version: 10
- defined in [IsNaN.py](../offline_tool/ops/tensor/IsNaN.py)
- map infinity to true and other values to false.
  
### Inputs(1)
- **X**: T, required

### Outputs(1)
- **Y**: T, required
 
### Type Constraints
- **T**: tensor(float)

### Attributes(0-2)
- **detect_negative**: int, optional(default is 1)
  - Whether map negative infinity to true. 1: yes, 0: no
- **detect_positive**: int, optional(default is 1)
  - Whether map positive  infinity to true. 1: yes, 0: no

---------
# 二元运算符
- 都支持multidirectional broadcasting
- defined in [Binary_ElementWise_Ops.py](../offline_tool/ops/math/Binary_ElementWise_Ops.py)

### Inputs(2)
- **lhs**: T1, required
- **rhs**: T1, required

### Outputs(1)
- **ohs**: T2, required

|   Operation    | onnx version |        Description         |                        Type Constraints                        |
| :------------: | :----------: | :------------------------: | :------------------------------------------------------------: |
|      Add       |      13      |      ohs = lhs + rhs       |       T1/T2: tensor(int8), tensor(int64), tensor(float)        |
|      And       |      13      |      ohs = lhs && rhs      |                      T1/T2:  tensor(int8)                      |
|      Div       |      13      |      ohs = lhs / rhs       |       T1/T2: tensor(int8), tensor(int64), tensor(float)        |
|     Equal      |      13      |      ohs = lhs == rhs      | T1:tensor(int8), tensor(int64), tensor(float), T2:tensor(int8) |
|    Greater     |      13      |      ohs = lhs > rhs       | T1:tensor(int8), tensor(int64), tensor(float), T2:tensor(int8) |
| GreaterOrEqual |      12      |      ohs = lhs >= rhs      | T1:tensor(int8), tensor(int64), tensor(float), T2:tensor(int8) |
|      Less      |      13      |      ohs = lhs < rhs       | T1:tensor(int8), tensor(int64), tensor(float), T2:tensor(int8) |
|  LessOrEqual   |      12      |      ohs = lhs <= rhs      | T1:tensor(int8), tensor(int64), tensor(float), T2:tensor(int8) |
|      Mul       |      13      |      ohs = lhs × rhs       |       T1/T2: tensor(int8), tensor(int64), tensor(float)        |
|       Or       |      7       | ohs = lhs &#124;&#124; rhs |                      T1/T2:  tensor(int8)                      |
|      Pow       |      13      |    ohs = pow(lhs, rhs)     |       T1/T2: tensor(int8), tensor(int64), tensor(float)        |
|      Sub       |      13      |      ohs = lhs - rhs       |       T1/T2: tensor(int8), tensor(int64), tensor(float)        |
|      Xor       |      7       |      ohs = lhs ^ rhs       |                      T1/T2:  tensor(int8)                      |

## Mod
- onnx version: 13
- defined in [Mod.py](../offline_tool/ops/math/Mod.py)
- integer mod: the sign of the remainder is the same as the Divisor
- fmod：the sign of the remainder is the same as the Dividend

### Inputs(2)
- **lhs**: T, required
- **rhs**: T, required

### Outputs(1)
- **ohs**: T, required

### Type Constraints
- T:tensor(float), tensor(int8), tensor(int64)

### Attributes(0-1)
- **fmod**: int, optional(default is 0)
  - 0: integer mod; 1: fmod
  - If the input type is float, fmod attribute must be set to 1
  
---------
# 多元运算符
- 都支持multidirectional broadcasting
- defined in [Binary_ElementWise_Ops.py](../offline_tool/ops/math/Binary_ElementWise_Ops.py)

### Inputs(1 - ∞)
- **data_n**: T, required

### Outputs(1)
- **ohs**: T, required

| Operation | onnx version |     Description     |               Type Constraints                |
| :-------: | :----------: | :-----------------: | :-------------------------------------------: |
|    Max    |      13      |  ohs = max(inputs)  | T: tensor(int8), tensor(int64), tensor(float) |
|    Min    |      13      |  ohs = min(inputs)  | T: tensor(int8), tensor(int64), tensor(float) |
|    Sum    |      13      |  ohs = sum(inputs)  | T: tensor(int8), tensor(int64), tensor(float) |
|   Mean    |      13      | ohs = sum(inputs)/n | T: tensor(int8), tensor(int64), tensor(float) |

-------
# ArgMax/ArgMin
- onnx version: 13
- defined in [ReductionOps.py](../offline_tool/ops/Reduction/ReductionOps.py)
- Computes the indices of the max/min elements of the input tensor's element along the provided axis

### Inputs(1)
- **data**: T1, required

### Outputs(1)
- **reduced**: T2, required

### Type Constraints
- **T1** : tensor(int8), tensor(int64), tensor(float)
- **T2** : tensor(int64)
  
### Attributes(0-3)
- **axis** : int, optional(default is 0), accepted range is [-r, r-1] where r = rank(data)
  - The axis in which to compute the arg indices.
- **keepdims** : int, optional(default is 1)
  - Whether to keep the reduced dimension
  - 1 mean keep reduced dimension.
- **select_last_index** : int, optional(default is 0)
  - select the last index or the first index if the max/min element appears in multiple indices
  - 0: the first index; 1: the last index

--------
# AveragePool/MaxPool
- onnx version: 11
- defined in [Pool.py](../offline_tool/ops/nn/Pool.py)
- 'output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)'
- 'output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)',if ceil_mode is enabled
- 支持2d, 不支持1d

### Inputs(1)
- **X**: T, required, shape(N, C, H, W) or (N, H, W, C)，只支持2d, 不支持1d

### Outputs(1)
- **Y**: T, required
 
### Type Constraints
- **T**: tensor(int8), tensor(float)

### Attributes(0-7)
- **auto_pad**: string, optional, 被弃用，现在只支持"NOTSET"
- **kernel**: list of ints, optional,默认为(1,1)
- **stride/strides**: list of ints, optional,默认为(1,1)
- **dilation/dilations**: list of ints, optional, 默认且只支持(1，1)
- **pad/pads**: list of ints, optional,默认为(0,0,0,0)
  - 格式一般为(h_begin, w_begin, h_end, w_end)
  - 如果h_begin==h_end,w_begin==w_end, 格式可以为(h_pad,w_pad)
  - 如果四个值都相等，格式可以为(pad)
- **layout**: string, optional, "NHWC"或"NCHW", 默认为"NCHW"
- **ceil**: int, optional, 默认为0
  - Whether to use ceil or floor (default) to compute the output shape
  - 0：floor; 1:ceil 

---------
# BitShift
- onnx version: 11
- defined in [BitShift.py](../offline_tool/ops/math/BitShift.py)
- Bitwise shift operator performs element-wise operation
- supports multidirectional broadcasting
- `z= x << y, or z = x >> y`

### Inputs(2)
- **X**: T, required
  - First operand, input to be shifted.
- **Y**: T, required
  - Second operand, amounts of shift.

### Outputs(1)
- **Z**: T, required

### Type Constraints
- **T**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64)
  
### Attributes(1)
- **direction**: string, required
  - Direction of moving bits.
  - "RIGHT" (for right shift) or "LEFT" (for left shift)

---------
# Cast
- onnx version: 13
- defined in [Cast.py](../offline_tool/ops/tensor/Cast.py)
- casts the elements of a given input tensor to a data type specified by the 'to' argument and returns an output tensor of the same size in the converted type

### Inputs(1)
- **X**: T1, required

### Outputs(1)
- **Y**: T2, required

### Type Constraints
- **T1/T2**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float), tensor(double)
  
### Attributes(1)
- **to**: int, required
  - The data type to which the elements of the input tensor are cast, Strictly must be one of the types from DataType enum in TensorProto
```
TensorProto_DataType_FLOAT = 1,
TensorProto_DataType_UINT8 = 2,
TensorProto_DataType_INT8 = 3,
TensorProto_DataType_UINT16 = 4,
TensorProto_DataType_INT16 = 5,
TensorProto_DataType_INT32 = 6,
TensorProto_DataType_INT64 = 7,
TensorProto_DataType_BOOL = 9,
TensorProto_DataType_FLOAT16 = 10,
TensorProto_DataType_DOUBLE = 11,
TensorProto_DataType_UINT32 = 12,
TensorProto_DataType_UINT64 = 13,
```

--------
# Clip
- onnx version: 13
- defined in [Clip.py](../offline_tool/ops/math/Clip.py)
- Clip operator limits the given input within an interval

### Inputs(1-3)
- **X**: T, required
- **min**: T(scalar), optional(default is numeric_limits::min())
  - Minimum value, under which element is replaced by min
- **max**: T(scalar), optional(default is numeric_limits::max()) 
  - Maximum value, above which element is replaced by max

### Outputs(1)
- **Y**: T, required

### Type Constraints
- T: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float), tensor(double)

--------
# Compress
- onnx version: 11
- defined in [Compress.py](../offline_tool/ops/tensor/Compress.py)
- Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index

### Inputs(2)
- **X**: T, required
- **condition**: tensor(int8), required
  - Rank 1 tensor of booleans to indicate which slices or data elements to be selected. 
  - Its length can be less than the input length along the axis or the flattened input size if axis is not specified. In such cases data slices or elements exceeding the condition length are discarded.

### Outputs(1)
- **Y**: T, required

### Type Constraints
- **T**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float), tensor(double)

### Attributes(0-1)
- **axis**: int, optional, accepted range is [-r, r-1] where r = rank(input).
  - Axis along which to take slices. 
  - If not specified, input is flattened before elements being selected

--------
# Concat
- onnx version: 13
- defined in [Concat.py](../offline_tool/ops/tensor/Concat.py)
- Concatenate a list of tensors into a single tensor.

### Inputs(1 - ∞)
- **inputs**: T, required

### Outputs(1)
- **concat_result**: T, required

### Type Constraints
- **T**: tensor(int8), tensor(int16), tensor(int32), tensor(float)

### Attributes(1)
- **axis**: int, required, Accepted range is [-r, r-1] where r = rank(input).
  - axis along which to concate on. 

--------
# ConstantOfShape
- onnx version: 9
- defined in [ConstantOfShape.py](../offline_tool/ops/generator/ConstantOfShape.py)
- Generate a tensor with given value and shape.
 
### Inputs(1)
- **input**: T1, required, 1D tensor, All values must be >= 0
  - The shape of the expected output tensor. If empty tensor is given, the output would be a scalar

### Outputs(1)
- **output**: T2, required

### Type Constraints
- **T1**: tensor(int64)
- **T2**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)

### Attributes(0-1)
- **value**: one-element tensor, optional
  - The value of the output elements.
  - If not specified, it defaults to a tensor of value 0 and datatype float32

--------
# Conv
- onnx version: 11
- defined in [Conv.py](../offline_tool/ops/nn/Conv.py)
- convolution operator 

### Inputs(2-3)
- **X**: T, required, shape(N, C, H, W) or (N, H, W, C) or (N, C, D) or (N, C, D, H, W)...支持1d到3d
- **W**: T, required, shape(M, C/group, kH, kW) or (M, kH, kW, C/group) or (M, C/group, kD)
  - The weight tensor that will be used in the convolutions
- **B**: T, optional, shape(M,)
  - 1D bias to be added to the convolution

### Outputs(1)
- **Y**: T, required, shape(N,M,outH, outW) or (N, outH, outW, M) or(N, M, outD)
 
### Type Constraints
- **T**: tensor(int8), tensor(float)

### Attributes(0-7)
- **auto_pad**: string, optional, 被弃用，现在只支持"NOTSET"
- **kernel/kernel_shape**: list of ints, optional,默认为(1,1)
- **stride/strides**: list of ints, optional,默认为(1,1)
- **dilation/dilations**: list of ints, optional, 默认(1，1)
- **pad/pads**: list of ints, optional,默认为(0,0,0,0)
  - 格式一般为(h_begin, w_begin, h_end, w_end)
  - 如果h_begin==h_end,w_begin==w_end, 格式可以为(h_pad,w_pad)
  - 如果四个值都相等，格式可以为(pad)
- **layout**: string, optional, "NHWC"或"NCHW", 默认为"NCHW"
- **group**: int,optional, 默认为1

--------
# ConvTranspose
- onnx version: 11
- defined in [ConvTranspose.py](../offline_tool/ops/nn/ConvTranspose.py)
- `output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]`

### Inputs(2-3)

- **X**: T, required, shape(N, C, H, W) or (N, H, W, C) or (N, C, D)
- **W**: T, required, shape(C,M/group, kH, kW) or (M, kH, kW, C/group) or (M, C/group, kD)
  - The weight tensor that will be used in the convolutions
- **B**: T, optional, shape(M,)
  - 1D bias to be added to the convolution

### Outputs(1)
- **Y**: T, required, shape(N,M,outH, outW) or (N, outH, outW, M) or( N, M, outD)
 
### Type Constraints
- **T**: tensor(float),tensor(int8)

### Attributes(0-9)
- **auto_pad**: string, optional, 被弃用，现在只支持"NOTSET"
- **kernel/kernel_shape**: list of ints, optional,默认为(1,1)
- **stride/strides**: list of ints, optional,默认为(1,1)
- **dilation/dilations**: list of ints, optional, 默认(1，1)
- **pad/pads**: list of ints, optional,默认为(0,0,0,0)
  - 格式一般为(h_begin, w_begin, h_end, w_end)
  - 如果h_begin==h_end,w_begin==w_end, 格式可以为(h_pad,w_pad)
  - 如果四个值都相等，格式可以为(pad)
- **layout**: string, optional, "NHWC"或"NCHW", 默认为"NCHW"
- **group**: int,optional, 默认为1
- **output_padding**: list of ints, optional, 默认为(0,0,0,0)
  - Additional elements added to the side with higher coordinate indices in the output.
- **output_shape**: list of ints, optional, 暂不支持
  
--------
# CumSum
- onnx version: 14
- defined in [CumSum.py](../offline_tool/ops/math/CumSum.py)
- performs cumulative sum of the input elements along the given axis

### Inputs(2)
- **X**: T1, required
- **axis**: T2, required, scalar, Accepted range is [-r, r-1] where r = rank(input)

### Outputs(1)
- **Y**: T1, required

### Type Constraints
- T1:tensor(uint32), tensor(uint64), tensor(int32), tensor(int64), tensor(float)
- T2:tensor(int64)

### Attributes(0-2)
- **exclusive**: int, optional (default is 0)
  - 1: the j-th output element would be the sum of the first (j-1) elements. 1st output = 0
  - 0: the j-th output element would be the sum of the first j elements
- **reverse**: int, optional (default is 0)
  - 1: perform the sums in reverse direction.

--------
# Det
- onnx version: 11
- defined in [Det.py](../offline_tool/ops/math/Det.py)
- calculates determinant of a square matrix

### Inputs(1)
- **X**: T, required, shape(*,M,M)

### Outputs(1)
- **Y**: T, required, shape(*)
  
### Type Constraints
- **T**: tensor(float)

--------
# Dropout
- onnx version: 13
- defined in [Dropout.py](../offline_tool/ops/nn/Dropout.py)
- the output is a random dropout of input if training_mode is true 
- `output = scale * data * mask, scale = 1. / (1. - ratio), mask = random(0 or 1) `

### Inputs(1-3)
- **data**: T1, required
- **ratio**: T1, optional(default is 0.5), accepted range is[0, 1)
  - If this input was not set, or if it was set to 0, the output would be a simple copy of the input
  - If it's non-zero, output will be a random dropout of the scaled input
- **training_mode**：T2, optional(default is 0)
  - whether to dropout or not

### Outputs(1-2)
- **output**: T1, required
- **mask**: T2, optional

### Type Constraints
- **T1**: tensor(float) 
- **T2**: tensor(int8) 

### Attributes(0-1)
- **seed**: int, optional
  - seed to the random generator

--------
# Expand
- onnx version: 13
- defined in [Expand.py](../offline_tool/ops/tensor/Expand.py)
- broadcast the input tensor into the given shape

### Inputs(2)
- **X**: T, required
- **shape**: tensor(int64), required
  - A 1-D tensor indicates the shape you want to expand to

### Outputs(1)
- **Y**: T, required

### Type Constraints
- **T**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)
  
--------
# EyeLike
- onnx version: 9
- defined in [EyeLike.py](../offline_tool/ops/tensor/EyeLike.py)
- generate a 2D tensor with ones on the diagonal and zeros everywhere else.
  
### Inputs(1)
- **X**: T, required

### Outputs(1)
- **Y**: T, required

### Type Constraints
- **T**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)

### Attributes(0-2)
- **dtype**: int, optional, 可取值和Cast的to属性一样
  - The data type for the elements of the output tensor. 
  - If not specified,the data type of the input tensor T is used. 
  - If input tensor T is also notspecified, then type defaults to 'float'.
- **k**: int, optional(default is 0)
  - Y[i, i+k] = 1
  
--------
# Flatten
- onnx version: 13
- defined in [Flatten.py](../offline_tool/ops/tensor/Flatten.py)
- Flattens the input tensor into a 2D matrix.
- `[D0, D1,....,Dn] ---> [D0*D1*...Daxis-1, Daxis *Daxis+1*....Dn]`

### Inputs(1)
- **X**: T, required

### Outputs(1)
- **Y**: T, required

### Type Constraints
- **T**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)

### Attributes(0-1)
- **axis**: int, optional(default is 1), accepted range is [-r, r-1] where r = rank(input)

--------
# GRU
- onnx version: 7
- defined in [GRU.py](../offline_tool/ops/rnn/Gru.py)
- Computes an one-layer GRU.

## Inputs(3-6)
- **data**:T, required
- **i2h_weight**:T, required
- **h2h_weight**:T, required
- **bias**:T, optional
- **mask**: tensor(int32), optional, sequence_lens
- **kHistH**:T, optional
    
## Outputs(1-2)
- **out**: T, required
- **hidden_o**, optional

### Type Constraints
- **T**: tensor(int8),tensor(float)

### Attributes(0-5)
- **go_forward**: int, optional(default is 1)
  - must be 1-forward(default), 0-reverse, or 2-bidirectional.
- **batch_first**: int, optional(default is 0)
  - layout of data
  - 0-[seq_lens, batch_size, input_size], 1-[batch_size, seq_lens, input_size]
- **hidden_size**: int, optional(default is 0) 
- **no_bias**:int, optional(default is 0)
- **multi_io**:int, optional(default is 0)
  - 0-no hidden/cell in I/O tensors, 1-hidden & cell in I/O tensors.

--------
# Gather
- onnx version: 13
- defined in [Gather.py](../offline_tool/ops/tensor/Gather.py)
- gather entries of the axis dimension of data indexed by indices, and concatenates them in output 
- `data:[d0,d1,....,dn], indices:[i0,i1...,in] ----> out:[d0,...,daxis-1,i0,...,in,daxis+1,...,dn]`

### Inputs(2)
- **data**: T, required
- **indices**: tensor(int64), required
  - All index values are expected to be within bounds [-s, s-1] along axis of size s

### Outputs(1)
- **Y**: T1, required

### Type Constraints
- **T**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)

### Attributes(0-1)
- **axis**: int, optional(default is 0), accepted range is [-r, r-1] where r = rank(input)
  - which axis to gather on

--------
# GatherElements
- onnx version: 13
- defined in [GatherElements.py](../offline_tool/ops/tensor/GatherElements.py)
- an indexing operation that produces its output by indexing into the input data tensor at index positions determined by elements of the indices tensor
- `output[d0]...[dn] = input[d0]...[daxis = k]...[dn], where k=indices[d0]...[dn] ` 

### Inputs(2)
- **data**: T, required
- **indices**: tensor(int64), required
  - All index values are expected to be within bounds [-s, s-1] along axis of size 
  - rank(data) == rank(indices)

### Outputs(1)
- **output**: T, required, shape(output) = shape(indices)

### Type Constraints
- **T**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)

### Attributes(0-1)
- **axis**: int, optional(default is 0), accepted range is [-r, r-1] where r = rank(input)

--------
# GatherND
- onnx version: 13
- defined in [GatherND.py](../offline_tool/ops/tensor/GatherND.py)
- Given data tensor of rank r >= 1, indices tensor of rank q >= 1, and batch_dims integer b, this operator gathers slices of data into an output tensor of rank q + r - indices_shape[-1 ] - 1 - b

### Inputs(2)
- **data**: T, required
- **indices**: tensor(int64)
, required
  - All index values are expected to be within bounds [-s, s-1] along axis of size s

### Outputs(1)
- **output**: T, required

### Type Constraints
- **T**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)

### Attributes(0-1)
- **batch_dims**: int, optional(default is 0)
  - the leading batch_dims number of dimensions of data tensor and indices are representing the batches, and the gather starts from the batch_dims+1 dimension

--------
# Gemm
- onnx version: 13
- defined in [Gemm.py](../offline_tool/ops/math/Gemm.py)
- General Matrix multiplication
- `Y = alpha * A * B + beta * C`

### Inputs(2-3)
- **A**: T, required
  - The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
- **B**: T, required
  - The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
- **C**: T, optional(default is a scalar 0)
  - The shape of C should be unidirectional broadcastable to (M, N).

### Outputs(1)
- output: T, required

### Type Constraints
- T: tensor(float), tensor(int8)

### Attributes(0-4)
- **alpha**: float, optional(default is 1.0) 
- **beta**: float, optional(default is 0.0) 
- **transA**: int, optional(default is 0)
- **transB**: int, optional(default is 1)

--------
# GlobalAveragePool/GlobalMaxPool
- onnx version: 1
- defined in [Pool.py](../offline_tool/ops/nn/Pool.py)
- applies average/max pooling across the values in the same channel. 
- This is equivalent to AveragePool/MaxPool with kernel size equal to the spatial dimension of input tensor.

### Inputs(1)
- **X**: T, required

### Outputs(1)
- **Y**: T, required

### Type Constraints
- **T**: tensor(float), tensor(int8)

--------
# GlobalLpPool
- onnx version: 1
- defined in [Pool.py](../offline_tool/ops/nn/Pool.py)
- applies lp pool pooling across the values in the same channel. 
- This is equivalent to LpPool with kernel size equal to the spatial dimension of input tensor.

### Inputs(1)
- **X**: T, required

### Outputs(1)
- **Y**: T, required

### Type Constraints
- **T**: tensor(float), tensor(int8)

### Attributes(0-1)
- **p**: int, optional(default is 2)

--------
## Hardmax
- onnx version: 13
- defined in [Hardmax.py](../offline_tool/ops/math/Hardmax.py)
- output = 1 if the input element is the first maximum value along the specified axis, 0 otherwise

### Inputs(1)
- **input**: T, required

### Outputs(1)
- **output**: T, required

### Type Constraints
- **T**: tensor(float)

### Attributes(0-1)
- **axis**: int, optional(default is -1).Accepted range is [-r, r-1] where r = rank(input)
 
--------
## LSTM
- onnx version: 7
- defined in [Lstm.py](../offline_tool/ops/rnn/Lstm.py)
- computes an one-layer LSTM.
  
### Inputs(3-9)
- **data**:T, required
- **i2h_weight**: T, required
- **h2h_weight**: T, required
- **bias**: T, optional
- **mask**: tensor(int32), optional, sequence_lens
- **HistH**: T, optional
- **HistC**: T, optional
- **PeepHoles**：T, 暂未支持
- **Project**：T, 暂未支持

### Output(1-3)
- **out**: T, required
- **hidden_o**: T, optional
- **cell_o**: T, optional

### Type Constraints
- **T**: tensor(int8), tensor(float)

### Attributes(0-9)
- **direction**: int, optional(default is 0)
  - must be 0-forward(default), 1-reverse, or 2-bidirectional.
- **batch_first**: int, optional(default is 0)
  - layout of data
  - 0-[seq_lens, batch_size, input_size], 1-[batch_size, seq_lens, input_size]
- **hidden_size**: int, optional(default is 0) 
- **state_size**: int, optional(default is 0)
- **no_bias**: int, optional(default is 0)
- **no_mask**: int, optional(default is 0)
- **multi_io**: int, optional(default is 0)
  - 0-no hidden/cell in I/O tensors, 1-hidden & cell in I/O tensors.
- **project**: int, optional(default is 0)
- **mode**: int, optional(default is 0)
  - 0-normal, 1-peepholes

--------
## LpPool
- onnx version: 11
- defined in [Pool.py](../offline_tool/ops/nn/Pool.py)
- 'output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)'
- 'output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)',if ceil_mode is enabled

### Inputs(1)
- **X**: T, required, shape(N, C, H, W) or (N, H, W, C)

### Outputs(1)
- **Y**: T, required
 
### Type Constraints
- **T**: tensor(int8), tensor(float)

### Attributes(0-8)
- **auto_pad**: string, optional, 被弃用，现在只支持"NOTSET"
- **kernel**: list of ints, optional,默认为(1,1)
- **stride/strides**: list of ints, optional,默认为(1,1)
- **dilation/dilations**: list of ints, optional, 默认且只支持(1，1)
- **pad/pads**: list of ints, optional,默认为(0,0,0,0)
  - 格式一般为(h_begin, w_begin, h_end, w_end)
  - 如果h_begin==h_end,w_begin==w_end, 格式可以为(h_pad,w_pad)
  - 如果四个值都相等，格式可以为(pad)
- **layout**: string, optional, "NHWC"或"NCHW", 默认为"NCHW"
- **ceil**: int, optional, 默认为0
  - Whether to use ceil or floor (default) to compute the output shape
  - 0：floor; 1:ceil 
- **p**: int, optional(default is 2)
  
--------
## MatMul
- onnx version: 13
- defined in [MatMul.py](../offline_tool/ops/math/MatMul.py)
- `C = A * B`
- A,B可以是多维的，当Ashape=(a0,a1,...,an),Bshape=(b0,b1,....bn)时
  - `M = a0 * a1 *....* an-1`
  - `K = an = b0 * b1 *...* bn-1`
  - `N = bn`
  - `Cshape=(a0,a1,...an-1,bn)`

### Inputs(2)
- **A**: T, required
- **B**: T, required

### Outputs(1)
- **C**: T, required

### Type Constraints
- **T**: T: tensor(float), tensor(int8)
  
--------
## MaxRoiPool
- onnx version: 1
- defined in [RoiPool.py](../offline_tool/ops/nn/RoiPool.py)
- apply max pooling across each RoI(region of interests)

### Inputs(2)
- **X**: T, required, shape(N,C,H,W)
- **rois**: T, required, shape(num_rois, 5) given as [[batch_id, x1, y1, x2, y2], ...].

### Outputs(1)
- **Y**: T, required, shape (num_rois, channels, pooled_shape[0], pooled_shape[1]).

### Type Constraints
- **T**: T: tensor(float)

### Attributes(1-2)
- **pooled_shape**: list of ints, required
  - ROI pool output shape (height, width).
- **spatial_scale**: float, optional(default is 1.0)
  - Multiplicative spatial scale factor to translate ROI coordinates from their input scale to the scale used when pooling.

--------
## MaxUnPool
- onnx version: 11
- defined in [UnPool](../offline_tool/ops/nn/UnPool.py)
- computes the partial inverse of the MaxPool op
- if output_shape is not specified
  - `original_h = (input_h - 1) * stride_h + kernel_h - pad_h_begin - pad_h_end`
  - `original_w = (input_w - 1) * stride_w + kernel_w - pad_w_begin - pad_w_end`

### Inputs(2,3)
- **X**: T1, required
- **indices** : T1, required
  - rank(indices) == rank(X) 
  - containing the indices corresponding to elements in the first input tensor X
  - The indices are linear, computed considering the tensor as flattened 1-D tensor, assuming row-major storage. 
  - the linear indices should not consider padding. 
  - the values in indices are in the range [0, size(X))
- **output_shape**: T2, optional
  - explicit set the shape of the output
  - if specified, pads values will be auto generated, and attr['pads' ] are ignored

### Outputs(1)
- **Y**: T1, required

### Type Constraints
- **T1**: tensor(float)
- **T2**: tensor(int64)

### Attributes(0-3)
- **kernel/kernel_shape**: list of ints, optional(default is (1,1))
- **pad/pads**: list of ints, optional(default is (0,0,0,0))
- **stride/strides**: list of ints, optional(default is(1,1))

--------
## NonMaxSuppression
- onnx version: 11
- defined in [NonMaxSuppression.py](../offline_tool/ops/object_detection/NonMaxSuppression.py)
- filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes

### Inputs(2-5)
- **boxes**: T1, required, shape(num_batches, spatial_dimension, 4)
- **scores**: T1, required, shape(num_batches, num_classes, spatial_dimension) 
- **max_output_boxes_per_class**: T2(scalar), optional(default is 0)
  - the maximum number of boxes to be selected per batch per class
- **iou_threshold**: T1(scalar), optional(default is 0), value range [0, 1]
  - the threshold for deciding when to remove boxes based on IOU
- **score_threshold**: T1(scalar), optional
  - the threshold for deciding when to remove boxes based on score
  - default means do not remove boxes based on score

### Outputs(1)
- **selected_indices**: T2, required, shape(num_selected_indices, 3), format is [[batch_index, class_index, box_index], ...]

### Type Constraints
- **T1**: tensor(float)
- **T2**: tensor(int64)

### Attributes(0-1)
- **center_point_box**: int, optional(default is 0)
  - indicate the format of the box data
  - 0: [y1, x1, y2, x2], (y1, x1) and (y2, x2) are the coordinates of any diagonal pair of box corners
  - 1: [x_center, y_center, width, height]

--------
## NonZero
- onnx version: 13
- defined in [NonZero.py](../offline_tool/ops/tensor/NonZero.py)
- Returns the indices of the elements that are non-zero (in row-major order - by dimension)

### Inputs(1)
- **X**: T1, required

### Outputs(1)
- **Y**: T2, required, shape(rank(X), numOfNonZeroElements)

### Type Constraints
- **T1**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)
- **T2**: tensor(int64)

--------
## OneHot
- onnx version: 11
- defined in [OneHot.py](../offline_tool/ops/tensor/OneHot.py)
- The locations represented by the index values in the 'indices' input tensor will have 'on_value' and the other locations will have 'off_value' in the output tensor
- `output_shape = indices_shape[0:axis] + depth + indices_shape[axis:]`
- `output[d0]..[daxis=k]...[dn] = on_value, where k = indices[d0]..d[n]`
  
### Inputs(3)
- **indices**: T1, required
  - In case 'indices' is of non-integer type, the values will be casted to int64 before use.
- **depth**: T2(Scalar), required
  - the size of the one-hot dimension (specified by 'axis' attribute) added on in the output tensor
  - The values in the 'indices' input tensor are expected to be in the range [-depth, depth-1]
  - In case 'depth' is of non-integer type, the values will be casted to int64 before use.
- **values**: T3, required
  - Rank 1 tensor containing exactly two elements, in the format [off_value, on_value]
  - 'on_value' is the value used for filling locations specified in 'indices' input tensor, and 'off_value' is the value used for filling locations other than those specified in 'indices' input tensor.
  
### Outputs(1)
- **output**: T3, required

### Type Constraints
- **T1**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)
- **T2**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)
- **T3**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)
  
### Attributes(0-1)
- **axis**: int, optional(default is -1).Accepted range is [-r-1, r] where r = rank(input)
  - Axis along which one-hot representation in added
  - axis=-1 means that the additional dimension will be inserted as the innermost/last dimension in the output tensor

--------
## Pad
- onnx version: 13
- defined in [Pad.py](../offline_tool/ops/tensor/Pad.py)
- three supported modes
  - constant: pads with a given constant value as specified by constant_value (which defaults to 0)
  - reflect: pads with the reflection of the vector mirrored on the first and last values of the vector along each axis
  - edge: pads with the edge values of array

### Inputs(2-3)
- **data**: T1, required
- **pads**: T2, required
  - indicating the number of padding elements to add or remove (if negative) at the beginning and end of each axis
  - should be a 1D tensor of shape [2 * rank(input)].
  - format: [x1_begin, x2_begin,...,x1_end, x2_end,...], where xi_begin is the number of pad values added at the beginning of axis 'i' and xi_end, the number of pad values added at the end of axis 'i'.
- **constant_value**: T1(scalar), optional(default is 0)
  -  A scalar value to be used if the mode chosen is 'constant' 

### Outputs(1)
- **output**: T2, required

### Type Constraints
- **T1**: tensor(int8), tensor(float)
- **T2**: tensor(int64)

### Attributes(0-1)
- **mode**: string, optional(default is 'constant')
  - 'constant', 'reflect', 'edge'

--------
## RandomNormal
- onnx version:1
- defined in [Random.py](../offline_tool/ops/generator/Random.py)
- generate a tensor with random values drawn from a normal distribution

### Inputs(0)

### Outputs(1)
- **output**: T, required

### Type Constraints
- **T**: tensor(float)

### Attributes(1-5)
- **shape**: list of ints, required
  - the shape of the output tensor.
- **dtype**: int, optional(default is 1), 也只支持1,因为这个算子目前只能生成float类型的输出
  - The data type for the elements of the output tensor
- **mean**: float, optional(default is 0.0)
  - The mean of the normal distribution
- **scale**: float, optional(default is 1.0)
  - The standard deviation of the normal distribution
- **seed**: float, optional
  - seed to the random generator

--------
## RandomNormalLike
- onnx version:1
- defined in [Random.py](../offline_tool/ops/generator/Random.py)
- generate a tensor with random values drawn from a normal distribution
- the shape of the output tensor is copied from the shape of the input tensor

### Inputs(1)
- **input**: T1, required

### Outputs(1)
- **output**: T2, required, shape = shape(input)

### Type Constraints
- **T1**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)
- **T2**: tensor(float)

### Attributes(0-4)
- **dtype**: int, optional, 只支持1,因为这个算子目前只能生成float类型的输出
  - The data type for the elements of the output tensor
  - if not provided, we will use the data type of the input tensor
- **mean**: float, optional(default is 0.0)
  - The mean of the normal distribution
- **scale**: float, optional(default is 1.0)
  - The standard deviation of the normal distribution
- **seed**: float, optional
  - seed to the random generator
  
--------
## RandomUniform
- onnx version:1
- defined in [Random.py](../offline_tool/ops/generator/Random.py)
- Generate a tensor with random values drawn from a uniform distribution

### Inputs(0)

### Outputs(1)
- **output**: T, required

### Type Constraints
- **T**: tensor(float)

### Attributes(1-5)
- **shape**: list of ints, required
  - the shape of the output tensor.
- **dtype**: int, optional(default is 1), 也只支持1,因为这个算子目前只能生成float类型的输出
  - The data type for the elements of the output tensor
- **high**: float, optional(default is 1.0)
  - Upper boundary of the output values
- **low**: float, optional(default is 0.0)
  - Lower boundary of the output values
- **seed**: float, optional
  - seed to the random generator

--------
## RandomUniformLike
- onnx version:1
- defined in [Random.py](../offline_tool/ops/generator/Random.py)
- generate a tensor with random values drawn from a uniform distribution
- the shape of the output tensor is copied from the shape of the input tensor

### Inputs(1)
- **input**: T1, required

### Outputs(1)
- **output**: T2, required, shape = shape(input)

### Type Constraints
- **T1**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)
- **T2**: tensor(float)

### Attributes(0-4)
- **dtype**: int, optional, 只支持1,因为这个算子目前只能生成float类型的输出
  - the data type for the elements of the output tensor
  - if not provided, we will use the data type of the input tensor
- **high**: float, optional(default is 1.0)
  - Upper boundary of the output values
- **low**: float, optional(default is 0.0)
  - Lower boundary of the output values
- **seed**: float, optional
  - seed to the random generator
  
--------
## Range
- onnx version:11
- defined in [Range.py](../offline_tool/ops/generator/Range.py)
- generate a tensor containing a sequence of numbers that begin at start and extends by increments of delta up to limit (exclusive)

### Inputs(3)
- **start**: T(scalar), required
  - first entry for the range of output values
- **limit**: T(scalar), required
  - exclusive upper limit for the range of output values
- **delta**: T(scalar), required
  - value to step by

### Outputs(1)
- **output**: T, required

### Type Constraints
- **T**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)
  
--------
## Reduce系列算子
- defined in [ReductionOps.py](../offline_tool/ops/Reduction/ReductionOps.py)
- computes the reduced results of the input tensor's element along the provided axes

### Inputs(1)
- **data**: T, required

### Outputs(1)
- **reduced**: T, required

### Type Constraints
- **T**:  tensor(int8), tensor(int64), tensor(float)

### Attributes(0-2)
- **axes**: list of ints, optional, accepted range is [-r, r-1] where r = rank(data)
  - default is to reduce over all the dimensions of the input tensor.
- **keepdims**: int, optional(default is 1)
  - Keep the reduced dimension or not, default 1 mean keep reduced dimension.

|    Operation    | onnx version |              Description               |
| :-------------: | :----------: | :------------------------------------: |
|    ReduceL1     |      13      |     L1 norm, y = sum(abs(inputs))      |
|    ReduceL2     |      13      | L2 norm, y = sqrt(sum(square(inputs))) |
|  ReduceLogSum   |      13      |          y = log(sum(inputs))          |
| ReduceLogSumExp |      13      |       y = log(sum(exp(inputs)))        |
|    ReduceMax    |      13      |            y = max(inputs)             |
|   ReduceMean    |      13      |           y = sum(inputs)/n            |
|    ReduceMin    |      13      |            y = min(inputs)             |
|   ReduceProd    |      13      |            y = prod(inputs)            |
|    ReduceSum    |      13      |            y = sum(inputs)             |
| ReduceSumSquare |      13      |        y = sum(square(inputs))         |

--------
## Reshape
- onnx version: 13
- defined in [Reshape.py](../offline_tool/ops/tensor/Reshape.py)
- Reshape the input tensor to the output shape specified by a shape tensor
  
### Inputs(2)
- **data**: T, required
- **shape**: tensor(int64), required
  - at most one dimension of the new shape can be -1. In this case, the value is inferred from the size of the tensor and the remaining dimensions. 
  - a dimension could also be 0, in which case the actual dimension value is unchanged (i.e. taken from the input tensor).

### Outputs(1)
- **reshaped**: T, required

### Type Constraints
- **T**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)

--------
## Resize
- onnx version:13
- defined in [Resize.py](../offline_tool/ops/tensor/Resize.py)
- calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor
  
### Inputs(1-4)
- **X**: T1, required
- **roi**: T2, optional, 暂不支持
  - 1-D tensor given as [start1, ..., startN, end1, ..., endN]
  - used when coordinate_transformation_mode is "tf_crop_and_resize"
- **scales**: tensor(float), optional
  - output_dimension = input_dimension * scale
- **sizes**: tensor(int64), optional
  - only one of 'scales' and 'sizes' can be specified.
  - give the size of the output tensor.

### Outputs(1)
- **Y**: T1, required

### Type Constraints
- **T1**: tensor(float)
- **T2**: tensor(float)

### Attributes(0-4)
- **coordinate_transformation_mode**: string, optional(default is 'half_pixel')
  - 'half_pixel': `x_original = (x_resized + 0.5) / scale - 0.5`
  - 'pytorch_half_pixel': `x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0`
  - 'align_corners': `x_original = x_resized * (length_original - 1) / (length_resized - 1)`
  - 'asymmetric': `x_original = x_resized / scale`
  - 'tf_crop_and_resize': `x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized * (end_x - start_x) * (length_original - 1) / (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original - 1)`
- **cubic_coeff_a**: float, optional(default is -0.75)
  - the coefficient a used in cubic interpolation.
- **mode**:string, optional(default is nearest)
  - 'nearest', 'linear' or 'cubic'
- **nearest_mode**:string, optional(default is round_prefer_floor)
  - only used by 'nearest' mode, indicates how to get "nearest" pixel in input tensor from x_original
  - 'round_prefer_floor', 'round_prefer_ceil', 'floor', or 'ceil'

--------
## ReverseSequence
- onnx version:10
- defined in [ReverseSequence.py](../offline_tool/ops/tensor/ReverseSequence.py)
- reverse batch of sequences having different lengths specified by sequence_lens
- for each slice i iterating on batch axis, the operator reverses the first sequence_lens[i] elements on time axis, and copies elements whose index's beyond sequence_lens[i] to the output.

### Inputs(2)
- **input**: T, required
- **sequence_lens**: tensor(int64), required, shape=[batch_size, ]
  - Tensor specifying lengths of the sequences in a batch

### Outputs(1)
- **Y**: T, required

### Type Constraints
- **T**: tensor(int8), tensor(float)

### Attributes(0-2)
- **batch_axis**: int, optional(default is 1)
  - Specify which axis is batch axis. Must be one of 1, or 0.
- **time_axis**: int, optional(default is 0)
  - Specify which axis is time axis. Must be one of 1, or 0.

-------
## RoiAlign
- onnx version :10
- defined in [RoiAlign.py](../offline_tool/ops/object_detection/RoiAlign.py)
- apply pooling across each RoI
- in each ROI bin, the value of the sampled locations are computed directly through bilinear interpolation

### Inputs(3)
- **X**: T1, required, shape should be [N,C,H,W ]
- **rois**: T1, required
  - a 2-D input of shape (num_rois, 4) given as [[x1, y1, x2, y2], .]
- **batch_indices**: T2, required
  - 1-D tensor of shape (num_rois,) with each element denoting the index of the corresponding image in the batch.

### Outputs(1)
- **Y**: T1, required, shape = [num_rois, C, output_height, output_width]

### Type Constraints
- **T1**: tensor(float)
- **T2**: tensor(int64)

### Attributes(0-5)
- **mode**: string, optional(default is avg)
  - 'avg' or 'max'
- **output_height**: int, optional(default is 1)
- **output_width**: int, optional(default is 1)
- **sampling_ratio**: int, optional(default is 0)
  - number of sampling points in the interpolation grid used to compute the output value of each pooled output bin.
  - If == 0, an adaptive number of grid points are used (computed as ceil(roi_width / output_width))
- **spatial_ratio**: float, optional(default is 1.0)
  - multiplicative spatial scale factor to translate ROI coordinates from their input spatial scale to the scale used when pooling

--------
## ScatterElements
- onnx version: 13
- defined in [ScatterElements.py](../offline_tool/ops/tensor/ScatterElements.py)
- The output is produced by creating a copy of the input data, and then updating its value to values specified by updates at specific index positions specified by indices
- `output[d0]..[daxis=k]..[dn]= updates[d0]...[dn], where k= indices[d0]...[dn]`

### Inputs(3)
- **data**: T, required
- **indices**: tensor(int64), required
- **updates**: T, required

### Outputs(1)
- **output**: T, required, shape = shape(data)

### Type Constraints
- **T**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)

### Attributes(0-1)
- **axis**: int, optional(default is -1).Accepted range is [-r, r-1] where r = rank(input)

--------
## ScatterND
- onnx version: 13
- defined in [ScatterND.py](../offline_tool/ops/tensor/ScatterND.py)
- The output of the operation is produced by creating a copy of the input data, and then updating its value to values specified by updates at specific index positions specified by indices
```
output = np.copy(data)
update_indices = indices.shape[:-1]
for idx in np.ndindex(update_indices):
    output[indices[idx]] = updates[idx]
```

### Inputs(3)
- **data**: T, required
- **indices**: tensor(int64), required
- **updates**: T, required
  - rank = rank(indices) + rank(data) - shape(indices)[-1 ] -1

### Outputs(1)
- **output**: T, required, shape = shape(data)

### Type Constraints
- **T**: tensor(float)

--------
## Shape
- onnx version: 13
- defined in [Shape.py](../offline_tool/ops/tensor/Shape.py)
- takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
  
### Inputs(1)
- **data**: T1, required

### Outputs(1)
- **shape**: T2, required

### Type Constraints
- **T1**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)
- **T2**: tensor(int64)

--------
## Shrink
- onnx version: 9
- defined in [Shrink.py](../offline_tool/ops/nn/Shrink.py)
- `if x < -lambd, y = x + bias; if x > lambd, y = x - bias; otherwise, y = 0.`

### Inputs(1)
- **X**: T, required

### Outputs(1)
- **Y**: T, required

### Type Constraints
- **T**: tensor(float)

### Attributes(0-2)
- **bias**: float, optional(default is 0.0)
- **lamda**: float, optional(default is 0.5)

--------
## Size
- onnx version: 9
- defined in [Size.py](../offline_tool/ops/tensor/Size.py)
- calculate the total number of elements of the input tensor

### Inputs(1)
- **data**: T1, required

### Outputs(1)
- **size**: T2(scalar), required

### Type Constraints
- **T1**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)
- **T2**: tensor(int64)

--------
## Slice
- onnx version: 13
- defined in [Slice.py](../offline_tool/ops/tensor/Slice.py)
- produces a slice of the input tensor along multiple axes.
- 现在的实现只支持对一个axes进行操作，即starts,ends,axes,steps必须是一个int或者元素个数为1的tensor

### Inputs(3-5)
- **data**: T, required
- **starts**: tensor(int64), required
  - 1-D tensor of starting indices of corresponding axis in `axes`
- **ends**:: tensor(int64), required
  - 1-D tensor of ending indices (exclusive) of corresponding axis in `axes`
- **axes**:: tensor(int64), optional
  - 1-D tensor of axes that `starts` and `ends` apply to, accepted range is [-r, r-1] where r = rank(data)
  - if omitted, set to [0, ..., ndim-1]
- **steps**:: tensor(int64), optional
  - 1-D tensor of slice step of corresponding axis in `axes`
  - 'steps' cannot be 0. Defaults to 1.
  
### Outputs(1)
- **output**: T, required

### Type Constraints
- **T**: tensor(int8), tensor(int16), tensor(float)

--------
## Split
- onnx version: 11
- 现在支持的版本11的split,即split位于属性而不是输入
- defined in [Split.py](../offline_tool/ops/tensor/Split.py)
- split a tensor into a list of tensors, along the specified 'axis'

### Inputs(1)
- **input**: T, required

### Outputs(1 - ∞)
- **outputs**: T, required

### Type Constraints
- **T**: tensor(int8), tensor(int16), tensor(int32), tensor(float)

### Attributes(1-2)
- **axis**: int, optional(default is 0).Accepted range is [-r, r-1] where r = rank(input)
- **split**: tensor(int64), optional
  - length of each output. values should be >= 0. Sum of the values must be equal to the dim value at 'axis' specified.
  - if not provided, the tensor is split to equal sized parts.

--------
## Split-13
- onnx version: 13
- defined in [Split.py](../offline_tool/ops/tensor/Split.py)
- split a tensor into a list of tensors, along the specified 'axis'

### Inputs(1-2)
- **input**: T, required
- **split**: tensor(int64), optional
  - length of each output. values should be >= 0. Sum of the values must be equal to the dim value at 'axis' specified.
  - if not provided, the tensor is split to equal sized parts.

### Outputs(1 - ∞)
- **outputs**: T, required

### Type Constraints
- **T**: tensor(int8), tensor(int16), tensor(int32), tensor(float)

### Attributes(0-1)
- **axis**: int, optional(default is 0).Accepted range is [-r, r-1] where r = rank(input)

--------
## Squeeze
- onnx version: 13
- defined in [Squeeze.py](../offline_tool/ops/tensor/Squeeze.py)
- remove single-dimensional entries from the shape of a tensor.

### Inputs(1-2)
- **data**: T, required
- **axes**: tensor(int64), optional
  - list of integers indicating the dimensions to squeeze, accepted range is [-r, r-1] where r = rank(input)
  - if not provided, all the single dimensions will be removed from the shape

### Outputs(1)
- **output**: T, required

### Type Constraints
- **T**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)

--------
## Tile
- onnx version: 13
- defined in [Tile.py](../offline_tool/ops/tensor/Tile.py)
- construct a tensor by tiling a given tensor

### Inputs(2)
- **input**: T, required
- **repeats**: tensor(int64), required

### Outputs(1)
- **output**: T, required
  - output_dim[i] = input_dim[i] * repeats[i]

### Type Constraints
- **T**: tensor(int64), tensor(float)

--------
## TopK
- onnx version: 11
- defined in [TopK.py](../offline_tool/ops/math/TopK.py)
- Retrieve the top-K largest or smallest elements along a specified axis.

### Inputs(2)
- **X**: T, required
- **K**: tensor(int64), scalar, required
  - a single positive value corresponding to the number of top elements to retrieve

### Outputs(2)
- **values**: T, required
  - top K values from the input tensor
- **indices**: tensor(int64), required
  - the corresponding input tensor indices for the top K values.

### Type Constraints
- **T**: tensor(float)

### Attributes(0-3)
- **axis**: int, optional(default is 0).Accepted range is [-r, r-1] where r = rank(input)
  - dimension on which to do the sort
- **largest**: int, optional(default is 1)
  - whether to return the top-K largest or smallest elements
  - 1: largest; 0: smallest
- **sorted**: int, optional(default is 1)
  - whether to return the elements in sorted order

--------
## Transpose
- onnx version: 13
- defined in [Transpose.py](../offline_tool/ops/tensor/Transpose.py)
- transpose the input tensor

### Inputs(1)
- **X**: T, required

### Outputs(1)
- **Y**: T, required

### Type Constraints
- **T**: tensor(int8), tensor(float)

### Attributes(0-1)
- **perm**: list of ints, optional
  - default is reverse the dimensions

--------
## Unsqueeze
- onnx version: 13
- defined in [Unsqueeze.py](../offline_tool/ops/tensor/Unsqueeze.py)
- insert single-dimensional entries to the shape of an input tensor

### Inputs(2)
- **data**: T, required
- **axes**: tensor(int64), optional
  - list of integers indicating the dimensions to be inserted. Accepted range is [-r, r-1] where r = rank(output)

### Outputs(1)
- **output**: T, required

### Type Constraints
- **T**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)

--------
## Where
- onnx version: 9
- defined in [Where.py](../offline_tool/ops/tensor/Where.py)
- return elements, either from X or Y, depending on condition
- support multidirectional broadcasting 

### Inputs(3)
- **condition**: tensor(int64), required
  - when True (nonzero):output = X, otherwise output = Y
- **X**: T, required
- **Y**: T, required

### Outputs(1)
- **output**: T, required

### Type Constraints
- **T**: tensor(int64),tensor(float)

--------
# 激活函数

--------
## 单输入激活函数
- defined in [Activations.py](../offline_tool/ops/activation/Activations.py)
  
### Inputs(1)
- **X**: T, required

### Outputs(1)
- **Y**: T, required

### Type Constraints
- **T**: tensor(float)

|    Operation    | onnx version |                                   Description                                    |                                          Attributes                                           |
| :-------------: | :----------: | :------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------: |
|      Celu       |      12      |                   y = max(0,x) + min(0,alpha*(exp(x/alpha)-1))                   |                          **alpha**: float, optional(default is 1.0)                           |
|       Elu       |      6       |              y = alpha * (exp(x) - 1.) for x < 0, y = x for x >= 0               |                          **alpha**: float, optional(default is 1.0)                           |
|   HardSigmoid   |      6       |                       y = max(0, min(1, alpha * x + beta))                       |     **alpha**: float, optional(default is 0.2); **beta**: float, optional(default is 0.5)     |
|    LeakyRelu    |      6       |                    y = alpha * x for x < 0, y = x for x >= 0                     |                          **alpha**: float, optional(default is 0.01)                          |
|   LogSoftmax    |      13      |                               y = log(Softmax(x))                                |                            **axis**: int, optional(default is -1)                             |
|      Relu       |      14      |                                  y = max(0, x)                                   |                                               \                                               |
|      Selu       |      6       |      y = gamma * (alpha * e^x - alpha) for x <= 0, y = gamma * x for x > 0       | **alpha**: float, optional(default is 1.67326); **gamma**: float, optional(default is 1.0507) |
|     Sigmoid     |      13      |                              y =  1 / (1 + exp(-x))                              |                                               \                                               |
|     Softmax     |      13      | Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1) |                            **axis**: int, optional(default is -1)                             |
|    Softplus     |      1       |                                y = ln(exp(x) + 1)                                |                                               \                                               |
|    Softsign     |      1       |                                 y = x/(1+abs(x))                                 |                                               \                                               |
| ThresholdedRelu |      10      |                       y = x for x > alpha, y = 0 otherwise                       |                          **alpha**: float, optional(default is 1.0)                           |
--------
## PRelu
- onnx version: 9
- defined in [Activations.py](../offline_tool/ops/activation/Activations.py)
- `y = slope * x for x < 0, y = x for x >= 0`

### Inputs(2)
- **X**: T, required
- **slope**: T, required
  - tensor slope should be unidirectional broadcastable to input tensor X

### Outputs(1)
- **Y**: T, required

### Type Constraints
- **T**: tensor(float), tensor(int8), tensor(int64)

--------
# Normalization算子
--------
## BatchNormalization
- onnx version: 9
- defined in [BatchNormalization.py](../offline_tool/ops/nn/BatchNormalization.py)
- `y = scale * (x - mean) / sqrt(var + epsilon) + bias`
- mean and variance are computed per channel.

### Inputs(5)
- **X**: T, required, shape(N, C, D0..Dn), 维度不大于4, 一般为(N, C, H, W)
- **scale**: T, required, shape(C)
- **bias**: T, required, shape(C)
- **mean**: T, required, shape(C)
- **var**: T, required, shape(C)

### Outputs(1)
- **Y**: T, required, shape = X.shape 

### Type Constraints
- **T**: tensor(float)

### Attributes(0-1)
- **epsilon**: float,optional, default is 1e-05
  - The epsilon value to use to avoid division by zero.

--------
## InstanceNormalization
- onnx version: 6
- defined in [InstanceNormalization.py](../offline_tool/ops/nn/InstanceNormalization.py)
- `y = scale * (x - mean) / sqrt(variance + epsilon) + bias`
- mean and variance are computed per instance per channel.

### Inputs(3)
- **X**: T, required, shape(N, C, D0..Dn), 一般为(N, C, H, W)
- **scale**: T, required, shape(C)
- **bias**: T, required, shape(C)

### Outputs(1)
- **Y**: T, required, shape = X.shape 

### Type Constraints
- **T**: tensor(float)

### Attributes(0-1)
- **epsilon**: float,optional, default is 1e-05
  - The epsilon value to use to avoid division by zero.

--------
## LRN
- onnx version: 13
- defined in [LRN.py](../offline_tool/ops/nn/LRN.py)
- Local Response Normalization: normalizes over local input regions, which is defined across the channels
- `square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2), where max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))`
- `Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta`

### Inputs(1)
- **X**: T, required, 现在的实现只支持shape(N, C, H, W)

### Outputs(1)
- **Y**: T, required, shape = X.shape 

### Type Constraints
- **T**: tensor(float)

### Attributes(1-4)
- **size**: int, required
  - The number of channels to sum over
- **alpha**: float,optional, default is 1e-05
- **beta**: float,optional, default is 0.75
- **bias**: float,optional, default is 1.0

--------
## LpNormalization
- onnx version: 1
- defined in [LpNormalization.py](../offline_tool/ops/nn/LpNormalization.py)
  
### Inputs(1)
- **X**: T, required

### Outputs(1)
- **Y**: T, required

### Type Constraints
- **T**: tensor(float)

### Attributes(0-2)
- **axis**: int, optional(default is -1)
- **p**: int ,optional(default is 2),only 1 or 2 are supported

--------
## MeanVarianceNormalization
- onnx version: 6
- defined in [MeanVarianceNormalization.py](../offline_tool/ops/tensor/MeanVarianceNormalization.py)
- `y = (x-E(x)) / sqrt(E(x^2)-E(x)^2)`
- E(x) and E(x^2) are computed according the axes
  
### Inputs(1)
- **X**: T, required

### Outputs(1)
- **Y**: T, required

### Type Constraints
- **T**: tensor(float)

### Attributes(0-1)
- **axes**: list of int, optional(default is [0, 2, 3])
  - A list of integers, along which to reduce
  - default axes [0 ,2 ,3] means calculating E(x) and E(x^2) along each channel.

--------
# 支持的quant算子列表

|               Operation               |
| :-----------------------------------: |
| [DequantizeLinear](#dequantizelinear) |
|           [GRUInt](#gruint)           |
|         [iqAdd](#iqbinaryop)          |
|         [iqDiv](#iqbinaryop)          |
|         [iqMul](#iqbinaryop)          |
|         [iqSub](#iqbinaryop)          |
|        [LinearInt](#linearint)        |
|          [LSTMInt](#lstmint)          |
|      [QLinearConv](#qlinearconv)      |
|    [QLinearDeConv](#qlineardeconv)    |
|    [QLinearMatMul](#qlinearmatmul)    |
|  [QuantizeConvert](#quantizeconvert)  |
|   [QuantizeLinear](#quantizelinear)   |
| [RequantizeLinear](#requantizelinear) |

---------
## DequantizeLinear
- defined in [DequantizeLinear.py](../offline_tool/ops/quant/DequantizeLinear.py)
- uint8/int8 -> float
- `y = x / scale`

### Inputs(2)
- **X**: T, required
- **scale**: tensor(float), scalar, required

### Outputs(1)
- **Y**: T, required
 
### Type Constraints
- **T**: tensor(int8), tensor(uint8)

--------
## GRUInt
- quantized gru
- defined in [GRUInt.py](../offline_tool/ops/quant/GRUInt.py)

### Inputs(7-13)
- **data**: tensor(int8), required
- **data_scale**: tensor(float), scalar, required
- **i2h_weight**: tensor(int8), required
- **i2h_weight_scale**: tensor(float), scalar, required
- **h2h_weight**: tensor(int8), required
- **h2h_weight_scale**: tensor(float), scalar, required
- **i2h_bias**: tensor(int8), optional
- **i2h_bias_scale**: tensor(float), scalar, optional
- **h2h_bias**: tensor(int8), optional
- **h2h_bias_scale**: tensor(float), scalar, optional
- **HistH**: tensor(int8), optional
- **out_scale**: tensor(float), scalar, required
- **hidden_scale**: : tensor(float), scalar, optional

### Outputs(1-2)
- **out**: tensor(int8), required
- **hidd_oen**: optional

### Attributes(0-5)
- **go_forward**: int, optional(default is 1)
  - must be 1-forward(default), 0-reverse, or 2-bidirectional.
- **batch_first**: int, optional(default is 0)
  - layout of data
  - 0-[seq_lens, batch_size, input_size], 1-[batch_size, seq_lens, input_size]
- **hidden_size**: int, optional(default is 0) 
- **no_bias**:int, optional(default is 0)
- **multi_io**:int, optional(default is 0)
  - 0-no hidden/cell in I/O tensors, 1-hidden & cell in I/O tensors.

--------

## iqBinaryOp
- 包括iqAdd, iqDiv, iqMul, iqSub

### Inputs(2)
- **lhs**: tensor(int8), required
- **rhs**: tensor(int8), required

### Outputs(1)
- **ohs**: tensor(int8), required

### Attributes(3-5)
- **scale_x**: float, required
- **scale_y**: float, required
- **scale_o**: float, required
- **quant_type**: string, optional(default is 'normal_quant')
  - 现在的实现只支持xdnn_kMaxValue
- **reserve**: 未实现

| Operation |                    Description                    |                definition file                |
| :-------: | :-----------------------------------------------: | :-------------------------------------------: |
|   iqAdd   | ohs / scale_o = (lhs / scale_x) + (rhs / scale_y) | [iqAdd.py](../offline_tool/ops/quant/iqAdd.py) |
|   iqDiv   | ohs / scale_o = (lhs / scale_x) / (rhs / scale_y) | [iqDiv.py](../offline_tool/ops/quant/iqDiv.py) |
|   iqMul   | ohs / scale_o = (lhs / scale_x) × (rhs / scale_y) | [iqMul.py](../offline_tool/ops/quant/iqMul.py) |
|   iqSub   | ohs / scale_o = (lhs / scale_x) - (rhs / scale_y) | [iqSub.py](../offline_tool/ops/quant/iqSub.py) |

--------
## LinearInt
- quantized gemm
- defined in [LinearInt.py](../offline_tool/ops/quant/LinearInt.py)
- `(Y / scale_o) = alpha * (A / scale_a) * (B /scale_b) + beta * (C / scale_c)`

### Inputs(5 or 7)
- **A**: tensor(int8), required
  - The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
- **scale_a**: float required
- **B**: tensor(int8), required
  - The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
- **scale_b**: float, required
- **C**: tensor(int8), optional(default is a scalar 0)
  - The shape of C should be unidirectional broadcastable to (M, N).
- **scale_c**: float, optional
- **scale_o**: float, required

### Outputs(1)
- **Y**: tensor(int8), required

### Attributes(0-4)
- **alpha**: float, optional(default is 1.0) 
- **beta**: float, optional(default is 0.0) 
- **transA**: int, optional(default is 0)
- **transB**: int, optional(default is 1)

--------
## LSTMInt
- quantized lstm
- defined in [LstmInt.py](../offline_tool/ops/quant/LstmInt.py)

### Inputs(7-20)
- **data**: tensor(int8), required
- **data_scale**: tensor(float), scalar, required
- **i2h_weight**: tensor(int8), required
- **i2h_weight_scale**: tensor(float), scalar, required
- **h2h_weight**: tensor(int8), required
- **h2h_weight_scale**: tensor(float), scalar, required
- **i2h_bias**: tensor(int8), optional
- **i2h_bias_scale**: tensor(float), scalar, optional
- **h2h_bias**: tensor(int8), optional
- **h2h_bias_scale**: tensor(float), scalar, optional
- **mask**: tensor(int32), optional, sequence_lens
- **HistH**: tensor(int8), optional
- **HistC**: tensor(int8), optional
- **PeepHoles**：optional, 暂未支持
- **PHScale**：optional, 暂未支持
- **Project**：optional, 暂未支持
- **PScale**：optional, 暂未支持
- **out_scale**: tensor(float), scalar, required
- **hidden_scale**: tensor(float), scalar, optional
- **cell_scale**：tensor(float), scalar, optional

### Output(1-3)
- **out**: tensor(int8), required
- **hidden_o**: optional
- **cell_o**: optional

### Attributes(0-9)
- **go_forward**: int, optional(default is 1)
  - must be 1-forward(default), 0-reverse, or 2-bidirectional.
- **batch_first**: int, optional(default is 0)
  - layout of data
  - 0-[seq_lens, batch_size, input_size], 1-[batch_size, seq_lens, input_size]
- **hidden_size**: int, optional(default is 0) 
- **state_size**: int, optional(default is 0)
- **no_bias**: int, optional(default is 0)
- **no_mask**: int, optional(default is 1)
- **multi_io**: int, optional(default is 0)
  - 0-no hidden/cell in I/O tensors, 1-hidden & cell in I/O tensors.
- **project**: int, optional(default is 0)
- **mode**: int, optional(default is 0)
  - 0-normal, 1-peepholes

--------
## QLinearConv
- quantized Conv
- defined in [QLinearConv.py](../offline_tool/ops/quant/QLinearConv.py)

### Inputs(8-9)
- **Data**: T1, required, 
- **DataScale**: tensor(float), scalar, required
- **DataZero**: T2, required, 暂未支持，但需要输入一个tensor
- **Weight**: T1, required, 
- **WeightScale**: tensor(float), scalar, required,
- **WeightZero**: T2, required, 暂未支持，但需要输入一个tensor
- **OutputScale**: tensor(float), scalar, required
- **OutputZero**: T2, required, 暂未支持，但需要输入一个tensor
- **Bias**: T3, optional

### Outputs(1)
- **Y**: T1, required
 
### Type Constraints
- **T1**: tensor(int8)
- **T2**: tensor(int8)
- **T3**: tensor(int32)

### Attributes(0-7)
- **auto_pad**: string, optional, 被弃用，现在只支持"NOTSET"
- **kernel/kernel_shape**: list of ints, optional,默认为(1,1)
- **stride/strides**: list of ints, optional,默认为(1,1)
- **dilation/dilations**: list of ints, optional, 默认(1，1)
- **pad/pads**: list of ints, optional,默认为(0,0,0,0)
  - 格式一般为(h_begin, w_begin, h_end, w_end)
  - 如果h_begin==h_end,w_begin==w_end, 格式可以为(h_pad,w_pad)
  - 如果四个值都相等，格式可以为(pad)
- **layout**: string, optional, "NHWC"或"NCHW", 默认为"NCHW"
- **group**: int,optional, 默认为1
- **quant_type**: string, optional(default is 'qmax_quant')
  - xdnn_kMaxValue or xdnn_kQValue

--------
## QLinearDeConv
- quantized ConvTranspose
- defined in [QLinearDeConv.py](../offline_tool/ops/quant/QLinearDeConv.py)

### Inputs(5,7)
- **Data**: T1, required
- **DataScale**: tensor(float), scalar, required
- **Weight**: T1, required
- **WeightScale**: tensor(float), scalar, required
- **Bias**: T2, optional
- **BiasScale**: tensor(float), scalar, optional
- **OutScale**: tensor(float), scalar, required

### Outputs(1)
- **Y**: T1, required
 
### Type Constraints
- **T1**: tensor(int8)
- **T2**: tensor(int32)

### Attributes(0-9)
- **auto_pad**: string, optional, 被弃用，现在只支持"NOTSET"
- **kernel/kernel_shape**: list of ints, optional,默认为(1,1)
- **stride/strides**: list of ints, optional,默认为(1,1)
- **dilation/dilations**: list of ints, optional, 默认(1，1)
- **pad/pads**: list of ints, optional,默认为(0,0,0,0)
  - 格式一般为(h_begin, w_begin, h_end, w_end)
  - 如果h_begin==h_end,w_begin==w_end, 格式可以为(h_pad,w_pad)
  - 如果四个值都相等，格式可以为(pad)
- **layout**: string, optional, "NHWC"或"NCHW", 默认为"NCHW"
- **group**: int,optional, 默认为1
- **output_padding**: list of ints, optional, 暂不支持
- **output_shape**: list of ints, optional, 暂不支持
  
--------
## QLinearMatMul
- quantized MatMul
- defined in [QLinearMatMul.py](../offline_tool/ops/quant/QLinearMatMul.py)
- 现在版本只有打包代码，还未实现

### Inputs(8)
- **A**: T1, required, 
- **AScale**: tensor(float), scalar, required
- **AZero**: T2, required, 暂未支持，但需要输入一个tensor
- **B**: T1, required, 
- **BScale**: tensor(float), scalar, required,
- **BZero**: T2, required, 暂未支持，但需要输入一个tensor
- **CScale**: tensor(float), scalar, required
- **CZero**: T2, required, 暂未支持，但需要输入一个tensor

### Outputs(1)
- **C**: T1, required

### Type Constraints
- **T1**: tensor(int8)
- **T2**: tensor(int8)
  
--------
## QuantizeConvert
- 只支持int8->uint8和int8->uint8的cast
- defined in [QuantizeConvert.py](../offline_tool/ops/quant/QuantizeConvert.py)

### Inputs(1)
- **X**: T1, required

### Outputs(1)
- **Y**: T2, required

### Type Constraints
- **T1/T2**: tensor(uint8),tensor(int8)
  
### Attributes(1)
- **to**: int, required
  - TensorProto_DataType_UINT8 = 2
  - TensorProto_DataType_INT8 = 3

---------
## QuantizeLinear
- defined in [QuantizeLinear.py](../offline_tool/ops/quant/QuantizeLinear.py)
- float -> int8
- `y = x * scale`

### Inputs(2)
- **X**: tensor(float), required
- **scale**: tensor(float), scalar, required

### Outputs(1)
- **Y**: tensor(int8), required
 
### Attributes(0-2)
- **data_bits**: int, optional(default is 8), 只支持8
- **platform_quant**: string, optional(default is 'normal_quant')
  - 支持xdnn_kQValue, xdnn_kMaxValue两种模式
  - 'normal_quant': xdnn_kMaxValue模式, scale = scale
  - other: xdnn_kQValue模式，scale = (1 << int(scale))

---------
## RequantizeLinear
- defined in [RequantizeLinear.py](../offline_tool/ops/quant/RequantizeLinear.py)
- float -> int8
- `y = x * yscale / xscale`

### Inputs(3)
- **X**: tensor(float), required
- **XScale**: tensor(float), scalar, required
- **YScale**: tensor(float), scalar, required

### Outputs(1)
- **Y**: tensor(int8), required
  
--------
# 支持的OnnxInfer算子列表

|                           Operation                           |
| :-----------------------------------------------------------: |
|                 [AvgPool2dInt](#avgpool2dint)                 |
|        [BatchNorm2dInt](#batchnorm2dintbatchnorm1dint)        |
|        [BatchNorm1dInt](#batchnorm2dintbatchnorm1dint)        |
|               [Conv1dInt](#conv1dintconv2dint)                |
|               [Conv2dInt](#conv1dintconv2dint)                |
|           [ConvTranspose2dInt](#convtranspose2dint)           |
|                        [iqcat](#iqcat)                        |
|                      [iqClamp](#iqclamp)                      |
| [iqmul_Iu8_Is8_Os8](#iqmuliqmul_iu8_is8_os8iqmul_is8_iu8_os8) |
| [iqmul_Is8_Iu8_Os8](#iqmuliqmul_iu8_is8_os8iqmul_is8_iu8_os8) |
|                    [iqSigmoid](#iqsigmoid)                    |
|                       [iqTanh](#iqtanh)                       |
|               [OnnxInferQuant](#onnxinferquant)               |
|             [OnnxInferDeQuant](#onnxinferdequant)             |
|            [ResizeBilinearInt](#resizebilinearint)            |

---------
## 实现方式介绍
- 这些算子的输入和输出都是int8类型，通过converter转换成对应的target算子(支持float类型)实现
- 初始算子的Attributes相较于target算子，多了一些scale项，记录着对应inputs和outputs的scale
- 第一种转换操作：IntOp to FloatOp
  - 适用于target算子没有scale输入的情况
  - 添加[DequantizeLinear算子](#dequantizelinear), 把初始算子int8类型的输入转换成float
  - 添加target算子，完成实际的计算步骤
  - 添加[QuantizeLinear算子](#quantizelinear), 把target得到的float类型的输出转换回int8
- 第二种转换操作：
  - 适用于targte算子有scale输入的情况
  - 把初始算子Attributes的scale项转化成输入项，然后再调用target算子完成实际的计算步骤
- 算子的converter转换操作见[converter.py](../offline_tool/converter.py)

## AvgPool2dInt
- 实际工作交给 [AveragePool](#averagepoolmaxpool)算子实现
- 新增的属性放在AveragePool原有属性的前方
  
### 新增的Attributes(2)
- **scale_x**: float
- **scale_o**: float

---------
## BatchNorm2dInt/BatchNorm1dI
- 用第一种转换把实际工作交给[BatchNormalization](#batchnormalization) 

### Inputs(5)
- **X**: tensor(int8), required, shape(N, C, D0..Dn), 维度不大于4, 一般为(N, C, H, W)
- **scale**: tensor(int8), required, shape(C)
- **bias**: tensor(int8), required, shape(C)
- **mean**: T, required, shape(C)
- **var**: T, required, shape(C)

### Outputs(1)
- **Y**: tensor(int8), required, shape = X.shape 

### Type Constraints
- **T**: tensor(float)

### Attributes(4-5)
- **scale_mul_x**：float, required, scale for X
- **scale_mul_w**：float, required, scale for scale
- **scale_add_b**：float, required, scale for bias
- **scale_add_o**：float, required, scale for Y
- **epsilon**: float,optional, default is 1e-05
  - The epsilon value to use to avoid division by zero.

---------
## Conv1dInt/Conv2dInt
- 用第二种转换把实际工作交给[QLinearConv](#qlinearconv)  
- QLinearConv的DataZero，WeightZero，OutputZero输入都设为0

### Inputs(2-3)
- **Data**: tensor(int8), required, 
- **Weight**: tensor(int8), required, 
- **Bias**: tensor(int32), optional
  - if provided , BiasScale = x_scale * w_scale

### Outputs(1)
- **Y**: tensor(int8), required
 
### Attributes(3-10)
- **scale_x**：float, required, 转换成QLinearConv里的DataScale输入
- **scale_w**：float, required, 转换成QLinearConv里的WeightScale输入
- **scale_o**：float, required, 转换成QLinearConv里的OutScale输入
- **auto_pad**: string, optional, 被弃用，现在只支持"NOTSET"
- **kernel/kernel_shape**: list of ints, optional,默认为(1,1)
- **stride/strides**: list of ints, optional,默认为(1,1)
- **dilation/dilations**: list of ints, optional, 默认(1，1)
- **pad/pads**: list of ints, optional,默认为(0,0,0,0)
  - 格式一般为(h_begin, w_begin, h_end, w_end)
  - 如果h_begin==h_end,w_begin==w_end, 格式可以为(h_pad,w_pad)
  - 如果四个值都相等，格式可以为(pad)
- **layout**: string, optional, "NHWC"或"NCHW", 默认为"NCHW"
- **group**: int,optional, 默认为1
  
---------
## ConvTranspose2dInt
- 用第二种转换把实际工作交给[QLinearDeConv](#qlineardeconv) 

### Inputs(2-3)
- **Data**: tensor(int8), required
- **Weight**: tensor(int8), required
- **Bias**: tensor(int32), optional
  - if provided , BiasScale = x_scale * w_scale
  
### Outputs(1)
- **Y**: T1, required
 
### Type Constraints
- **T1**: tensor(int8)
- **T2**: tensor(int32)

### Attributes(3-12)
- **scale_x**：float, required, 转换成QLinearDeConv里的DataScale输入
- **scale_w**：float, required, 转换成QLinearDeConv里的WeightScale输入
- **scale_o**：float, required, 转换成QLinearDeConv里的OutScale输入
- **auto_pad**: string, optional, 被弃用，现在只支持"NOTSET"
- **kernel/kernel_shape**: list of ints, optional,默认为(1,1)
- **stride/strides**: list of ints, optional,默认为(1,1)
- **dilation/dilations**: list of ints, optional, 默认(1，1)
- **pad/pads**: list of ints, optional,默认为(0,0,0,0)
  - 格式一般为(h_begin, w_begin, h_end, w_end)
  - 如果h_begin==h_end,w_begin==w_end, 格式可以为(h_pad,w_pad)
  - 如果四个值都相等，格式可以为(pad)
- **layout**: string, optional, "NHWC"或"NCHW", 默认为"NCHW"
- **group**: int,optional, 默认为1
- **output_padding**: list of ints, optional, 暂不支持
- **output_shape**: list of ints, optional, 暂不支持
  
---------
## iqcat
-  用第一种转换把实际工作交给[Concat](#concat)

### Inputs(1 - ∞)
- **inputs**: tensor(int8), required

### Outputs(1)
- **concat_result**: tensor(int8), required

### Attributes(3 - ∞)
- **dim**: int, required, 转换成Concat里的axis属性
- **scale_x_i**:  float, required, scale for inputs[i]
- **scale_o**: float, required, scale for concat_result

---------
## iqClamp
- 用第一种转换把实际工作交给[Clip](#clip)

### Inputs(1)
- **X**: tensor(int8), required

### Outputs(1)
- **Y**: tensor(int8), required

### Attributes(4)
- **min**: float,int, required, 转换成Clip里的min输入
- **max**: float,int, required, 转换成Clip里的max输入
- **scale_x**：float, required, scale for x
- **scale_o**：float, required, scale for y

---------
## iqmul/iqmul_Iu8_Is8_Os8/iqmul_Is8_Iu8_Os8
- 用第一种转换把实际工作交给[Mul](#二元运算符)  

### Inputs(2)
- **lhs**: tensor(int8), required
- **rhs**: tensor(int8), required

### Outputs(1)
- **ohs**: tensor(int8), required

### Attributes(3)
- **scale_x**：float, required, scale for lhs
- **scale_y**：float, required, scale for rhs
- **scale_o**：float, required, scale for ohs
  
---------
## iqSigmoid   
- 用第一种转换把实际工作交给[Sigmoid](#单输入激活函数) 

### Inputs(1)
- **X**: tensor(int8), required

### Outputs(1)
- **Y**: tensor(int8), required

### Attributes(2)
- **scale_x**：float, required, scale for x
- **scale_y**：float, required, scale for y
  
--------
## iqTanh
- 用第一种转换把实际工作交给[Tanh](#三角函数运算符)

### Inputs(1)
- **X**: tensor(int8), required

### Outputs(1)
- **Y**: tensor(int8), required

### Attributes(2)
- **scale_x**：float, required, scale for x
- **scale_y**：float, required, scale for y

## OnnxInferQuant
- 用第二种转换把实际工作交给[QuantizeLinear](#quantizelinear) 

### Inputs(1)
- **X**: tensor(float), required

### Outputs(1)
- **Y**: tensor(int8), required

### Attributes(1-3)
- **scale_x**: float, required
  - 转换成QuantizeLinear里的scale输入
- **data_bits**: int, optional(default is 8), 只支持8
- **platform_quant**: string, optional(default is 'normal_quant')
  - 支持xdnn_kQValue, xdnn_kMaxValue两种模式
  - 'normal_quant': xdnn_kMaxValue模式, scale = scale
  - other: xdnn_kQValue模式，scale = (1 << int(scale))

--------
## OnnxInferDeQuant
- 用第二种转换把实际工作交给[DequantizeLinear](#dequantizelinear)

### Inputs(2)
- **X**: T, required

### Outputs(1)
- **Y**: T, required
 
### Type Constraints
- **T**: tensor(int8), tensor(uint8)

### Attributes(1)
- **scale_o**: float, required
  - 转换成DequantizeLinear里的scale输入

--------
## ResizeBilinearInt 
- 用第一种转换把实际工作交给[Resize](#resize)
- 输入输出和Resize一样

### Attributes(比resize相比多了两个属性)
- **scale_x**：float, required, scale for x
- **scale_y**：float, required, scale for y
  
--------
# 支持的自定义算子列表
|             Operation             |
| :-------------------------------: |
|        [MhaMask](#mhamask)        |
|       [ShiftPad](#shiftpad)       |
| [ShuffleChannel](#shufflechannel) |

--------
## MhaMask
- defined in [MhaMask.py](../offline_tool/ops/plugin/MhaMask.py)
- 每个batch中，在(height, width)大小的图像中截取h=w, h=w-window_width两条直线之间的区域
  - 截取区域内output[h][w] = input[h][w]
  - 截取区域外output[h][w] = forward_val

### Inputs(2)
- **input**: T, required, shape=(batch, height, width)
- **mask**: T, required, shape=(batch, width)
  - 每个batch中，如果mask[h]或mask[w]为0, output[h][w]=forward_val

### Outputs(1)
- **output**: T, required
 
### Type Constraints
- **T**: tensor(float)

### Attributes(3)
- **backward_val**: float, required, 未支持
- **forward_val**: float, required
- **window_width**: int, required

--------
## ShiftPad
- defined in [ShiftPad.py](../offline_tool/ops/math/ShiftPad.py)
- output对应axis的前index元素来自history,后一个元素来自data
 
### Inputs(2)
- **data**: T, required
- **history**: T, required
- **Index**: tensor(float), scalar

### Outputs(1)
- **output**: T, required
 
### Type Constraints
- **T**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)

### Attributes(1)
- **axis**: int, required, >=0

--------
## ShuffleChannel
- defined in [ShuffleChannel.py](../offline_tool/ops/plugin/ShuffleChannel.py)
- 初始输入的channel可以分成group组，每组包含channel/num_group个channel
- 初始输入的排列顺序是group0_c0,group0_c1.....group1_c0,group1_c1
- 转化成group0_c0,group1_c0.....group0_c1,group1_c1....

### Inputs(2)
- **X**: T, required, (NCHW) or(NHWC)

### Outputs(1)
- **Y**: T, required
 
### Type Constraints
- **T**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64),tensor(float)

### Attributes(0-2)
- **num_group**: int, optional(default is 1)
- **axis**: int, optional(default is 1)
  - 1:NCHW, 3:NHWC

