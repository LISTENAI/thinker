### 使用 linger 导出 onnx
如果调用`linger.init(...)`接口后，使用`torch.onnx.export`会被自动替换为`linger.onnx.export`进行调用，即`torch.onnx.export = linger.onnx.export`

```python
import linger
.....
linger.init(...)
torch.onnx.export(...) # 实际上调用的是 linger.onnx.export
```

### 导出支持动态输入大小的图

``` python
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
```

其中 dynamic_axes使用有几种形式:

- 仅提供索引信息
例如下例子表示 把`input_1`的`0,2,3`维作为动态输入，第`1`仍然保持固定输入，'input_2'第`0`维作为动态输入，`output`的`0,1`维作为动态输入，对于动态输入的维度，PyTorch会自动给该维度生成一个名字以替换维度信息
``` python
dynamic_axes = {'input_1':[0, 2, 3],
                  'input_2':[0],
                  'output':[0, 1]}

```

- 对于给定的索引信息，指定名字
对于`input_1`，指定动态维0、1、2的名字分别为`batch`、`width`、`height`，其他输入同理
``` python
dynamic_axes = {'input_1':{0:'batch',
                             1:'width',
                             2:'height'},
                  'input_2':{0:'batch'},
                  'output':{0:'batch',
                            1:'detections'}
```
- 将上面两者进行混用
``` python
dynamic_axes = {'input_1':[0, 2, 3],
                  'input_2':{0:'batch'},
                  'output':[0,1]}
```
### 带有可选参数的导出
例如想命名输入输出tensor名字或者比较超前的op可以加上`torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK`
``` python
import torch
import torch.onnx
torch_model = ...
# set the model to inference mode
torch_model.eval()
dummy_input = torch.randn(1,3,244,244)
torch.onnx.export(torch_model,dummy_input,"test.onnx",
                    opset_version=11,input_names=["input"],output_names=["output"],operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
```
### torch.no_grad()
torch 1.6 版本后，需要`with torch.no_grad()`,即

``` python
import torch
import torch.onnx
torch_model = ...
# set the model to inference mode
torch_model.eval()
dummy_input = torch.randn(1,3,244,244)
with torch.no_grad():
    torch.onnx.export(torch_model,dummy_input,"test.onnx",
                        opset_version=11,input_names=["input"],output_names=["output"],operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
```
`警告`：如果不使用`with torch.no_grad()`，则会报以下错误
>RuntimeError: isDifferentiableType(variable.scalar_type()) INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/autograd/functions/utils.h":59, please report a bug to PyTorch.

