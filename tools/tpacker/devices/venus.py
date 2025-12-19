from .device_info import Device

# 定义VENUS平台配置
VENUS_CONFIG = {
    "name": "VENUS",
    "sram_size": 640 * 1024,  # 1024 KB
    "psram_size": 8192 * 1024,  # 8192 KB
    "dma_support": True,
    "supported_operators": [
        "ArgMax", "Conv1dInt", "Conv2dInt", "ConvTranspose2dInt",
        "Relu", "Prelu", "Clip", "MaxPool", "AvgPool2dInt",
        "LinearInt", "GRUInt", "LSTMInt", "BmmInt",
        "iqSigmoid", "iqTanh", "iqCat", "iqPad",
        "Transpose", "Reshape", "Squeeze", "Unsqueeze",
        "Flatten", "Slice", "iqSum", "iqSub", "iqAdd",
        "iqMul", "iqDiv", "Resize", "Gather",
        "Quant", "Dequant", "SoftmaxInt", "ShuffleChannel",
        "LogSoftmaxInt", "Split", "BatchNorm2dInt",
        "ReduceMean", "Requant", "LayerNormInt",
        "iqVar", "Cast", "Expand", "Tile",
        "topN", "topN2", "LogSoftmax", "Shape",
        "Packing", "GluInt"
    ],
    "supported_precision": ["INT8", "INT16", "INT32"]
}

# 创建VENUS设备实例
venus_device = Device(
    name=VENUS_CONFIG["name"],
    sram_size=VENUS_CONFIG["sram_size"],
    psram_size=VENUS_CONFIG["psram_size"],
    dma_support=VENUS_CONFIG["dma_support"]
)

# 添加支持的算子
for op in VENUS_CONFIG["supported_operators"]:
    venus_device.add_supported_operator(op)

# 添加支持的精度
for precision in VENUS_CONFIG["supported_precision"]:
    venus_device.add_supported_precision(precision)

__all__ = ["venus_device"]
