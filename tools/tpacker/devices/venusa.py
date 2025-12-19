from .device_info import Device

VENUSA_CONFIGS = {
    "name": "ARCS",
    "sram_size": 384 * 1024,  # 384 KB
    "psram_size": 8192 * 1024,  # 8 MB
    "dma_support": True,
    "supported_operators": [
        "ArgMax", "Conv1dInt", "Conv2dInt", "ConvTranspose2dInt",
        "Relu", "Prelu", "Clip", "MaxPool", "AvgPool2dInt",
        "LinearInt", "GRUInt", "LSTMInt", "BmmInt",
        "iqSigmoid", "Sigmoid", "iqTanh", "Tanh",
        "iqCat", "iqPad", "Transpose", "Reshape",
        "Squeeze", "Unsqueeze", "Flatten", "Slice",
        "iqSum", "iqSub", "iqAdd", "iqMul", "iqDiv",
        "Gather", "Quant", "Dequant",
        "SoftmaxInt", "LogSoftmaxInt",
        "Split", "BatchNorm2dInt", "ReduceMean",
        "Requant", "LayerNormInt", "iqVar",
        "Cast", "Expand", "Tile", "topN", "topN2",
        "Shape", "Packing", "GluInt"
    ],
    "supported_precision": ["INT8", "INT16", "INT32"]
}

# 创建ARCS设备实例
venusa_device = Device(
    name=VENUSA_CONFIGS["name"],
    sram_size=VENUSA_CONFIGS["sram_size"],
    psram_size=VENUSA_CONFIGS["psram_size"],
    dma_support=VENUSA_CONFIGS["dma_support"]
)

# 添加支持的算子
for op in VENUSA_CONFIGS["supported_operators"]:
    venusa_device.add_supported_operator(op)

# 添加支持的精度
for precision in VENUSA_CONFIGS["supported_precision"]:
    venusa_device.add_supported_precision(precision)

__all__ = ["venusa_device"]
