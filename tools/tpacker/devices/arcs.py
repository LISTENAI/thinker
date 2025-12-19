from .device_info import Device

ARCS_CONFIGS = {
    "name": "ARCS",
    "sram_size": 384 * 1024,  # 4096 KB
    "psram_size": 8192 * 1024,  # 32768 KB
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
    "supported_precision":["INT8", "INT32"]
}

# 创建ARCS设备实例
arcs_device = Device(
    name=ARCS_CONFIGS["name"],
    sram_size=ARCS_CONFIGS["sram_size"],
    psram_size=ARCS_CONFIGS["psram_size"],
    dma_support=ARCS_CONFIGS["dma_support"]
)

# 添加支持的算子
for op in ARCS_CONFIGS["supported_operators"]:
    arcs_device.add_supported_operator(op)

# 添加支持的精度
for precision in ARCS_CONFIGS["supported_precision"]:
    arcs_device.add_supported_precision(precision)

__all__ = ["arcs_device"]
