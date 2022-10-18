# Copyright (C) 2022 listenai Co.Ltd
# All rights reserved. 
# Created by leifang on 2022.09.31

from enum import Enum
from typing import List


def ALIGN2(x):
    return (x + 1) >> 1 << 1


def ALIGN4(x):
    return (x + 3) >> 2 << 2


def ALIGN8(x):
    return (x + 7) >> 3 << 3


def ALIGN16(x):
    return (x + 15) >> 4 << 4


def ALIGN32(x):
    return (x + 31) >> 5 << 5


class DevType(Enum):
    HIFI = 0
    LUNA = 1

    @staticmethod
    def from_str(dev: str) -> "DevType":
        if dev.upper() == "HIFI":
            return DevType.HIFI
        elif dev.upper() == "LUNA":
            return DevType.LUNA
        assert 0, "{} is not found in dev list".format(dev)


class Platform(Enum):
    VENUS = 0

    @staticmethod
    def from_str(platform: str) -> "Platform":
        if platform.upper() == "VENUS":
            return Platform.VENUS
        assert 0, "{} is not found in Platform list".format(platform)

    def get_cpu_list(platform: str) -> List[DevType]:
        if platform.upper() == "VENUS":
            return [DevType.HIFI, DevType.LUNA]

    def get_support_op(platform: str) -> List[str]:
        if platform.upper() == "VENUS":
            return [
                "Conv1dInt",
                "Conv2dInt",
                "ConvTranspose2dInt",
                "Relu",
                "Prelu",
                "Clip",
                "MaxPool",
                "AvgPool2dInt",
                "LinearInt",
                "GRUInt",
                "LSTMInt",
                "BmmInt",
                "iqSigmoid",
                "iqTanh",
                "iqCat",
                "Transpose",
                "Reshape",
                "Squeeze",
                "Unsqueeze",
                "Flatten",
                "Slice",
                "iqSum",
                "iqAdd",
                "iqMul",
                "iqDiv",
                "Resize",
                "Gather",
                "Quant",
                "Dequant",
                "SoftmaxInt",
                "ShuffleChannel",
                "LogSoftmaxInt",
                "Split",
                "BatchNorm2dInt",
                "ReduceMean",
                "Requant",
                "LayerNormInt",
                "iqVar",
            ]


class MemType(Enum):
    FLASH = 0
    PSRAM = 1
    SHARE_MEM = 2

    @staticmethod
    def from_str(dev: str) -> "MemType":
        if dev.upper() == "FLASH":
            return MemType.FLASH
        elif dev.upper() == "PSRAM":
            return MemType.PSRAM
        elif dev.upper() == "SHARE-MEM":
            return MemType.SHARE_MEM
        assert 0, "{} is not found in dev".format(dev)


class Layout(Enum):
    NCHW = 0
    NHWC = 1
    NCWH = 2
    NHWC8 = 3
    NWHC8 = 4

    @staticmethod
    def from_str(layout_str: str) -> "Layout":
        if layout_str == "NCHW":
            return Layout.NCHW
        if layout_str == "NHWC":
            return Layout.NHWC
        if layout_str == "NCWH":
            return Layout.NCWH
        if layout_str == "NHWC8":
            return Layout.NHWC8
        if layout_str == "NWHC8":
            return Layout.NWHC8


class TensorType(Enum):
    Normal = 0
    Input = 2
    Output = 3
    Emptry = 4


class LayoutPerfData:
    def __init__(self, kernel: str = None, performance: int = 10):
        self.kernel = kernel
        self.performance = performance
        self.inputs_layout = []
        self.outputs_layout = []


__all__ = [
    "ALIGN2",
    "ALIGN4",
    "ALIGN8",
    "ALIGN16",
    "ALIGN32",
    "DevType",
    "Platform",
    "MemType",
    "Layout",
    "TensorType",
    "LayoutPerfData",
]
