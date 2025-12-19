import os


class Device(object):
    """
    表示一个计算设备，包含设备的属性和相关操作。

    Attributes:
        name (str): 设备的名称。
        supported_operators (set): 设备支持的算子集合。
        sram_size (int): 设备的SRAM大小（字节）。
        psram_size (int): 设备的PSRAM大小（字节）。
        dma_support (bool): 是否支持DMA。
        dma_channels (int): 可用的DMA通道数量。
        max_operation_speed (int): 最大运算速度（OPS）。
        memory_bandwidth (int): 内存带宽（B/s）。
        supported_precision (set): 设备支持的数据精度集合。
    """

    def __init__(self, name: str, sram_size: int, psram_size: int, 
                 dma_support: bool = False, dma_channels: int = 0, 
                 max_operation_speed: int = 0, memory_bandwidth: int = 0, 
                 supported_precision: set = None):
        """
        初始化Device对象。

        Args:
            name (str): 设备的名称。
            sram_size (int): 设备的SRAM大小（字节）。
            psram_size (int): 设备的PSRAM大小（字节）。
            dma_support (bool, optional): 是否支持DMA。默认为False。
            dma_channels (int, optional): 可用的DMA通道数量。默认为0。
            max_operation_speed (int, optional): 最大运算速度（OPS）。默认为0。
            memory_bandwidth (int, optional): 内存带宽（B/s）。默认为0。
            supported_precision (set, optional): 设备支持的数据精度集合。默认为空集合。
        """
        self._name = name
        self._sram_size = sram_size
        self._psram_size = psram_size
        self._dma_support = dma_support
        self._dma_channels = dma_channels
        self._supported_operators = set()
        self._supported_precision = supported_precision if supported_precision is not None else set()

    @property
    def name(self) -> str:
        """获取设备的名称."""
        return self._name

    @property
    def sram_size(self) -> int:
        """获取设备的SRAM大小（字节）."""
        return self._sram_size

    @sram_size.setter
    def sram_size(self, value: int):
        """设置设备的SRAM大小（字节）."""
        if not isinstance(value, int) or value < 0:
            raise ValueError("SRAM size must be a non-negative integer.")
        self._sram_size = value

    @property
    def psram_size(self) -> int:
        """获取设备的PSRAM大小（字节）."""
        return self._psram_size

    @psram_size.setter
    def psram_size(self, value: int):
        """设置设备的PSRAM大小（字节）."""
        if not isinstance(value, int) or value < 0:
            raise ValueError("PSRAM size must be a non-negative integer.")
        self._psram_size = value

    @property
    def dma_support(self) -> bool:
        """获取设备是否支持DMA."""
        return self._dma_support

    @dma_support.setter
    def dma_support(self, value: bool):
        """设置设备是否支持DMA."""
        if not isinstance(value, bool):
            raise ValueError("DMA support must be a boolean.")
        self._dma_support = value

    @property
    def dma_channels(self) -> int:
        """获取可用的DMA通道数量."""
        return self._dma_channels

    @dma_channels.setter
    def dma_channels(self, value: int):
        """设置可用的DMA通道数量."""
        if not isinstance(value, int) or value < 0:
            raise ValueError("DMA channels must be a non-negative integer.")
        self._dma_channels = value

    @property
    def max_operation_speed(self) -> int:
        """获取设备的最大运算速度（OPS）."""
        return self._max_operation_speed

    @max_operation_speed.setter
    def max_operation_speed(self, value: int):
        """设置设备的最大运算速度（OPS）."""
        if not isinstance(value, int) or value < 0:
            raise ValueError("Max operation speed must be a non-negative integer.")
        self._max_operation_speed = value

    @property
    def memory_bandwidth(self) -> int:
        """获取设备的内存带宽（B/s）."""
        return self._memory_bandwidth

    @memory_bandwidth.setter
    def memory_bandwidth(self, value: int):
        """设置设备的内存带宽（B/s）."""
        if not isinstance(value, int) or value < 0:
            raise ValueError("Memory bandwidth must be a non-negative integer.")
        self._memory_bandwidth = value

    @property
    def supported_operators(self) -> set:
        """获取设备支持的算子集合."""
        return self._supported_operators

    def add_supported_operator(self, operator: str):
        """添加设备支持的算子."""
        if not isinstance(operator, str):
            raise ValueError("Operator must be a string.")
        self._supported_operators.add(operator)

    def remove_supported_operator(self, operator: str):
        """移除设备支持的算子."""
        if operator in self._supported_operators:
            self._supported_operators.remove(operator)

    @property
    def supported_precision(self) -> set:
        """获取设备支持的数据精度集合."""
        return self._supported_precision

    def add_supported_precision(self, precision: str):
        """添加设备支持的数据精度."""
        if not isinstance(precision, str):
            raise ValueError("Precision must be a string.")
        self._supported_precision.add(precision)

    def remove_supported_precision(self, precision: str):
        """移除设备支持的数据精度."""
        if precision in self._supported_precision:
            self._supported_precision.remove(precision)

    def __str__(self) -> str:
        """返回设备的详细信息字符串."""
        return f"Device: {self.name}\n" \
               f"SRAM Size: {self.sram_size} bytes\n" \
               f"PSRAM Size: {self.psram_size} bytes\n" \
               f"DMA Support: {self.dma_support}\n" \
               f"DMA Channels: {self.dma_channels}\n" \
               f"Max Operation Speed: {self.max_operation_speed} OPS\n" \
               f"Memory Bandwidth: {self.memory_bandwidth} B/s\n" \
               f"Supported Operators: {', '.join(self.supported_operators)}\n" \
               f"Supported Precision: {', '.join(self.supported_precision)}"

    def check_memory_usage(self, memory_usage: int) -> bool:
        """
        检查内存使用情况是否在设备的容量范围内。

        Args:
            memory_usage (int): 需要的内存大小（字节）。

        Returns:
            bool: 如果内存使用在设备容量范围内，返回True；否则返回False。
        """
        if not isinstance(memory_usage, int) or memory_usage < 0:
            raise ValueError("Memory usage must be a non-negative integer.")
        total_memory = self.sram_size + self.psram_size
        return memory_usage <= total_memory

    def allocate_memory(self, memory_size: int) -> bool:
        """
        在设备上分配指定大小的内存。

        Args:
            memory_size (int): 需要分配的内存大小（字节）。

        Returns:
            bool: 如果成功分配内存，返回True；否则返回False。
        """
        if not isinstance(memory_size, int) or memory_size < 0:
            raise ValueError("Memory size must be a non-negative integer.")
        total_memory = self.sram_size + self.psram_size
        if memory_size > total_memory:
            return False
        # 这里可以添加实际的内存分配逻辑
        return True

    def release_memory(self, memory_size: int) -> bool:
        """
        释放设备上的指定大小的内存。

        Args:
            memory_size (int): 需要释放的内存大小（字节）。

        Returns:
            bool: 如果成功释放内存，返回True；否则返回False。
        """
        if not isinstance(memory_size, int) or memory_size < 0:
            raise ValueError("Memory size must be a non-negative integer.")
        # 这里可以添加实际的内存释放逻辑
        return True

__all__ = ["Device"]
