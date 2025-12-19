from .device_info import *
from .venus import *
from .arcs import *
from .mars import *
from .venusa import *
from ..enum_defines import Colors

def create_device_from_name(platform_name):
    """
    根据平台名称创建对应的Device实例。
    
    Args:
        platform_name (str): 平台名称，如 "VENUSA" 或 "ARCS"。
        
    Returns:
        Device: 创建的Device实例。
        
    Raises:
        ValueError: 如果平台名称不存在。
    """
    if platform_name.upper() == 'VENUS':
        return venus_device
    elif platform_name.upper() == 'MARS':
        return mars_device
    elif platform_name.upper() == 'ARCS':
        return arcs_device
    elif platform_name.upper() == 'VENUSA':
        return venusa_device
    else:
        raise ValueError(f"{Colors.RED}Platform {platform_name} is not supported.{Colors.RESET}")

__all__ = ["Device", 'venus_device', 'mars_device', 'arcs_device', 'venusa_device', 'create_device_from_name']