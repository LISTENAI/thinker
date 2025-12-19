import sympy
import numpy as np
from typing import List, Union

def digit_version(version_str: str) -> List[int]:
    """
    将版本字符串转换为数字列表，便于版本比较。
    
    Args:
        version_str: 版本字符串，例如 '1.6rc1'
        
    Returns:
        数字列表表示的版本，例如 [1, 5, 1] 表示 '1.6rc1'
    """
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif 'rc' in x:
            # 处理预发布版本，例如 '1.6rc1' 转换为 [1, 5, 1]
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version

# 定义最小支持的Sympy版本
sympy_minimum_version = digit_version('1.6')
sympy_version = digit_version(sympy.__version__)

def check(data) -> bool:
    """
    检查数据是否为Sympy表达式。
    
    Args:
        data: 待检查的数据
        
    Returns:
        True表示是Sympy表达式，False表示不是
    """
    if sympy_version >= sympy_minimum_version:
        return isinstance(data, sympy.core.expr.Expr)
    else:
        return isinstance(data, tuple(sympy.core.all_classes))

def is_sympy(data) -> bool:
    """
    检查数据或其元素是否为Sympy表达式。
    
    Args:
        data: 待检查的数据
        
    Returns:
        True表示是Sympy表达式或包含Sympy表达式的容器，False表示不是
    """
    if isinstance(data, (tuple, list)):
        return any(check(item) for item in data)
    elif isinstance(data, np.ndarray):
        flattened = data.reshape(-1)
        if len(flattened) < 10:
            return any(check(item) for item in flattened)
    else:
        return check(data)
    return False

__all__ = ["is_sympy"]