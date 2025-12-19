from typing import List
from collections import defaultdict

from ...graph import Graph


class MethodManager(object):
    """method管理类
    管理所有method，并封装了method apply接口
    类成员变量：
      methods: Dict[str, func]
        用于存储op融合方式；
    """

    methods = defaultdict()

    @classmethod
    def apply(cls, g: Graph, ignore_methods: List[str]) -> Graph:

        """遍历所有的methods进行算子融合
        Args:
          g: Graph
            待融合graph
          specify_methods: List[str]
            只指定特定算子融合, 其他方式将忽略；
          ignore_methods: List[str]
            忽略特定算子融合；
        Returns:
          g: Graph
            融合后的graph
        """
        support_methods = list(cls.methods.keys())
        methods = set(support_methods)
        if len(ignore_methods) > 0:
            methods.difference_update(set(ignore_methods))

        for method_name in support_methods:
            if method_name in methods:
                g = cls.methods[method_name](g)

        return g

    @classmethod
    def remove(cls, name):
        """删除指定method"""
        del cls.methods[name]

    @classmethod
    def clear(cls):
        """清空所有的method"""
        cls.methods.clear()


def register_method(name):
    """用于注册method的装饰器"""

    def wrapper(cls):
        MethodManager.methods[name] = cls
        return cls

    return wrapper


__all__ = ["MethodManager", "register_method"]
