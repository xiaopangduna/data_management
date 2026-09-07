"""
DemoClass 模块
包含一个简单的类和加法方法，用于演示测试用例编写
"""
import logging

logger = logging.getLogger(__name__)

class DemoClass:
    """
    演示类，用于展示如何编写Python测试
    """

    def __init__(self):
        """
        初始化演示类
        """
        pass

    def add(self, a: float, b: float) -> float:
        """
        对两个数执行加法运算

        Args:
            a (float): 第一个数
            b (float): 第二个数

        Returns:
            float: 两个数的和

        Examples:
            >>> demo = DemoClass()
            >>> demo.add(2, 3)
            5
            >>> demo.add(-1, 1)
            0
        """
        logger.debug("Adding %s and %s", a, b)
        return a + b