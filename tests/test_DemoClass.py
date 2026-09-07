"""
DemoClass 的标准pytest测试用例
严格按照测试规范编写，使用类组织测试
"""

import pytest
from project_name.demo_module.DemoClass import DemoClass


class TestDemoClass:
    """DemoClass测试类"""
    
    # ---- 1. 仅用于调试的简单测试 ----
    def test_debug_add(self):
        """用于调试的简单测试"""
        demo = DemoClass()
        assert demo.add(2, 3) == 5
    
    # ---- 2. 参数化测试（覆盖多种情况）----
    @pytest.mark.parametrize("a,b,expected", [
        # 基本运算
        (2, 3, 5),
        (-2, -3, -5),
        (-2, 3, 1),
        
        # 零值测试
        (5, 0, 5),
        (0, 0, 0),
        
        # 浮点数测试
        (2.5, 3.7, 6.2),
        (2.5, 2.5, 5.0),
        
        # 大数测试
        (1000000, 2000000, 3000000),
        
        # 边界条件
        (10, -5, 5),
        (-10, -20, -30),
    ])
    def test_add_param(self, a, b, expected):
        """参数化测试DemoClass的add方法"""
        demo = DemoClass()
        
        # 对于浮点数使用近似比较
        if isinstance(expected, float):
            assert abs(demo.add(a, b) - expected) < 0.0001
        else:
            assert demo.add(a, b) == expected