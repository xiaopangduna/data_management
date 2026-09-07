# skeleton_python

一个Python项目骨架示例，展示了标准的项目结构和测试实践。

## 项目概述

本项目是一个标准的Python项目骨架结构，旨在演示如何组织一个具有良好工程实践的Python项目。项目包含了标准的目录结构、测试用例、配置文件等常见项目组件。

## 项目结构

```
.
├── configs/              # 配置文件目录
├── datasets/             # 数据集目录
├── docs/                 # 文档目录
├── models/               # 模型文件目录
├── notebooks/            # Jupyter Notebooks目录
├── scripts/              # 脚本文件目录
├── src/                  # 源代码目录
│   └── project_name/     # 主项目包
│       ├── demo_module/  # 示例模块
│       │   └── DemoClass.py  # 示例类实现
├── tests/                # 测试文件目录
├── runs/                 # 运行结果目录
├── README.md             # 项目说明文件
└── pyproject.toml        # 项目配置文件
```

## 核心组件

### DemoClass 类

- 位置：[src/project_name/demo_module/DemoClass.py](file:///home/ubuntu/Desktop/project/skeleton_python/src/project_name/demo_module/DemoClass.py)
- 功能：提供加法运算功能
- 方法：add(a, b) - 执行两个数的加法运算

### 测试用例

- 位置：[tests/test_DemoClass.py](file:///home/ubuntu/Desktop/project/skeleton_python/tests/test_DemoClass.py)
- 功能：对 DemoClass 类进行测试
- 测试方法：
  - test_debug_add() - 调试用简单测试
  - test_add_param() - 参数化测试，覆盖多种场景

## 安装步骤

### 环境要求

- Python >= 3.12
- Poetry (推荐) 或 pip

### 使用 Poetry 安装（推荐）

```bash
# 克隆项目
git clone <repository-url>
cd skeleton_python

# 安装依赖
poetry install

# 激活虚拟环境
poetry shell
```

### 使用 pip 安装

```bash
# 克隆项目
git clone <repository-url>
cd skeleton_python

# 创建虚拟环境（可选）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .
```

## 常用指令

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_DemoClass.py -v

# 运行调试测试
pytest tests/test_DemoClass.py::TestDemoClass::test_debug_add -v

# 运行参数化测试
pytest tests/test_DemoClass.py::TestDemoClass::test_add_param -v

# 运行日志测试
pytest -v --log-cli-level=DEBUG
```

### 使用DemoClass

```python
from project_name.demo_module.DemoClass import DemoClass

demo = DemoClass()
result = demo.add(2, 3)
print(result)  # 输出: 5
```

## 开发规范

### 测试规范

- 使用 pytest 作为测试框架
- 采用参数化测试覆盖多种场景
- 遵循"两个测试方法"原则：一个用于调试，一个用于全面测试

### 代码规范

- 遵循 PEP8 编码规范
- 类名使用 PascalCase 命名
- 函数/方法使用 snake_case 命名
- 变量使用 snake_case 命名

## 依赖管理

本项目使用 Poetry 进行依赖管理：

```bash
# 添加依赖
poetry add package_name

# 添加开发依赖
poetry add --group dev package_name

# 更新依赖
poetry update

# 导出 requirements.txt
poetry export -f requirements.txt --output requirements.txt
```

配置文件：[pyproject.toml](file:///home/ubuntu/Desktop/project/skeleton_python/pyproject.toml)