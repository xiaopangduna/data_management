# data_management

数据管理项目。当前基于标准 Python 包结构（uv + src 布局）维护。

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
│   └── data_management/  # 主项目包
│       ├── demo_module/  # 示例模块
│       │   └── DemoClass.py  # 示例类实现
├── tests/                # 测试文件目录
├── runs/                 # 运行结果目录
├── README.md             # 项目说明文件
└── pyproject.toml        # 项目配置文件
```

## 核心组件

### DemoClass 类

- 位置：[src/data_management/demo_module/DemoClass.py](src/data_management/demo_module/DemoClass.py)
- 功能：提供加法运算功能
- 方法：add(a, b) - 执行两个数的加法运算

### 测试用例

- 位置：[tests/test_DemoClass.py](tests/test_DemoClass.py)
- 功能：对 DemoClass 类进行测试
- 测试方法：
  - test_debug_add() - 调试用简单测试
  - test_add_param() - 参数化测试，覆盖多种场景

## 安装步骤

### 环境要求

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)（推荐）

### 使用 uv 安装（推荐）

```bash
# 克隆项目
git clone <repository-url>
cd data_management

# 安装依赖并创建 .venv
uv sync

# 可选：激活虚拟环境
source .venv/bin/activate
```

未安装 uv 时，Linux / WSL 可用：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

## 常用指令

### 运行测试

不必先激活环境，直接用 `uv run`：

```bash
# 运行所有测试
uv run pytest -v

# 运行特定测试文件
uv run pytest tests/test_DemoClass.py -v

# 运行调试测试
uv run pytest tests/test_DemoClass.py::TestDemoClass::test_debug_add -v

# 运行参数化测试
uv run pytest tests/test_DemoClass.py::TestDemoClass::test_add_param -v

# 运行日志测试
uv run pytest -v --log-cli-level=DEBUG
```

### 使用DemoClass

```python
from data_management.demo_module.DemoClass import DemoClass

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

本项目使用 uv 进行依赖管理：

```bash
# 添加运行依赖
uv add package_name

# 添加开发依赖
uv add --dev package_name

# 按锁文件同步环境
uv sync

# 只更新锁文件
uv lock
```

配置文件：[pyproject.toml](pyproject.toml)