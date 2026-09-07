# data_management

用 [FiftyOne](https://docs.voxel51.com) 管理深度学习数据：图片留在原目录，元数据（路径、tags、检测框）写入本机 FiftyOne 数据库（默认 `~/.fiftyone`）。

当前已支持将 **YOLO 目录布局的 COCO** 导入为检测数据集。标注 UI、按筛选导出训练清单后续再加。

## 环境

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)

```bash
cd data_management
uv sync
# 可选
source .venv/bin/activate
```

未安装 uv 时（Linux / WSL）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

依赖见 [pyproject.toml](pyproject.toml)（含 `fiftyone`）。第一次 `import fiftyone` 可能在 `~/.fiftyone` 拉起本机 Mongo。

## YOLO 数据目录

导入脚本**只读** `--coco-root` 下的内容，不读同级 csv。需要满足：

```
<coco-root>/
  images/
    train2017/*.jpg
    val2017/*.jpg
    test2017/*.jpg      # 可无标签
  labels/
    train2017/*.txt     # YOLO 检测：class_id cx cy w h（相对坐标，中心点）
    val2017/*.txt
```

- `images/<split>/stem.jpg` 与 `labels/<split>/stem.txt` 按文件名配对。
- 无 txt 的图仍入库，只是没有检测框（如 `test2017`）。
- 忽略 `labels` 下的 `*.cache`。
- 类别为内置 COCO 80 类（`0` = `person`），与 Ultralytics 顺序一致。

本机默认根目录（可用 `--coco-root` 覆盖）：

`/home/xiaopangdun/project/deep_learning/src/train/datasets/COCO/coco`

## 导入 YOLO → FiftyOne

脚本：[scripts/import_coco_yolo.py](scripts/import_coco_yolo.py)

参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--coco-root` | 上表路径 | 含 `images/`、`labels/` 的目录 |
| `--dataset-name` | `coco2017` | FiftyOne 中的 Dataset 名 |
| `--label-types` | `detection` | 目前仅 `detection`，写入字段 `ground_truth_detect` |
| `--dry-run` | 关 | 只扫描校验，不写数据库 |

**先 dry-run**（全量扫 train/val/test，可能数分钟）：

```bash
uv run python scripts/import_coco_yolo.py \
  --coco-root /home/xiaopangdun/project/deep_learning/src/train/datasets/COCO/coco \
  --dataset-name coco2017 \
  --label-types detection \
  --dry-run
```

关注报告中的 `merge_ok=true`、`parse_errors=0`。`unlabeled_images` 在 train/val 上少量存在是正常的（部分图没有 thing 实例）；test 无标签也正常。

启动时若出现 `glob2` 的 `SyntaxWarning`（`invalid escape sequence '\Z'`），可忽略，来自 FiftyOne 依赖，不是导入失败。

确认后再正式入库（Dataset 已存在会拒绝写入，需换名或先删库）：

```bash
uv run python scripts/import_coco_yolo.py \
  --coco-root /home/xiaopangdun/project/deep_learning/src/train/datasets/COCO/coco \
  --dataset-name coco2017 \
  --label-types detection
```

成功时会打印 `imported_dataset=coco2017 samples=...`。图片**不会被拷贝**，库里只存绝对路径。

写入的字段：

- `filepath`：原图路径
- `tags`：`coco` + split 文件夹名（`train2017` / `val2017` / `test2017`）
- `ground_truth_detect`：有框才有；每框为类名 `label` + 相对框 `bounding_box`（左上角 xywh）

## FiftyOne 使用建议

App 用来**看图、筛数据、打工作流 tag**，不是画框工具。

```bash
uv run fiftyone app launch coco2017
```

浏览器打开后（默认本机端口，以终端提示为准）：

1. **先筛 val**  
   左侧 tags 只勾 `val2017`（约 5k）。不要一上来勾满 `train2017`（11 万+），网格会很卡。

2. **核对检测**  
   点开一张 val 图：框应套在物体上，类名是 `person` 等，不是数字 id。  
   `test2017` 多数没有 `ground_truth_detect`，无框是正常的。

3. **按类别筛**  
   边栏 `ground_truth_detect` 勾选类别（如只要 `person`）。这只改变当前视图，不改磁盘上的 YOLO txt。

4. **tags 含义**  
   - 导入时已有：`coco`（来源）、`train2017` / `val2017` / `test2017`（划分）  
   - 可在 App 里给选中样本再加 `hard`、`discard` 等，供以后导出筛选  
   - 类别名在检测字段里，不要和 tags 混淆

5. **Patches**  
   可按每个检测框切小图，适合检查某类标得密不密、有没有明显错框。

6. **不要在 App 里找的功能**  
   选择本地 COCO 文件夹导入、导出 YOLO 训练目录、精细改框：导入已由脚本完成；导出与 CVAT/X-AnyLabeling 标注尚未接入。

Python 里加载同一份库：

```python
import fiftyone as fo

dataset = fo.load_dataset("coco2017")
val = dataset.match_tags("val2017")
session = fo.launch_app(val)
```

## 项目结构

```
.
├── scripts/
│   └── import_coco_yolo.py   # YOLO 布局 COCO → FiftyOne
├── src/data_management/      # 包代码（示例模块仍保留）
├── tests/
├── notebooks/
├── README.md
└── pyproject.toml
```

## 测试与示例模块

```bash
uv run pytest -v
```

`DemoClass` 仅作测试示例，与数据入库无关。

## 依赖管理

```bash
uv add package_name
uv add --dev package_name
uv sync
uv lock
```
