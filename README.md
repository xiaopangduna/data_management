# data_management

用 [FiftyOne](https://docs.voxel51.com) 管理深度学习数据：图片留在原目录，元数据（路径、tags、检测框）写入本机 FiftyOne 数据库（默认 `~/.fiftyone`）。

当前已支持将 **YOLO 目录布局的 COCO** 导入为检测数据集，以及把筛出的样本导出给 **X-AnyLabeling** 重标后再写回 FiftyOne。

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

## 用法

图片留在原目录，FiftyOne 只存路径和元数据。无标注图用 `import_images.py` 入库，再用 `enrich_fiftyone_media.py` 补哈希和尺寸，然后用 `dedup_fiftyone.py` 去重，用 `attach_yolo_labels.py` 挂 YOLO 检测框。需要重标时用 `export_xlabel.py` 导出软链和 JSON，在 X-AnyLabeling 中改完后用 `attach_xlabel_labels.py` 写回。导出 CSV 后再用软链接生成训练用 YOLO 目录。带 YOLO 标签的 COCO 见下文「导入 YOLO → FiftyOne」。

### Step：导入图片

脚本：[scripts/import_images.py](scripts/import_images.py)

从本地目录创建**持久化**图片数据集。只扫描图片、写入路径和 tags，不拷贝文件，也不推断类别或 train/val 划分。同名数据集已存在时拒绝写入（不覆盖、不追加）。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--images-root` | 必填 | 图片根目录，支持相对路径及 `~` |
| `--dataset-name` | 必填 | 新数据集名称 |
| `--tags` | 无 | 统一添加的标签，逗号分隔 |
| `--recursive` / `--no-recursive` | 递归 | 是否扫描子目录；不遍历目录符号链接 |
| `--extensions` | `.jpg,.jpeg,.png,.webp,.bmp` | 扩展名，忽略大小写，可省略点 |
| `--verify-images` | 关闭 | 解码校验图片；跳过坏图并记录路径 |
| `--batch-size` | `1000` | 每批写入数量，必须大于零 |
| `--dry-run` | 关闭 | 只扫描，不连接数据库；不检查数据集是否同名 |

**先 dry-run：**

```bash
uv run python scripts/import_images.py \
  --images-root /data/baby_monitor/images \
  --dataset-name baby_monitor_raw \
  --tags baby_monitor,raw \
  --dry-run
```

关注报告中的 `valid=`。目录不存在或没有有效图片时不会创建数据集。默认只按扩展名扫描，不解码；需要检查坏图时加 `--verify-images`（dry-run 同样生效）。

确认后再正式导入：

```bash
uv run python scripts/import_images.py \
  --images-root /data/baby_monitor/images \
  --dataset-name baby_monitor_raw \
  --tags baby_monitor,raw
```

成功时打印 `imported_dataset=... samples=...`。每张图写入：

- `filepath`：解析符号链接后的绝对路径
- `relpath`：相对 `--images-root` 的 POSIX 路径
- `tags`：`--tags` 中的标签

相同规范化绝对路径只导入一次；不同目录中的同名文件会保留。多个符号链接指向同一文件时，保留首次遇到的相对路径。扫描结束会报告 `scanned` / `valid` / `duplicates` / `invalid` / `skipped`。正式写入中途失败会返回非零状态，**保留部分数据集**并报告已写入数量；重试需换新名称或自行处理该数据集。

### Step：完善基础信息

脚本：[scripts/enrich_fiftyone_media.py](scripts/enrich_fiftyone_media.py)

对**已存在**的数据集补齐媒体字段，不改 `filepath`、检测框、tags，也不创建或删除数据集。目标数据集不存在则退出。

先列出本机已有数据集，确认 `--dataset-name` 再跑补全：

```bash
uv run fiftyone datasets list
```

输出即为可传给 `--dataset-name` 的名称（例如导入后的 `baby_monitor_raw`）。没有对应名称时先回到上一步导入，不要凭空指定。

默认写入：

| 字段 | 说明 |
|---|---|
| `sha256` | 文件字节 SHA-256（小写 hex），用于内容去重/校验 |
| `phash` | 64-bit DCT 感知哈希（16 位小写 hex），用于找视觉相近图 |
| `phash_algorithm` | 固定 `phash-dct-64-v1`，标识算法版本 |
| `metadata` | 宽、高、文件大小、MIME 等 |

磁盘上找不到的文件会跳过并记 warning。默认增量补全：字段已有值则跳过。`phash` 例外：没有 `phash_algorithm=phash-dct-64-v1` 的旧值会强制重算。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset-name` | `coco2017` | 已有 FiftyOne 数据集名 |
| `--hashes` | `sha256,phash` | 逗号分隔：`sha256`、`phash`，或 `none` |
| `--metadata` / `--no-metadata` | 计算 | 是否写 FiftyOne metadata |
| `--overwrite` | 关闭 | 已有值也重算 |
| `--dry-run` | 关闭 | 只统计将要写入的条数，不写库 |

**先 dry-run：**

```bash
uv run python scripts/enrich_fiftyone_media.py \
  --dataset-name baby_monitor_raw \
  --dry-run
```

关注 `sha256_to_write` / `phash_to_write` / `metadata_to_write` / `missing_files`。确认后再正式补全：

```bash
uv run python scripts/enrich_fiftyone_media.py --dataset-name baby_monitor_raw
```

成功时打印 `enrich_done=true`。之后可打开 App：

```bash
uv run fiftyone app launch baby_monitor_raw
```

只算尺寸、不算哈希：`--hashes none`。只算哈希、不算尺寸：`--no-metadata`。

### Step：去重

脚本：[scripts/dedup_fiftyone.py](scripts/dedup_fiftyone.py)

对**已存在**的数据集做内容去重，需先跑完「完善基础信息」（至少有 `sha256` / `phash`）。不创建或删除数据集，**不删磁盘文件**。

两类重复分开处理：

| 类型 | 判定 | 动作 |
|---|---|---|
| 精确重复 | `sha256` 相同，组内 ≥ 2 | 每组留 1 张，其余从数据集删除；写入记录 CSV |
| 相似 | 64-bit `phash` 的 Hamming 距离 ≤ `--hamming-max`，组内 ≥ 2 | 留下的那张**不**打 `dup_near`；其余打 `dup_near`；整组写 `dup_group`（留下那张的 phash）和 `dup_of`（多余张指向留下那张的 `filepath`） |

留下哪一张：`relpath` 字典序最小（无则用 `filepath`），其次分辨率更大，再其次 sample id 更小。每次都先删精确重复，再在剩余样本上做相似分组。已有 `dup_near` / `dup_group` 的样本不再改标记。

**相似组与 `--hamming-max`：** 按 pHash 的 Hamming 距离做连通分量聚类。默认 **`2`**（64-bit 里最多差 2 bit）。实现上按「鸽笼分块」建索引，只比较共享同一 bit 块的哈希，27 万张不必做全量两两比较。若标记明显偏少可再加大；偏多则降到 `0`（仅 phash 完全相同）。

先列出数据集：

```bash
uv run fiftyone datasets list
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset-name` | 必填 | 已有 FiftyOne 数据集名 |
| `--hamming-max` | `2` | 相似组最大 Hamming 距离；`0` = phash 完全相同 |
| `--dry-run` | 关闭 | 只统计并写 CSV，不删、不打标 |

缺 `sha256` / `phash` 的样本跳过并计入报告。精确删除不受已有 tag 保护。

**先 dry-run：**

```bash
uv run python scripts/dedup_fiftyone.py \
  --dataset-name baby_monitor_raw \
  --dry-run
```

关注 `exact_groups` / `exact_to_delete` / `near_groups` / `near_to_tag` / `csv_dir`。每次运行（包括 dry-run）都会在**当前工作目录的 `tmp/`** 下写 CSV（`tmp/` 已在 .gitignore 中）：

| 文件 | 内容 |
|---|---|
| `tmp/dedup_<数据集>.csv` | 全量归档（精确 + 相似，全部列） |
| `tmp/dedup_<数据集>_exact.csv` | 仅精确组；没有精确重复时不写 |
| `tmp/dedup_<数据集>_near_01.csv` … | 相似组瘦表，每文件最多 200 个 `dup_group`，列只有 `kind,action,relpath,kept_relpath,dup_group` |

重跑会覆盖同名文件，并删掉旧的 `near_*.csv` 再按当前组数重写。表里包含每组**每一张**（含留下的那张）：

| 列 | 说明 |
|---|---|
| `kind` | `exact` 或 `near` |
| `action` | `keep` / `delete`（精确多余张）/ `tag_dup_near`（相似多余张） |
| `dup_group` | 精确组为 sha256；相似组为留下那张的 phash |
| `sample_id` / `filepath` / `relpath` | 当前这张 |
| `sha256` / `phash` | 哈希 |
| `kept_sample_id` / `kept_filepath` / `kept_relpath` | 该组留下的那张 |

确认后再正式去重（会再算一遍并改库）：

```bash
uv run python scripts/dedup_fiftyone.py --dataset-name baby_monitor_raw
```

成功时打印 `dedup_done=true`。库记录删了可以靠这份 CSV 核对；原图仍在磁盘上。

相似图在 App 里复核（只勾 `dup_near`，不要用别的 tag 做全选删除）：

```bash
uv run fiftyone app launch baby_monitor_raw
```

- 只勾 `dup_near`：看到的都是多余张，Select All 也删不到留下的那张
- 看整组：点开一张候选，按 `dup_group` 筛选（含未打 tag 的主图）；看完清掉字段筛选
- 其实该留：去掉 `dup_near`；确实多余：在网页删除 sample，或之后 `dataset.match_tags("dup_near").delete()`

### Step：更新标签

脚本：[scripts/attach_yolo_labels.py](scripts/attach_yolo_labels.py)

给**已存在**的数据集挂 YOLO 检测框。按 `relpath` 去掉扩展名后与 `labels/<stem>.txt` 配对（子目录则整段相对路径配对）。不改 filepath、哈希、tags，也不覆盖库里已有的框。冲突写入 `tmp/attach_labels_<数据集>.csv`，不中断整次任务。

`--class-names` 为逗号分隔的**有序列表**，下标即 YOLO `class_id`（`baby_head` 表示 `0`）。框写入字段 `ground_truth_detect`（FiftyOne `Detections`，左上角相对 xywh），并用 `label_relpath` 记下对应 txt。不要把类别写进 sample tags。

忽略 `*.cache`。无 txt 的图计为 `unlabeled`，不算错误。有 txt 但库里没有对应图（例如精确去重已删的 sample）记为 `orphan_label`。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset-name` | 必填 | 已有 FiftyOne 数据集名 |
| `--labels-dir` | 必填 | YOLO txt 目录 |
| `--class-names` | 必填 | 类别名列表，顺序 = class_id |
| `--dry-run` | 关闭 | 只解析并写问题 CSV，不改库 |

问题 CSV 的 `issue`：

| issue | 含义 |
|---|---|
| `stem_collision` | 多个 sample 同一文件名，无法唯一对应一份 txt |
| `class_id_out_of_range` | txt 的 id 不在 `--class-names` 里 |
| `parse_error` | 行不是 5 个数、或框宽高非法 |
| `box_mismatch` | 库里已有框且与 txt 不一致，**不覆盖** |
| `orphan_label` | 有 txt，库里没有对应图 |
| `empty_label` | txt 为空或没有有效框 |

**先 dry-run：**

```bash
uv run python scripts/attach_yolo_labels.py \
  --dataset-name BBM08S_head \
  --labels-dir /mnt/nvme_data/data/head_train_data/head_train_26w_val_0.3w/train/labels \
  --class-names baby_head \
  --dry-run
```

关注 `matched` / `unlabeled` / `to_write` / `issues` / `csv_path`。确认后再正式写入：

```bash
uv run python scripts/attach_yolo_labels.py \
  --dataset-name BBM08S_head \
  --labels-dir /mnt/nvme_data/data/head_train_data/head_train_26w_val_0.3w/train/labels \
  --class-names baby_head
```

成功时打印 `attach_done=true`。在 App 里点开有框的图，类名应为 `baby_head` 而不是 `0`。`dup_near` 的图同样会挂框。

### Step：导出给 X-AnyLabeling 重标

脚本：[scripts/export_xlabel.py](scripts/export_xlabel.py)

从已有数据集筛出要重标的样本，写到**独立任务目录**（不要写回原图目录）：每张图一个软链，旁边一份 X-AnyLabeling JSON（已有 `ground_truth_detect` 会转成 rectangle）。不拷贝原图，不改 FiftyOne。先在 App 里给要导出的图打 tag（例如 `relabel`），再用 `--include-tags` 选出它们。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset-name` | 必填 | 已有 FiftyOne 数据集名 |
| `--out-dir` | 必填 | 任务目录 |
| `--include-tags` | 必填 | 带其中任一 tag 的样本才导出 |
| `--exclude-tags` | `dup_near` | 排除这些 tag；`none` 表示不排除 |
| `--class-names` | 无 | 检测框类名白名单：只有这些 `label` 写入 JSON；未指定则全部导出 |
| `--dry-run` | 关闭 | 只规划并写问题 CSV，不建目录 |

**先 dry-run：**

```bash
uv run python scripts/export_xlabel.py \
  --dataset-name BBM08S_head \
  --out-dir /mnt/nvme_data/data/relabel_BBM08S_head \
  --include-tags relabel \
  --class-names baby_head \
  --dry-run
```

关注 `to_write` / `issues` / `csv_path`。确认后再去掉 `--dry-run`。目录结构：

```
<out-dir>/
  manifest.csv
  <relpath>.jpg     # 软链 → 原 filepath
  <relpath>.json    # XLABEL，含 sample_id / fo_sample_id=
```

在 X-AnyLabeling 中打开 `<out-dir>`（或其中有图的子目录），**不要**打开 Save Image Data。类名须与 `--class-names` 一致。JSON 与图在同一层，一般不必再改 output 目录。

问题 CSV 在 `tmp/export_xlabel_<数据集>.csv`。

### Step：把 X-AnyLabeling 结果写回 FiftyOne

脚本：[scripts/attach_xlabel_labels.py](scripts/attach_xlabel_labels.py)

扫描 `--label-dir` 下的 `*.json`（可含子目录，不跟随目录符号链接），用 JSON 里的 `sample_id`（或 `description` 中的 `fo_sample_id=`）对上 sample。**只替换 `--class-names` 中的类**：删掉库里这些类的旧框，再写入 JSON 里的框；其它类（如 `car`）不动。JSON 里没有这类框就清空这类。目录里没有 JSON 的图一律不动。不改 filepath、哈希。

`--tags` 打在本批处理过的所有样本上。框确实改过的再加 tag `changed`；框没变则不加（重跑时会去掉已有的 `changed`）。App 里：勾批次 tag 看整批；再勾 `changed` 就是这批改过的；只勾批次、不勾 `changed` 就是这批没改的。清批次时删掉该 `--tags` 即可，`changed` 是共用名。

polygon 会先变成轴对齐外接框；`rotation` 等其它类型整份 JSON 拒绝写入。不在 `--class-names` 里的 shape 会忽略，不阻断该文件。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset-name` | 必填 | 已有 FiftyOne 数据集名 |
| `--label-dir` | 必填 | JSON 所在目录，如 `tmp/images/val2017` |
| `--class-names` | 必填 | 要替换的检测类名 |
| `--tags` | 必填 | 本批 tag；改过框的样本另加 `changed` |
| `--dry-run` | 关闭 | 只解析并写问题 CSV |

```bash
uv run python scripts/attach_xlabel_labels.py \
  --dataset-name coco2017 \
  --label-dir tmp/images/val2017 \
  --class-names person \
  --tags label_person_260909 \
  --dry-run
```

关注 `matched` / `to_write` / `unchanged` / `issues`。确认后去掉 `--dry-run`。问题 CSV：`tmp/attach_xlabel_<数据集>.csv`。

| issue | 含义 |
|---|---|
| `missing_sample_id` | JSON 里没有可解析的 sample id |
| `orphan_label` | id 在库里不存在 |
| `sample_id_collision` | 两个 JSON 指向同一个 sample |
| `unsupported_shape` | 非 rectangle/polygon |
| `parse_error` / `missing_size` | JSON 损坏或没有宽高 |

写回后若要训练，再跑「导出图片和标签的 csv」和 `csv_to_yolo.py`。

### Step：导出图片和标签的csv

脚本：[scripts/export_training_csv.py](scripts/export_training_csv.py)

从 FiftyOne 当前库导出训练清单，**不拷贝图片、不写 YOLO txt**。默认只要有 `ground_truth_detect` 的样本，并排除 tag `dup_near`。精确去重已从库删除的图不会出现。

写出（均在 `tmp/`）：

| 文件 | 内容 |
|---|---|
| `export_<数据集>.csv` | **总表**，一行一张图；`csv_to_yolo.py` 只读这份 |
| `export_<数据集>_part_01.csv` … | 与总表相同列，每 5000 张一份，方便 Excel 打开 |
| `export_<数据集>_issues.csv` | 仅当确有跳过项时才写 |

总表列：`sample_id,filepath,relpath,tags,box_count,labels`。`labels` 为该图全部 YOLO 行（`class_id cx cy w h`），多框用 `;` 连接。`--class-names` 必须与挂框时一致。`--exclude-tags none` 可把 `dup_near` 也导出。重跑会覆盖总表并重建分片，同时删掉旧的 `_images.csv` / `_boxes.csv`。

```bash
uv run python scripts/export_training_csv.py \
  --dataset-name BBM08S_head \
  --class-names baby_head
```

关注 `images` / `boxes` / `issues` / `csv_path` / `part_files`。无框、未知类名、`relpath` 冲突的图不会进总表。

### Step：根据CSV生成数据集yolo格式

脚本：[scripts/csv_to_yolo.py](scripts/csv_to_yolo.py)

只读**总表**，不读 part 分片，**不连接 FiftyOne**。在 `--out-dir` 下生成：

```
<out-dir>/
  images/train/<relpath>   # 软链接 → filepath（不复制原图）
  labels/train/<stem>.txt  # 由 labels 列写出
  data.yaml                # names 与 --class-names 一致；暂无独立 val，val 指向 train
```

```bash
uv run python scripts/csv_to_yolo.py \
  --csv tmp/export_BBM08S_head.csv \
  --out-dir /mnt/nvme_data/data/head_train_data/BBM08S_head_yolo \
  --class-names baby_head \
  --dry-run
```

确认 `images` / `missing_file` 后去掉 `--dry-run` 正式建链。训练请把 `data.yaml` 的 `path` 指到这个新目录，不要再用原来的 `head_train_26w_val_0.3w`。App 里改过 tag 或删过图之后，应重新导出 CSV 再生成，不要手工改生成目录。

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
   选择本地 COCO 文件夹导入、导出 YOLO 训练目录、精细改框：导入与 YOLO 导出已由脚本完成；改框请用 X-AnyLabeling（见上文导出 / 写回两步），不要在 App 里当画框工具用。

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
│   ├── import_images.py           # 无标注图片目录 → FiftyOne
│   ├── enrich_fiftyone_media.py   # 补哈希与图片 metadata
│   ├── dedup_fiftyone.py          # 精确删除 + 相似打 dup_near
│   ├── attach_yolo_labels.py      # 按文件名把 YOLO txt 挂到已有库
│   ├── export_xlabel.py           # 筛库 → 软链 + X-AnyLabeling JSON
│   ├── attach_xlabel_labels.py    # JSON 目录 → 覆盖写回检测框并打批次 tag
│   ├── export_training_csv.py     # 筛库 → 训练总表 CSV + 分片
│   ├── csv_to_yolo.py             # CSV → 软链接 YOLO 目录
│   └── import_coco_yolo.py        # YOLO 布局 COCO → FiftyOne
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
