# data_management

用 [FiftyOne](https://docs.voxel51.com) 管深度学习数据：图片留在原目录，路径 / tags / 检测框写入本机数据库（默认 `~/.fiftyone`）。

```bash
cd data_management
uv sync   # Python >= 3.12
uv run fiftyone datasets list
uv run fiftyone app launch <dataset>
```

各脚本都支持 `--dry-run`：先看统计，确认后再去掉该参数正式写库。问题 CSV 在 `tmp/`。

## 流程

```
导入 → 补哈希/尺寸 → 去重 → 导出标注 → 写回标注 → 导出 YOLO 训练
```

### Step 1：导入 FiftyOne

库不存在则创建；已有 `filepath` 跳过。不拷贝图片。

**YOLO 叶子目录**（`images/train` + 可选 `labels/train`，不递归、不读 `dataset.yaml`）：

```bash
uv run python scripts/import_yolo.py \
  --dataset-name BBM08S_head \
  --images-dir /path/to/images/train \
  --labels-dir /path/to/labels/train \
  --class-names baby_head,adult_head \
  --tags head,train
```

**仅图片**（无标注）：不传 `--labels-dir` / `--class-names`。

**YOLO 布局的 COCO**（`images/<split>/` + `labels/<split>/`，固定 COCO 80 类）：

```bash
uv run python scripts/import_coco_yolo.py \
  --coco-root /path/to/coco \
  --dataset-name coco2017
```

整库重导加 `--replace`。多个 YOLO split 各跑一次 `import_yolo.py`。

### Step 2：补哈希与尺寸

```bash
uv run python scripts/update_media.py --dataset-name BBM08S_head
```

写入 `sha256`、`phash`、`metadata`。不改路径、框、tags。增量跳过已有值。

### Step 3：去重

不删磁盘文件。精确重复（同 `sha256`）从库里删多余张；相似图（pHash Hamming ≤ 2）打 `dup_near`。

```bash
uv run python scripts/dedup_fiftyone.py --dataset-name BBM08S_head
```

在 App 里只勾 `dup_near` 复核。CSV：`tmp/dedup_<数据集>*.csv`。

### Step 4：导出给 X-AnyLabeling

先在 App 给要重标的图打 tag（如 `relabel`）。导出到**独立目录**，不要写回原图目录。

```bash
uv run python scripts/export_xlabel.py \
  --dataset-name BBM08S_head \
  --out-dir /path/to/relabel_task \
  --sample-tags relabel \
  --label-field ground_truth
```

默认复制图片；本机可加 `--export-media symlink`。在 X-AnyLabeling 打开该目录，不要开 Save Image Data。

不经过 FiftyOne、只把 YOLO txt 转成 X-AnyLabeling JSON：`scripts/convert_yolo_to_xlabel.py`。

### Step 5：写回标注

只替换 `--class-names` 里的类；其它类不动。变更样本会打 `--tags` 和 `changed`。

```bash
uv run python scripts/update_xlabel_labels.py \
  --dataset-name BBM08S_head \
  --label-dir /path/to/relabel_task \
  --class-names baby_head,adult_head \
  --tags label_import_260911
```

无导出 `sample_id` 时加 `--images-dir` 按原图绝对路径匹配。已有库要补 YOLO txt：先 `convert_yolo_to_xlabel.py`，再走本脚本写回 `ground_truth`。

### Step 6：导出 YOLO 训练

按 tag **并集**筛样本，写出一份目录。`--output-dir` 须为空。

```bash
uv run python scripts/export_yolo.py \
  --dataset BBM08S_head \
  --output-dir /path/to/BBM08S_head_yolo \
  --tags train \
  --classes baby_head,adult_head
```

默认软链原图。无框图保留为空 txt。`--label-field` 默认 `ground_truth`。

## 约定

- 检测框字段默认 `ground_truth`。
- 类别在检测字段里，不要写进 sample tags。
- App 用来看图、筛数据、打工作流 tag，不在里面画框。

脚本按动词前缀：`import_` 入库，`update_` 改已有库，`export_` 导出，`convert_` 只转文件，`dedup_` 去重。

```
scripts/
  import_yolo.py / import_coco_yolo.py
  update_media.py / update_xlabel_labels.py
  dedup_fiftyone.py
  export_xlabel.py / export_yolo.py
  convert_yolo_to_xlabel.py
```

```bash
uv run pytest -v
```
