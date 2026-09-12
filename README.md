# data_management

用 [FiftyOne](https://docs.voxel51.com) 管深度学习数据：图片留在原目录，路径 / tags / 检测框写入本机数据库（默认 `~/.fiftyone`）。

```bash
cd data_management
uv sync   # Python >= 3.12
uv run fiftyone datasets list
uv run fiftyone app launch <dataset>

cd /tmp
curl -fL -o mongodb-database-tools.deb \
  https://fastdl.mongodb.org/tools/db/mongodb-database-tools-ubuntu2404-x86_64-100.18.0.deb
sudo apt install -y ./mongodb-database-tools.deb
mongodump --version

.venv/lib/python3.12/site-packages/fiftyone/db/bin/mongod --dbpath /home/lee/huangwenhua/.fiftyone/mongo --port 27018 --fork --logpath /home/lee/huangwenhua/.fiftyone/mongod.log
pgrep -af 'mongod.*27018'
export FIFTYONE_DATABASE_URI=mongodb://127.0.0.1:27018
# 备份
STAMP=$(date +%Y%m%d)
OUT=/mnt/nvme_data/backup/fiftyone/fiftyone_${STAMP}.archive.gz
mkdir -p /mnt/nvme_data/backup/fiftyone
mongodump --uri="$FIFTYONE_DATABASE_URI" --db fiftyone --gzip --archive="$OUT"
ls -lh "$OUT"

# 恢复（会覆盖当前库）
# mongorestore --uri="$FIFTYONE_DATABASE_URI" --gzip \
#   --archive=/mnt/nvme_data/backup/fiftyone/fiftyone_20260910.archive.gz --drop

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

不删磁盘文件。默认只打标，不从库里删样本。

完全重复（同 `sha256`）：全员 `dup_repeat`，当前保留张 `dup_repeat_keep`，建议删除 `dup_repeat_drop`。近似重复（pHash Hamming ≤ 2）：全员 `dup_near`（含当前保留张）。`dup_repeat_drop` 不参与近重复聚类。分组字段：`dup_group` / `dup_of`。

```bash
uv run python scripts/dedup_fiftyone.py --dataset-name BBM08S_head --dry-run
uv run python scripts/dedup_fiftyone.py --dataset-name BBM08S_head
```

App 里用 `dup_repeat` / `dup_repeat_drop` / `dup_near` 复核，按 `dup_group` 分组。CSV：`tmp/dedup_<数据集>*.csv`。导出默认排除 `dup_repeat_drop`。

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

按 tag **并集**筛样本，写出一份目录。默认排除 `dup_repeat_drop`。`--output-dir` 须为空。

```bash
uv run python scripts/export_yolo.py \
  --dataset BBM08S_head \
  --output-dir /path/to/BBM08S_head_yolo \
  --tags train \
  --classes baby_head,adult_head
```

默认软链原图。无框图保留为空 txt。`--label-field` 默认 `ground_truth`。

图片和标注采用相同名称主体，例如 `baby_head__adult_head_000001.jpg/.txt`。
前缀取实际导出标注的类别，去重后按导出类别 ID 顺序排列（未指定 `--classes` 时按类别名排序），最多 6 个类别，超出追加 `__more`；无框图使用 `negative`。类别名中的不安全字符替换为下划线，过长名称截短。后缀为本次导出从 1 开始的全局序号，至少 6 位，原图扩展名保留。不同批次的序号可能重复，不保证跨批次文件名唯一。

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

原图入库前可按文件内容的完整 SHA-256 统一命名（默认只处理当前层）：

```bash
uv run python scripts/rename_images_by_hash.py /path/to/images --dry-run
uv run python scripts/rename_images_by_hash.py /path/to/images
```

需要处理各级子目录时增加 `--recursive`。扩展名会转为小写；相同内容、相同扩展名的重复图片以 `-2`、`-3` 保留。脚本不会修改标注文件或 FiftyOne 中已有的文件路径，因此应在入库前运行。

## 按图片内容追加 tags

指定目录中的图片按文件内容 SHA-256 匹配已有数据集，只追加 sample tags，保留原标签并去重。文件名和路径不参与匹配，不导入新样本。先用 `update_media.py` 补齐库中的 `sha256`。

```bash
uv run python scripts/update_tags.py \
  --dataset-name BBM08S_head \
  --images-dir /path/to/selected_images \
  --tags relabel,review \
  --dry-run
```

默认只扫描当前层；包含子目录时加 `--recursive`。确认统计后去掉 `--dry-run` 正式写库。相同内容的多份输入图片只更新对应样本一次；同一哈希对应多个库内样本时跳过。未匹配图片、读取失败、库内缺少哈希及多样本匹配记录到 `tmp/update_tags_<数据集>.csv`（每次运行覆盖，预览也会生成）。整库缺少 `sha256` 字段时直接报错，提示先补哈希。
