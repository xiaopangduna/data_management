"""Shared YOLO leaf import into FiftyOne.

Scan one image directory (and optional YOLO txt directory), skip existing
filepaths, and append new samples. COCO-style trees are a loop over splits.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BATCH_SIZE = 1000
REPORT_SUBDIR = "tmp"
ISSUE_COLUMNS = ("issue", "filepath", "filename", "label_filepath", "detail")
SCAN_LOG_INTERVAL = 50000
PARSE_LOG_INTERVAL = 20000
PARSE_CHUNK_SIZE = 4096
PARSE_WORKERS = min(32, max(8, (os.cpu_count() or 4) * 2))
RELPATH_FIELD = "relpath"
SOURCE_TAG_DEFAULT = "coco"

COCO_80_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


@dataclass
class PendingSample:
    filepath: str
    filename: str
    boxes: list[tuple[str, list[float]]] | None
    label_filepath: str | None
    tags: list[str]
    relpath: str | None = None


@dataclass
class LeafPrepareResult:
    items_count: int = 0
    skipped: int = 0
    unlabeled: int = 0
    parse_errors: int = 0
    issues: list[dict[str, str]] = field(default_factory=list)
    pending: list[PendingSample] = field(default_factory=list)
    new_filepaths: list[str] = field(default_factory=list)


def nonempty(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("must not be empty")
    return stripped


def class_names(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("provide at least one class name")
    return names


def tag_list(value: str) -> list[str]:
    tags = [part.strip() for part in value.split(",") if part.strip()]
    if not tags:
        raise argparse.ArgumentTypeError("provide at least one tag")
    return list(dict.fromkeys(tags))


def _suffix(name: str) -> str:
    return os.path.splitext(name)[1].lower()


def _stem(name: str) -> str:
    return os.path.splitext(name)[0]


def _dirent_filepath(entry: os.DirEntry[str]) -> str:
    """Absolute path; follow symlinks. Regular files skip a realpath syscall."""
    if entry.is_symlink():
        filepath = os.path.realpath(entry.path)
        if not os.path.exists(filepath):
            raise FileNotFoundError(entry.path)
        return filepath
    return entry.path


def list_leaf_files(directory: Path, suffixes: set[str]) -> tuple[list[Path], int]:
    """List files in one directory. Does not recurse. Returns files and subdir count."""
    names: list[str] = []
    subdirs = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                subdirs += 1
                continue
            if _suffix(entry.name) in suffixes:
                names.append(entry.name)
    names.sort()
    return [directory / name for name in names], subdirs


def scan_images(images_dir: Path) -> tuple[list[tuple[str, str, str]], int]:
    """Return (resolved filepath, filename, stem) and ignored subdirectory count."""
    logger.info("Scanning images in %s", images_dir)
    items: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    subdirs = 0
    with os.scandir(images_dir) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                subdirs += 1
                continue
            if _suffix(entry.name) not in IMAGE_SUFFIXES:
                continue
            filepath = _dirent_filepath(entry)
            if filepath in seen:
                continue
            seen.add(filepath)
            items.append((filepath, entry.name, _stem(entry.name)))
            if len(items) % SCAN_LOG_INTERVAL == 0:
                logger.info("Scanned %s images", len(items))
    logger.info("Scanned %s images", len(items))
    return items, subdirs


def index_labels(labels_dir: Path) -> tuple[dict[str, str], int]:
    """Map stem -> absolute txt path in one leaf directory."""
    logger.info("Indexing labels in %s", labels_dir)
    label_by_stem: dict[str, str] = {}
    subdirs = 0
    with os.scandir(labels_dir) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                subdirs += 1
                continue
            if _suffix(entry.name) != ".txt":
                continue
            label_by_stem[_stem(entry.name)] = _dirent_filepath(entry)
            if len(label_by_stem) % SCAN_LOG_INTERVAL == 0:
                logger.info("Indexed %s labels", len(label_by_stem))
    logger.info("Indexed %s labels", len(label_by_stem))
    return label_by_stem, subdirs


def parse_yolo_txt(
    path: Path | str, names: list[str]
) -> tuple[list[tuple[str, list[float]]] | None, str | None]:
    """Parse one YOLO txt. Empty file is unlabeled (no error). Malformed lines return an error."""
    with open(os.fspath(path), encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    if not text.strip():
        return None, None
    boxes: list[tuple[str, list[float]]] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            return None, f"line {line_number}: expected 5 numbers, got {len(parts)}"
        try:
            class_id = int(float(parts[0]))
            center_x, center_y, width, height = (float(item) for item in parts[1:])
        except ValueError:
            return None, f"line {line_number}: not numeric"
        if not 0 <= class_id < len(names) or width <= 0 or height <= 0:
            return None, f"line {line_number}: bad class_id or box size"
        boxes.append(
            (
                names[class_id],
                [
                    round(center_x - width / 2.0, 6),
                    round(center_y - height / 2.0, 6),
                    round(width, 6),
                    round(height, 6),
                ],
            )
        )
    if not boxes:
        return None, None
    return boxes, None


def _parse_label_chunk(
    names: list[str], paths: list[str]
) -> list[tuple[list[tuple[str, list[float]]] | None, str | None]]:
    return [parse_yolo_txt(path, names) for path in paths]


def parse_yolo_txt_many(
    paths: list[str], names: list[str]
) -> list[tuple[list[tuple[str, list[float]]] | None, str | None]]:
    """Parse label files, using threads when the batch is large enough."""
    total = len(paths)
    if total == 0:
        return []
    if total < PARSE_CHUNK_SIZE:
        results = [parse_yolo_txt(path, names) for path in paths]
        logger.info("Parsed %s/%s labels", total, total)
        return results
    workers = min(PARSE_WORKERS, max(1, (total + PARSE_CHUNK_SIZE - 1) // PARSE_CHUNK_SIZE))
    logger.info("Parsing %s label files with %s workers", total, workers)
    chunks = [paths[index:index + PARSE_CHUNK_SIZE] for index in range(0, total, PARSE_CHUNK_SIZE)]
    results: list[tuple[list[tuple[str, list[float]]] | None, str | None]] = []
    next_log = PARSE_LOG_INTERVAL
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for chunk_result in pool.map(partial(_parse_label_chunk, names), chunks):
            results.extend(chunk_result)
            done = len(results)
            if done >= next_log or done == total:
                logger.info("Parsed %s/%s labels", done, total)
                next_log += PARSE_LOG_INTERVAL
    return results


def issue_row(
    issue: str,
    filepath: str = "",
    filename: str = "",
    label_filepath: str = "",
    detail: str = "",
) -> dict[str, str]:
    return {
        "issue": issue,
        "filepath": filepath,
        "filename": filename,
        "label_filepath": label_filepath,
        "detail": detail,
    }


def issue_csv_path(dataset_name: str) -> Path:
    directory = Path.cwd() / REPORT_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"import_yolo_{dataset_name}.csv"


def write_issue_csv(path: Path, rows: list[dict[str, str]]) -> None:
    rows = sorted(
        rows,
        key=lambda row: (row["issue"], row["label_filepath"], row["filepath"]),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ISSUE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def relative_posix_path(image_path: Path, root: Path) -> str:
    return image_path.resolve().relative_to(root.resolve()).as_posix()


def list_split_names(images_root: Path) -> list[str]:
    if not images_root.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_root}")
    return sorted(
        entry.name
        for entry in os.scandir(images_root)
        if entry.is_dir(follow_symlinks=False)
    )


def _append_ignored_subdir(issues: list[dict[str, str]], directory: Path, subdirs: int) -> None:
    if subdirs:
        issues.append(
            issue_row(
                "ignored_subdir",
                filepath=str(directory),
                detail=f"{subdirs} subdirectory(ies) ignored",
            )
        )


def _make_sample(fo_module: object, pending: PendingSample) -> object:
    sample = fo_module.Sample(
        filepath=pending.filepath,
        filename=pending.filename,
        tags=list(pending.tags),
    )
    if pending.boxes:
        sample["ground_truth"] = fo_module.Detections(
            detections=[
                fo_module.Detection(label=label, bounding_box=bbox)
                for label, bbox in pending.boxes
            ]
        )
        sample["label_filepath"] = pending.label_filepath
    if pending.relpath:
        sample[RELPATH_FIELD] = pending.relpath
    return sample


def add_samples(dataset: object, fo_module: object, pending: list[PendingSample]) -> int:
    written = 0
    total = len(pending)
    for start in range(0, total, BATCH_SIZE):
        batch = [_make_sample(fo_module, item) for item in pending[start:start + BATCH_SIZE]]
        dataset.add_samples(batch)
        written += len(batch)
        logger.info("Imported %s/%s", written, total)
    return written


def _sample_relpath(
    images_dir: Path, filename: str, relpath_root: Path | None
) -> str | None:
    if relpath_root is None:
        return None
    return relative_posix_path(images_dir / filename, relpath_root)


def prepare_leaf(
    images_dir: Path,
    tags: list[str],
    labels_dir: Path | None,
    names: list[str] | None,
    existing: set[str],
    dry_run: bool,
    *,
    relpath_root: Path | None = None,
) -> LeafPrepareResult:
    items, image_subdirs = scan_images(images_dir)
    result = LeafPrepareResult(items_count=len(items))
    _append_ignored_subdir(result.issues, images_dir, image_subdirs)
    if not items:
        return result

    collisions = {stem for stem, count in Counter(stem for _, _, stem in items).items() if count > 1}

    label_by_stem: dict[str, str] = {}
    if labels_dir is not None:
        label_by_stem, label_subdirs = index_labels(labels_dir)
        _append_ignored_subdir(result.issues, labels_dir, label_subdirs)
        image_stems = {stem for _, _, stem in items}
        for stem, label_path in label_by_stem.items():
            if stem not in image_stems:
                result.issues.append(
                    issue_row(
                        "orphan_label",
                        label_filepath=label_path,
                        detail=f"no image for stem {stem}",
                    )
                )

    parse_jobs: list[tuple[str, str, str]] = []
    for filepath, filename, stem in items:
        if filepath in existing:
            result.skipped += 1
            continue
        result.new_filepaths.append(filepath)
        relpath = _sample_relpath(images_dir, filename, relpath_root)
        if names is None:
            result.unlabeled += 1
            if not dry_run:
                result.pending.append(
                    PendingSample(filepath, filename, None, None, list(tags), relpath)
                )
            continue
        if stem in collisions:
            result.issues.append(
                issue_row(
                    "stem_collision",
                    filepath=filepath,
                    filename=filename,
                    detail=f"multiple images share stem {stem}",
                )
            )
            result.unlabeled += 1
            if not dry_run:
                result.pending.append(
                    PendingSample(filepath, filename, None, None, list(tags), relpath)
                )
            continue
        label_path = label_by_stem.get(stem)
        if label_path is None:
            result.unlabeled += 1
            if not dry_run:
                result.pending.append(
                    PendingSample(filepath, filename, None, None, list(tags), relpath)
                )
            continue
        parse_jobs.append((filepath, filename, label_path))

    parsed = parse_yolo_txt_many(
        [label_path for _, _, label_path in parse_jobs],
        names or [],
    )
    for (filepath, filename, label_path), (boxes, error) in zip(parse_jobs, parsed, strict=True):
        if error:
            result.parse_errors += 1
            result.issues.append(
                issue_row(
                    "parse_error",
                    filepath=filepath,
                    filename=filename,
                    label_filepath=label_path,
                    detail=error,
                )
            )
        if boxes is None:
            result.unlabeled += 1
            label_filepath = None
        else:
            label_filepath = label_path
        if not dry_run:
            result.pending.append(
                PendingSample(
                    filepath,
                    filename,
                    boxes,
                    label_filepath,
                    list(tags),
                    _sample_relpath(images_dir, filename, relpath_root),
                )
            )
    return result


def _import_fiftyone():
    warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"glob2(\.|$)")
    import fiftyone as fo

    return fo


def load_existing_filepaths(fo_module: object, dataset_name: str) -> tuple[object | None, set[str]]:
    if not fo_module.dataset_exists(dataset_name):
        return None, set()
    logger.info("Loading dataset %s", dataset_name)
    dataset = fo_module.load_dataset(dataset_name)
    existing = {os.path.realpath(str(path)) for path in dataset.values("filepath")}
    logger.info("Existing samples=%s", len(existing))
    return dataset, existing


def delete_dataset(fo_module: object, dataset_name: str, reason: str) -> None:
    if not fo_module.dataset_exists(dataset_name):
        return
    logger.warning("%s: deleting dataset %s", reason, dataset_name)
    fo_module.delete_dataset(dataset_name)


def _print_leaf_counts(
    result: LeafPrepareResult,
    csv_path: str,
) -> None:
    new_samples = result.items_count - result.skipped
    labeled = new_samples - result.unlabeled
    print(
        f"scanned={result.items_count} new_samples={new_samples} skipped_existing={result.skipped} "
        f"labeled={labeled} unlabeled={result.unlabeled} parse_errors={result.parse_errors} "
        f"issues={len(result.issues)}"
        + (f" csv_path={csv_path}" if csv_path else ""),
        flush=True,
    )


def _write_issues(dataset_name: str, issues: list[dict[str, str]]) -> str:
    if not issues:
        return ""
    path = issue_csv_path(dataset_name)
    write_issue_csv(path, issues)
    return str(path)


def run_leaf(
    dataset_name: str,
    images_dir: Path,
    tags: list[str],
    labels_dir: Path | None,
    names: list[str] | None,
    dry_run: bool,
) -> int:
    print(f"mode={'dry-run' if dry_run else 'import'} dataset_name={dataset_name}", flush=True)
    fo_module = _import_fiftyone()
    dataset, existing = load_existing_filepaths(fo_module, dataset_name)
    result = prepare_leaf(images_dir, tags, labels_dir, names, existing, dry_run)
    if result.items_count == 0:
        logger.error("No images in leaf directory: %s", images_dir)
        return 1
    csv_path = _write_issues(dataset_name, result.issues)
    _print_leaf_counts(result, csv_path)
    if dry_run:
        return 0
    if dataset is None:
        dataset = fo_module.Dataset(name=dataset_name, persistent=True)
    written = add_samples(dataset, fo_module, result.pending)
    dataset.save()
    print(f"written_samples={written} import_done=true", flush=True)
    return 0


def coco_split_dirs(coco_root: Path, split_name: str) -> tuple[Path, Path | None]:
    images_dir = coco_root / "images" / split_name
    labels_dir = coco_root / "labels" / split_name
    if not labels_dir.is_dir():
        return images_dir, None
    return images_dir, labels_dir


def run_coco(
    coco_root: Path,
    dataset_name: str,
    dry_run: bool,
    replace: bool,
) -> int:
    names = list(COCO_80_CLASSES)
    split_names = list_split_names(coco_root / "images")
    if not split_names:
        logger.error("No split folders under %s", coco_root / "images")
        return 1

    print(f"mode={'dry-run' if dry_run else 'import'} dataset_name={dataset_name}", flush=True)
    print(f"coco_root={coco_root} replace={str(replace).lower()}", flush=True)

    fo_module = _import_fiftyone()
    dataset_existed = fo_module.dataset_exists(dataset_name)
    print(f"dataset_exists={str(dataset_existed).lower()}", flush=True)

    existing: set[str] = set()
    dataset = None
    treat_as_empty = replace
    if not treat_as_empty:
        dataset, existing = load_existing_filepaths(fo_module, dataset_name)

    split_results: list[tuple[str, LeafPrepareResult]] = []
    all_issues: list[dict[str, str]] = []
    total = LeafPrepareResult()
    for split_name in split_names:
        images_dir, labels_dir = coco_split_dirs(coco_root, split_name)
        if labels_dir is None:
            logger.warning("No labels directory for split %s", split_name)
        split_names_for_labels = names if labels_dir is not None else None
        result = prepare_leaf(
            images_dir,
            [SOURCE_TAG_DEFAULT, split_name],
            labels_dir,
            split_names_for_labels,
            existing,
            dry_run,
            relpath_root=coco_root,
        )
        existing.update(result.new_filepaths)
        split_results.append((split_name, result))
        all_issues.extend(result.issues)
        total.items_count += result.items_count
        total.skipped += result.skipped
        total.unlabeled += result.unlabeled
        total.parse_errors += result.parse_errors
        total.pending.extend(result.pending)
        print(
            f"[split {split_name}] scanned={result.items_count} "
            f"new_samples={result.items_count - result.skipped} "
            f"skipped_existing={result.skipped} "
            f"labeled={result.items_count - result.skipped - result.unlabeled} "
            f"unlabeled={result.unlabeled} "
            f"parse_errors={result.parse_errors} issues={len(result.issues)}",
            flush=True,
        )

    if total.items_count == 0:
        logger.error("No images under %s", coco_root / "images")
        return 1

    csv_path = _write_issues(dataset_name, all_issues)
    _print_leaf_counts(total, csv_path)
    if dry_run:
        if replace and dataset_existed:
            print("replace_ok=true (real import will DELETE and recreate this dataset)", flush=True)
        return 0

    created_this_run = False
    try:
        if replace:
            delete_dataset(fo_module, dataset_name, "Replacing existing dataset")
            dataset = None
        if dataset is None:
            if not fo_module.dataset_exists(dataset_name):
                dataset = fo_module.Dataset(name=dataset_name, persistent=True)
                created_this_run = True
            else:
                dataset, _ = load_existing_filepaths(fo_module, dataset_name)
        written = add_samples(dataset, fo_module, total.pending)
        dataset.save()
        print(f"written_samples={written} import_done=true", flush=True)
        return 0
    except Exception:
        if created_this_run:
            delete_dataset(fo_module, dataset_name, "Import failed; removing incomplete dataset")
        raise
