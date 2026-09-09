"""Import a YOLO leaf directory into FiftyOne.

Creates a persistent dataset if needed and appends images whose resolved
filepath is not already present. Optional YOLO txt files are paired by stem
(``foo.jpg`` ↔ ``foo.txt``). Existing samples are skipped entirely; changing
their labels is left for a later update script.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True, type=nonempty)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--tags", required=True, type=tag_list)
    parser.add_argument("--labels-dir", type=Path)
    parser.add_argument("--class-names", type=class_names)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.labels_dir is not None and not args.class_names:
        parser.error("--class-names is required when --labels-dir is set")
    if args.class_names and args.labels_dir is None:
        parser.error("--class-names requires --labels-dir")
    return args


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


def _make_sample(
    fo_module: object,
    tags: list[str],
    filepath: str,
    filename: str,
    boxes: list[tuple[str, list[float]]] | None,
    label_filepath: str | None,
) -> object:
    sample = fo_module.Sample(filepath=filepath, filename=filename, tags=list(tags))
    if boxes:
        sample["ground_truth"] = fo_module.Detections(
            detections=[fo_module.Detection(label=label, bounding_box=bbox) for label, bbox in boxes]
        )
        sample["label_filepath"] = label_filepath
    return sample


def add_samples(
    dataset: object,
    fo_module: object,
    pending: list[tuple[str, str, list[tuple[str, list[float]]] | None, str | None]],
    tags: list[str],
) -> int:
    written = 0
    total = len(pending)
    for start in range(0, total, BATCH_SIZE):
        batch = [
            _make_sample(fo_module, tags, filepath, filename, boxes, label_filepath)
            for filepath, filename, boxes, label_filepath in pending[start:start + BATCH_SIZE]
        ]
        dataset.add_samples(batch)
        written += len(batch)
        logger.info("Imported %s/%s", written, total)
    return written


def _append_ignored_subdir(issues: list[dict[str, str]], directory: Path, subdirs: int) -> None:
    if subdirs:
        issues.append(
            issue_row(
                "ignored_subdir",
                filepath=str(directory),
                detail=f"{subdirs} subdirectory(ies) ignored",
            )
        )


def run(
    dataset_name: str,
    images_dir: Path,
    tags: list[str],
    labels_dir: Path | None,
    names: list[str] | None,
    dry_run: bool,
) -> int:
    print(f"mode={'dry-run' if dry_run else 'import'} dataset_name={dataset_name}", flush=True)
    items, image_subdirs = scan_images(images_dir)
    if not items:
        logger.error("No images in leaf directory: %s", images_dir)
        return 1

    issues: list[dict[str, str]] = []
    _append_ignored_subdir(issues, images_dir, image_subdirs)

    collisions = {stem for stem, count in Counter(stem for _, _, stem in items).items() if count > 1}

    label_by_stem: dict[str, str] = {}
    if labels_dir is not None:
        label_by_stem, label_subdirs = index_labels(labels_dir)
        _append_ignored_subdir(issues, labels_dir, label_subdirs)

    warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"glob2(\.|$)")
    import fiftyone as fo

    dataset = None
    existing: set[str] = set()
    if fo.dataset_exists(dataset_name):
        logger.info("Loading dataset %s", dataset_name)
        dataset = fo.load_dataset(dataset_name)
        existing = {os.path.realpath(str(path)) for path in dataset.values("filepath")}
        logger.info("Existing samples=%s", len(existing))

    pending: list[tuple[str, str, list[tuple[str, list[float]]] | None, str | None]] = []
    parse_jobs: list[tuple[str, str, str]] = []
    skipped = unlabeled = parse_errors = 0
    for filepath, filename, stem in items:
        if filepath in existing:
            skipped += 1
            continue
        if names is None:
            unlabeled += 1
            if not dry_run:
                pending.append((filepath, filename, None, None))
            continue
        if stem in collisions:
            issues.append(
                issue_row(
                    "stem_collision",
                    filepath=filepath,
                    filename=filename,
                    detail=f"multiple images share stem {stem}",
                )
            )
            unlabeled += 1
            if not dry_run:
                pending.append((filepath, filename, None, None))
            continue
        label_path = label_by_stem.get(stem)
        if label_path is None:
            unlabeled += 1
            if not dry_run:
                pending.append((filepath, filename, None, None))
            continue
        parse_jobs.append((filepath, filename, label_path))

    parsed = parse_yolo_txt_many(
        [label_path for _, _, label_path in parse_jobs],
        names or [],
    )
    for (filepath, filename, label_path), (boxes, error) in zip(parse_jobs, parsed, strict=True):
        if error:
            parse_errors += 1
            issues.append(
                issue_row(
                    "parse_error",
                    filepath=filepath,
                    filename=filename,
                    label_filepath=label_path,
                    detail=error,
                )
            )
        if boxes is None:
            unlabeled += 1
            label_filepath = None
        else:
            label_filepath = label_path
        if not dry_run:
            pending.append((filepath, filename, boxes, label_filepath))

    new_samples = len(items) - skipped

    csv_path = ""
    if issues:
        csv_path = str(issue_csv_path(dataset_name))
        write_issue_csv(Path(csv_path), issues)

    print(
        f"scanned={len(items)} new_samples={new_samples} skipped_existing={skipped} "
        f"unlabeled={unlabeled} parse_errors={parse_errors} issues={len(issues)}"
        + (f" csv_path={csv_path}" if csv_path else ""),
        flush=True,
    )
    if dry_run:
        return 0

    if dataset is None:
        dataset = fo.Dataset(name=dataset_name, persistent=True)
    written = add_samples(dataset, fo, pending, tags)
    dataset.save()
    print(f"written_samples={written} import_done=true", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    try:
        images_dir = args.images_dir.expanduser().resolve(strict=True)
        if not images_dir.is_dir():
            raise ValueError(f"Not a directory: {images_dir}")
        labels_dir = None
        if args.labels_dir is not None:
            labels_dir = args.labels_dir.expanduser().resolve(strict=True)
            if not labels_dir.is_dir():
                raise ValueError(f"Not a directory: {labels_dir}")
        return run(args.dataset_name, images_dir, args.tags, labels_dir, args.class_names, args.dry_run)
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except Exception as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
