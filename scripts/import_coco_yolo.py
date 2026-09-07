"""Import YOLO-layout COCO into FiftyOne (delete-and-recreate).

Reads only ``<coco-root>/images`` and ``labels``. Writes ``relpath``, tags
``coco`` + split, and ``ground_truth_detect``. Hashes go in enrich_fiftyone_media.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"glob2(\.|$)",
)

import fiftyone as fo

logger = logging.getLogger(__name__)

DEFAULT_COCO_ROOT = Path(
    "/home/xiaopangdun/project/deep_learning/src/train/datasets/COCO/coco"
)
DEFAULT_DATASET_NAME = "coco2017"
SUPPORTED_LABEL_TYPES = frozenset({"detection"})
SOURCE_TAG = "coco"
DETECT_FIELD = "ground_truth_detect"
RELPATH_FIELD = "relpath"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ADD_SAMPLES_BATCH_SIZE = 1000
SCAN_LOG_INTERVAL = 5000
MAX_REPORTED_PARSE_ERRORS = 20

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
class SplitScanResult:
    """Per-split counts collected while scanning the COCO tree."""

    split_name: str
    image_count: int = 0
    labeled_image_count: int = 0
    unlabeled_image_count: int = 0
    orphan_label_count: int = 0
    detection_box_count: int = 0
    skipped_non_detection_line_count: int = 0
    parse_error_count: int = 0
    parse_error_messages: list[str] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for COCO import.

    Args:
        argv: Optional argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Import YOLO COCO into FiftyOne; deletes same-name dataset."
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=DEFAULT_COCO_ROOT,
        help="Directory that contains images/ and labels/.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="FiftyOne dataset name to create.",
    )
    parser.add_argument(
        "--label-types",
        default="detection",
        help="Comma-separated kinds. Only detection -> ground_truth_detect.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and validate only; do not delete or write a dataset.",
    )
    return parser.parse_args(argv)


def parse_label_types(raw_label_types: str) -> list[str]:
    """Split and validate ``--label-types``.

    Args:
        raw_label_types: Comma-separated label type names.

    Returns:
        Normalized label type names.

    Raises:
        ValueError: If empty or any type is not yet implemented.
    """
    label_types = [part.strip() for part in raw_label_types.split(",") if part.strip()]
    if not label_types:
        raise ValueError("--label-types must not be empty")
    unsupported = sorted(set(label_types) - SUPPORTED_LABEL_TYPES)
    if unsupported:
        supported = ", ".join(sorted(SUPPORTED_LABEL_TYPES))
        raise ValueError(
            f"Unsupported --label-types {unsupported}; implemented: {supported}"
        )
    return label_types


def list_split_names(images_root: Path) -> list[str]:
    """Return sorted split folder names under ``images/``."""
    if not images_root.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_root}")
    return sorted(path.name for path in images_root.iterdir() if path.is_dir())


def list_image_paths(split_images_dir: Path) -> list[Path]:
    """Return sorted image files in one split folder."""
    return sorted(
        path
        for path in split_images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def list_label_stems(split_labels_dir: Path) -> set[str]:
    """Return YOLO txt stems in one split labels folder."""
    if not split_labels_dir.is_dir():
        return set()
    return {
        path.stem
        for path in split_labels_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    }


def relative_posix_path(image_path: Path, coco_root: Path) -> str:
    """Return a POSIX path relative to ``coco_root``."""
    return image_path.resolve().relative_to(coco_root).as_posix()


def _record_parse_error(result: SplitScanResult, message: str) -> None:
    result.parse_error_count += 1
    if len(result.parse_error_messages) < MAX_REPORTED_PARSE_ERRORS:
        result.parse_error_messages.append(message)


def parse_yolo_detection_file(
    label_path: Path, result: SplitScanResult, emit_detections: bool
) -> list[fo.Detection]:
    """Parse a YOLO txt into FiftyOne detections.

    Args:
        label_path: Path to a YOLO label file.
        result: Split counters to update in place.
        emit_detections: If False, only update counters (dry-run).

    Returns:
        Detection objects for valid 5-number rows, or an empty list.
    """
    detections: list[fo.Detection] = []
    text = label_path.read_text(encoding="utf-8", errors="replace")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        detection = _parse_yolo_detection_line(
            raw_line, label_path, line_number, result, emit_detections
        )
        if detection is not None:
            detections.append(detection)
    return detections


def _parse_yolo_detection_line(
    raw_line: str,
    label_path: Path,
    line_number: int,
    result: SplitScanResult,
    emit_detections: bool,
) -> fo.Detection | None:
    line = raw_line.strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) > 5:
        result.skipped_non_detection_line_count += 1
        return None
    parsed = _parse_yolo_box_numbers(parts, label_path, line_number, result)
    if parsed is None:
        return None
    class_id, center_x, center_y, width, height = parsed
    result.detection_box_count += 1
    if not emit_detections:
        return None
    return fo.Detection(
        label=COCO_80_CLASSES[class_id],
        bounding_box=[center_x - width / 2.0, center_y - height / 2.0, width, height],
    )


def _parse_yolo_box_numbers(
    parts: list[str],
    label_path: Path,
    line_number: int,
    result: SplitScanResult,
) -> tuple[int, float, float, float, float] | None:
    if len(parts) != 5:
        _record_parse_error(
            result, f"{label_path}:{line_number}: expected 5 numbers, got {len(parts)}"
        )
        return None
    try:
        class_id = int(float(parts[0]))
        center_x, center_y, width, height = (float(item) for item in parts[1:])
    except ValueError:
        _record_parse_error(result, f"{label_path}:{line_number}: not numeric")
        return None
    if not 0 <= class_id < len(COCO_80_CLASSES):
        _record_parse_error(
            result, f"{label_path}:{line_number}: class_id {class_id} out of range"
        )
        return None
    return class_id, center_x, center_y, width, height


def scan_split(
    coco_root: Path, split_name: str, dataset: fo.Dataset | None
) -> SplitScanResult:
    """Scan one split and optionally insert samples in batches.

    Args:
        coco_root: COCO root containing ``images`` and ``labels``.
        split_name: Folder name under ``images/``.
        dataset: Destination dataset, or None for dry-run.

    Returns:
        Scan counters for the split.
    """
    result = SplitScanResult(split_name=split_name)
    labels_dir = coco_root / "labels" / split_name
    image_paths = list_image_paths(coco_root / "images" / split_name)
    result.image_count = len(image_paths)
    result.orphan_label_count = len(
        list_label_stems(labels_dir) - {path.stem for path in image_paths}
    )
    if not labels_dir.is_dir():
        logger.warning("No labels directory for split %s: %s", split_name, labels_dir)
    logger.info("Scanning split %s (%s images)", split_name, result.image_count)
    _scan_split_images(coco_root, image_paths, labels_dir, split_name, result, dataset)
    return result


def _scan_one_image(
    coco_root: Path,
    image_path: Path,
    labels_dir: Path,
    split_name: str,
    result: SplitScanResult,
    build_samples: bool,
) -> fo.Sample | None:
    label_path = labels_dir / f"{image_path.stem}.txt"
    detections: list[fo.Detection] = []
    if label_path.is_file():
        result.labeled_image_count += 1
        detections = parse_yolo_detection_file(
            label_path, result, emit_detections=build_samples
        )
    else:
        result.unlabeled_image_count += 1
    if not build_samples:
        return None
    sample = fo.Sample(filepath=str(image_path.resolve()))
    sample.tags = [SOURCE_TAG, split_name]
    sample[RELPATH_FIELD] = relative_posix_path(image_path, coco_root)
    if detections:
        sample[DETECT_FIELD] = fo.Detections(detections=detections)
    return sample


def _scan_split_images(
    coco_root: Path,
    image_paths: list[Path],
    labels_dir: Path,
    split_name: str,
    result: SplitScanResult,
    dataset: fo.Dataset | None,
) -> None:
    batch: list[fo.Sample] = []
    added_count = 0
    build_samples = dataset is not None
    for image_index, image_path in enumerate(image_paths, start=1):
        sample = _scan_one_image(
            coco_root, image_path, labels_dir, split_name, result, build_samples
        )
        if sample is not None:
            batch.append(sample)
        added_count += _flush_sample_batch(
            dataset, batch, split_name, added_count, result.image_count, force=False
        )
        if image_index % SCAN_LOG_INTERVAL == 0 or image_index == result.image_count:
            logger.info(
                "Split %s scanned %s/%s", split_name, image_index, result.image_count
            )
    _flush_sample_batch(
        dataset, batch, split_name, added_count, result.image_count, force=True
    )


def _flush_sample_batch(
    dataset: fo.Dataset | None,
    batch: list[fo.Sample],
    split_name: str,
    added_count: int,
    image_count: int,
    force: bool,
) -> int:
    if dataset is None or not batch:
        return 0
    if not force and len(batch) < ADD_SAMPLES_BATCH_SIZE:
        return 0
    dataset.add_samples(batch)
    written = len(batch)
    logger.info("Split %s wrote %s/%s", split_name, added_count + written, image_count)
    batch.clear()
    return written


def print_scan_report(
    coco_root: Path,
    dataset_name: str,
    label_types: list[str],
    split_results: list[SplitScanResult],
    dataset_exists: bool,
    dry_run: bool,
) -> None:
    """Print a human-readable scan summary.

    Args:
        coco_root: Data root that was scanned.
        dataset_name: Target FiftyOne dataset name.
        label_types: Requested label types.
        split_results: Per-split counters.
        dataset_exists: Whether the named dataset existed before this run.
        dry_run: Whether this run writes nothing.
    """
    print(f"mode={'dry-run' if dry_run else 'import'}")
    print(f"coco_root={coco_root}")
    print(f"dataset_name={dataset_name}")
    print(f"label_types={','.join(label_types)}")
    print(f"dataset_exists={dataset_exists}")
    for result in split_results:
        _print_split_result(result)
    parse_error_total = sum(item.parse_error_count for item in split_results)
    if dry_run and dataset_exists:
        print("merge_ok=true (real import will DELETE and recreate this dataset)")
    elif parse_error_total:
        print("merge_ok=true with parse errors (bad lines skipped)")
    else:
        print("merge_ok=true")


def _print_split_result(result: SplitScanResult) -> None:
    print(f"[split {result.split_name}]")
    print(f"  images={result.image_count}")
    print(f"  labeled_images={result.labeled_image_count}")
    print(f"  unlabeled_images={result.unlabeled_image_count}")
    print(f"  orphan_labels={result.orphan_label_count}")
    print(f"  detection_boxes={result.detection_box_count}")
    print(f"  skipped_non_detection_lines={result.skipped_non_detection_line_count}")
    print(f"  parse_errors={result.parse_error_count}")
    for message in result.parse_error_messages:
        print(f"  parse_error: {message}")


def create_empty_dataset(dataset_name: str) -> fo.Dataset:
    """Create a persistent FiftyOne dataset.

    Args:
        dataset_name: New dataset name; must not already exist.

    Returns:
        The created dataset.
    """
    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True
    return dataset


def delete_dataset(dataset_name: str, reason: str) -> None:
    """Delete a FiftyOne dataset if it exists.

    Args:
        dataset_name: Dataset name to delete.
        reason: Log message explaining why.
    """
    if not fo.dataset_exists(dataset_name):
        return
    logger.warning("%s: deleting dataset %s", reason, dataset_name)
    fo.delete_dataset(dataset_name)


def run_splits(
    coco_root: Path,
    split_names: list[str],
    dataset: fo.Dataset | None,
) -> list[SplitScanResult]:
    """Scan each split; write samples when ``dataset`` is not None.

    Args:
        coco_root: COCO root directory.
        split_names: Split folder names under ``images/``.
        dataset: Destination dataset, or None for dry-run.

    Returns:
        Per-split scan counters.
    """
    return [scan_split(coco_root, split_name, dataset) for split_name in split_names]


def _import_dataset(
    coco_root: Path,
    dataset_name: str,
    split_names: list[str],
    dataset_existed: bool,
) -> list[SplitScanResult]:
    if dataset_existed:
        delete_dataset(dataset_name, "Replacing existing dataset")
    dataset = create_empty_dataset(dataset_name)
    try:
        split_results = run_splits(coco_root, split_names, dataset)
        dataset.save()
        print(f"imported_dataset={dataset_name} samples={len(dataset)}")
        return split_results
    except Exception:
        delete_dataset(dataset_name, "Import failed; removing incomplete dataset")
        raise


def main(argv: list[str] | None = None) -> int:
    """Run dry-run validation or a delete-and-recreate import.

    Args:
        argv: Optional CLI arguments.

    Returns:
        Process exit code (0 on success).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)7s] %(name)s:%(lineno)d - %(message)s",
    )
    args = parse_args(argv)
    coco_root = args.coco_root.expanduser().resolve()
    try:
        label_types = parse_label_types(args.label_types)
        split_names = list_split_names(coco_root / "images")
    except (ValueError, FileNotFoundError) as error:
        logger.error("%s", error)
        return 1
    if not split_names:
        logger.error("No split folders under %s", coco_root / "images")
        return 1
    dataset_existed = fo.dataset_exists(args.dataset_name)
    if args.dry_run:
        split_results = run_splits(coco_root, split_names, None)
    else:
        split_results = _import_dataset(
            coco_root, args.dataset_name, split_names, dataset_existed
        )
    print_scan_report(
        coco_root, args.dataset_name, label_types, split_results,
        dataset_existed, args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
