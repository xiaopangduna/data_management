"""Import a YOLO-layout COCO tree into a FiftyOne dataset.

Only reads ``<coco-root>/images`` and ``<coco-root>/labels``.
Detection is stored in ``ground_truth_detect``; other label types are rejected
until added. Samples are tagged ``coco`` plus the split folder name.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import fiftyone as fo

logger = logging.getLogger(__name__)

DEFAULT_COCO_ROOT = Path(
    "/home/xiaopangdun/project/deep_learning/src/train/datasets/COCO/coco"
)
DEFAULT_DATASET_NAME = "coco2017"
SUPPORTED_LABEL_TYPES = frozenset({"detection"})
SOURCE_TAG = "coco"
DETECT_FIELD = "ground_truth_detect"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ADD_SAMPLES_BATCH_SIZE = 1000
MAX_REPORTED_PARSE_ERRORS = 20

# Ultralytics / COCO 2017 thing classes, index 0 == person.
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
        description="Import YOLO-layout COCO images/labels into FiftyOne."
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
        help=(
            "Comma-separated label kinds. Only 'detection' is implemented "
            f"(writes {DETECT_FIELD})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and validate only; do not write to FiftyOne.",
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
    """Return split folder names under ``images/``.

    Args:
        images_root: Path to ``<coco-root>/images``.

    Returns:
        Sorted split directory names.
    """
    if not images_root.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_root}")
    return sorted(path.name for path in images_root.iterdir() if path.is_dir())


def list_image_paths(split_images_dir: Path) -> list[Path]:
    """List image files in one split folder.

    Args:
        split_images_dir: Path such as ``images/val2017``.

    Returns:
        Sorted image paths.
    """
    return sorted(
        path
        for path in split_images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def list_label_stems(split_labels_dir: Path) -> set[str]:
    """Return YOLO txt stems in one split labels folder.

    Args:
        split_labels_dir: Path such as ``labels/val2017``.

    Returns:
        Set of file stems (excludes ``*.cache``).
    """
    if not split_labels_dir.is_dir():
        return set()
    return {
        path.stem
        for path in split_labels_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    }


def _record_parse_error(result: SplitScanResult, message: str) -> None:
    result.parse_error_count += 1
    if len(result.parse_error_messages) < MAX_REPORTED_PARSE_ERRORS:
        result.parse_error_messages.append(message)


def parse_yolo_detection_file(
    label_path: Path, result: SplitScanResult, emit_detections: bool
) -> list[fo.Detection]:
    """Parse a YOLO txt into FiftyOne detections.

    Lines with more than 5 numbers are treated as non-detection (e.g. YOLO-seg)
    and skipped. Invalid lines increment ``result.parse_error_count``.

    Args:
        label_path: Path to a YOLO label file.
        result: Split counters to update in place.
        emit_detections: If False, only update counters (dry-run).

    Returns:
        Detection objects for valid 5-number rows, or an empty list.
    """
    detections: list[fo.Detection] = []
    text = label_path.read_text(encoding="utf-8")
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
        logger.debug("Skip non-detection line %s:%s", label_path, line_number)
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
    coco_root: Path, split_name: str, build_samples: bool
) -> tuple[SplitScanResult, list[fo.Sample]]:
    """Scan one split and optionally build FiftyOne samples.

    Args:
        coco_root: COCO root containing ``images`` and ``labels``.
        split_name: Folder name under ``images/``.
        build_samples: If True, construct ``fo.Sample`` objects.

    Returns:
        Scan counters and samples (empty when ``build_samples`` is False).
    """
    result = SplitScanResult(split_name=split_name)
    images_dir = coco_root / "images" / split_name
    labels_dir = coco_root / "labels" / split_name
    image_paths = list_image_paths(images_dir)
    label_stems = list_label_stems(labels_dir)
    image_stems = {path.stem for path in image_paths}
    result.image_count = len(image_paths)
    result.orphan_label_count = len(label_stems - image_stems)
    samples: list[fo.Sample] = []
    for image_path in image_paths:
        sample = _scan_one_image(
            image_path, labels_dir, split_name, result, build_samples
        )
        if sample is not None:
            samples.append(sample)
    return result, samples


def _scan_one_image(
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
    if detections:
        sample[DETECT_FIELD] = fo.Detections(detections=detections)
    return sample


def add_samples_in_batches(dataset: fo.Dataset, samples: list[fo.Sample]) -> None:
    """Insert samples in fixed-size batches.

    Args:
        dataset: Destination FiftyOne dataset.
        samples: Samples to add.
    """
    for start_index in range(0, len(samples), ADD_SAMPLES_BATCH_SIZE):
        batch = samples[start_index : start_index + ADD_SAMPLES_BATCH_SIZE]
        dataset.add_samples(batch)
        logger.info(
            "Added samples %s-%s",
            start_index + 1,
            start_index + len(batch),
        )


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
        dataset_exists: Whether ``dataset_name`` already exists.
        dry_run: Whether this run writes nothing.
    """
    mode = "dry-run" if dry_run else "import"
    print(f"mode={mode}")
    print(f"coco_root={coco_root}")
    print(f"dataset_name={dataset_name}")
    print(f"label_types={','.join(label_types)}")
    print(f"dataset_exists={dataset_exists}")
    for result in split_results:
        _print_split_result(result)
    parse_error_total = sum(item.parse_error_count for item in split_results)
    if dataset_exists:
        print("merge_ok=false (dataset already exists; real import would abort)")
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
    split_results: list[SplitScanResult] = []
    build_samples = dataset is not None
    for split_name in split_names:
        result, samples = scan_split(coco_root, split_name, build_samples)
        split_results.append(result)
        if dataset is not None:
            logger.info("Importing split %s (%s samples)", split_name, len(samples))
            add_samples_in_batches(dataset, samples)
    return split_results


def main(argv: list[str] | None = None) -> int:
    """Run dry-run validation or FiftyOne import.

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
    dataset_exists = fo.dataset_exists(args.dataset_name)
    if dataset_exists and not args.dry_run:
        logger.error("Dataset already exists: %s", args.dataset_name)
        return 2
    dataset = None if args.dry_run else create_empty_dataset(args.dataset_name)
    split_results = run_splits(coco_root, split_names, dataset)
    print_scan_report(
        coco_root, args.dataset_name, label_types, split_results,
        dataset_exists, args.dry_run,
    )
    if dataset is not None:
        dataset.save()
        print(f"imported_dataset={args.dataset_name} samples={len(dataset)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
