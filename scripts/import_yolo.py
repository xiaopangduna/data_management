"""Import a YOLO leaf directory into FiftyOne.

Creates a persistent dataset if needed and appends images whose resolved
filepath is not already present. Optional YOLO txt files are paired by stem
(``foo.jpg`` ↔ ``foo.txt``). Existing samples are skipped entirely; changing
their labels is left for a later update script.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BATCH_SIZE = 1000


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


def list_leaf_files(directory: Path, suffixes: set[str]) -> list[Path]:
    """List files in one directory. Does not recurse."""
    files: list[Path] = []
    subdirs = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                subdirs += 1
                continue
            path = Path(entry.path)
            if path.suffix.lower() in suffixes:
                files.append(path)
    if subdirs:
        logger.warning("Ignoring %s subdirectory(ies) under %s; pass a leaf directory", subdirs, directory)
    files.sort()
    return files


def scan_images(images_dir: Path) -> list[tuple[str, str, str]]:
    """Return (resolved filepath, filename, stem), skipping duplicate targets."""
    items: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for path in list_leaf_files(images_dir, IMAGE_SUFFIXES):
        filepath = str(path.resolve(strict=True))
        if filepath in seen:
            continue
        seen.add(filepath)
        items.append((filepath, path.name, path.stem))
    return items


def parse_yolo_txt(path: Path, names: list[str]) -> list[tuple[str, list[float]]] | None:
    """Parse one YOLO txt into (class_name, top-left xywh) boxes. None if unusable."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        logger.warning("empty label: %s", path)
        return None
    boxes: list[tuple[str, list[float]]] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            logger.warning("parse error %s:%s: expected 5 numbers", path, line_number)
            return None
        try:
            class_id = int(float(parts[0]))
            center_x, center_y, width, height = (float(item) for item in parts[1:])
        except ValueError:
            logger.warning("parse error %s:%s: not numeric", path, line_number)
            return None
        if not 0 <= class_id < len(names) or width <= 0 or height <= 0:
            logger.warning("parse error %s:%s: bad class_id or box size", path, line_number)
            return None
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
        logger.warning("empty label: %s", path)
        return None
    return boxes


def labels_for_stem(
    stem: str,
    collisions: set[str],
    label_by_stem: dict[str, Path],
    names: list[str],
) -> tuple[list[tuple[str, list[float]]] | None, str | None, bool]:
    """Return boxes, label path, and whether a parse warning was emitted."""
    if stem in collisions:
        return None, None, False
    label_path = label_by_stem.get(stem)
    if label_path is None:
        return None, None, False
    boxes = parse_yolo_txt(label_path, names)
    if boxes is None:
        return None, None, True
    return boxes, str(label_path.resolve()), False


def add_samples(dataset: object, samples: list[object]) -> int:
    written = 0
    for start in range(0, len(samples), BATCH_SIZE):
        batch = samples[start:start + BATCH_SIZE]
        dataset.add_samples(batch)
        written += len(batch)
        logger.info("Imported %s/%s", written, len(samples))
    return written


def run(
    dataset_name: str,
    images_dir: Path,
    tags: list[str],
    labels_dir: Path | None,
    names: list[str] | None,
    dry_run: bool,
) -> int:
    items = scan_images(images_dir)
    if not items:
        logger.error("No images in leaf directory: %s", images_dir)
        return 1

    collisions = {stem for stem, count in Counter(stem for _, _, stem in items).items() if count > 1}
    for stem in sorted(collisions):
        logger.warning("stem collision %s: images imported, labels skipped", stem)

    label_by_stem: dict[str, Path] = {}
    if labels_dir is not None:
        label_by_stem = {path.stem: path for path in list_leaf_files(labels_dir, {".txt"})}

    warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"glob2(\.|$)")
    import fiftyone as fo

    dataset = None
    existing: set[str] = set()
    if fo.dataset_exists(dataset_name):
        dataset = fo.load_dataset(dataset_name)
        existing = {str(Path(str(path)).resolve()) for path in dataset.values("filepath")}

    pending: list[tuple[str, str, list[tuple[str, list[float]]] | None, str | None]] = []
    skipped = unlabeled = parse_warnings = 0
    for filepath, filename, stem in items:
        if filepath in existing:
            skipped += 1
            continue
        boxes = None
        label_filepath = None
        if names is not None:
            boxes, label_filepath, warned = labels_for_stem(stem, collisions, label_by_stem, names)
            parse_warnings += int(warned)
        if boxes is None:
            unlabeled += 1
        pending.append((filepath, filename, boxes, label_filepath))

    print(f"mode={'dry-run' if dry_run else 'import'} dataset_name={dataset_name}")
    print(
        f"scanned={len(items)} new_samples={len(pending)} skipped_existing={skipped} "
        f"unlabeled={unlabeled} parse_warnings={parse_warnings}"
    )
    if dry_run:
        return 0

    if dataset is None:
        dataset = fo.Dataset(name=dataset_name, persistent=True)
    samples = []
    for filepath, filename, boxes, label_filepath in pending:
        sample = fo.Sample(filepath=filepath, filename=filename, tags=list(tags))
        if boxes:
            sample["ground_truth"] = fo.Detections(
                detections=[fo.Detection(label=label, bounding_box=bbox) for label, bbox in boxes]
            )
            sample["label_filepath"] = label_filepath
        samples.append(sample)
    written = add_samples(dataset, samples)
    dataset.save()
    print(f"written_samples={written} import_done=true")
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
