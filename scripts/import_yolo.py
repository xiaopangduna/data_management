"""Import a YOLO leaf directory into FiftyOne.

Creates a persistent dataset if needed and appends images whose resolved
filepath is not already present. Optional YOLO txt files are paired by stem
(``foo.jpg`` ↔ ``foo.txt``). Existing samples are skipped entirely; changing
their labels is left for a later update script.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data_management.yolo_import import (
    IMAGE_SUFFIXES,
    PARSE_CHUNK_SIZE,
    PARSE_WORKERS,
    class_names,
    index_labels,
    list_leaf_files,
    nonempty,
    parse_yolo_txt,
    parse_yolo_txt_many,
    run_leaf,
    scan_images,
    tag_list,
)

logger = logging.getLogger(__name__)

__all__ = [
    "IMAGE_SUFFIXES",
    "PARSE_CHUNK_SIZE",
    "PARSE_WORKERS",
    "class_names",
    "index_labels",
    "list_leaf_files",
    "nonempty",
    "parse_args",
    "parse_yolo_txt",
    "parse_yolo_txt_many",
    "scan_images",
    "tag_list",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=nonempty)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--sample-tags", required=True, type=tag_list)
    parser.add_argument("--label-dir", type=Path)
    parser.add_argument("--classes", type=class_names)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.label_dir is not None and not args.classes:
        parser.error("--classes is required when --label-dir is set")
    if args.classes and args.label_dir is None:
        parser.error("--classes requires --label-dir")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    try:
        images_dir = args.images_dir.expanduser().resolve(strict=True)
        if not images_dir.is_dir():
            raise ValueError(f"Not a directory: {images_dir}")
        labels_dir = None
        if args.label_dir is not None:
            labels_dir = args.label_dir.expanduser().resolve(strict=True)
            if not labels_dir.is_dir():
                raise ValueError(f"Not a directory: {labels_dir}")
        return run_leaf(
            args.dataset, images_dir, args.sample_tags, labels_dir, args.classes, args.dry_run
        )
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except Exception as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
