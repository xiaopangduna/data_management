"""Import a YOLO-layout COCO tree into FiftyOne.

Reads ``<coco-root>/images/<split>/`` and optional ``labels/<split>/``.
Uses COCO 80 classes and tags ``coco`` + split. Appends by filepath unless
``--replace`` deletes and recreates the dataset.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data_management.yolo_import import nonempty, run_coco

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-root", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True, type=nonempty)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    try:
        coco_root = args.coco_root.expanduser().resolve(strict=True)
        if not coco_root.is_dir():
            raise ValueError(f"Not a directory: {coco_root}")
        return run_coco(
            coco_root,
            args.dataset_name,
            args.dry_run,
            args.replace,
        )
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except Exception as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
