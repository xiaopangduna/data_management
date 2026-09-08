"""Create a persistent FiftyOne dataset from an unlabeled image directory."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonempty(value: str) -> str:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def extensions(value: str) -> set[str]:
    parts = {part.strip().lower().lstrip(".") for part in value.split(",")}
    if not parts or any(not part or not part.isalnum() for part in parts):
        raise argparse.ArgumentTypeError("provide comma-separated image extensions")
    return {f".{part}" for part in parts}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--dataset-name", type=nonempty, required=True)
    parser.add_argument("--tags", default="", help="Comma-separated sample tags")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extensions", type=extensions, default=".jpg,.jpeg,.png,.webp,.bmp")
    parser.add_argument("--verify-images", action="store_true", help="Decode images; skip unreadable files")
    parser.add_argument("--batch-size", type=positive_int, default=1000)
    parser.add_argument("--dry-run", action="store_true", help="Scan only; do not connect to the database")
    return parser.parse_args(argv)


def verify_image(path: Path) -> None:
    from PIL import Image

    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()


def scan_images(root: Path, args: argparse.Namespace) -> list[tuple[Path, str]]:
    """Return canonical paths and original relative paths, deduplicating symlinks."""
    images: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    scanned = duplicates = invalid = 0

    def raise_walk_error(error: OSError) -> None:
        raise error

    for directory, dirs, filenames in os.walk(root, onerror=raise_walk_error):
        dirs.sort()
        if not args.recursive:
            dirs.clear()
        for filename in sorted(filenames):
            path = Path(directory) / filename
            if path.suffix.lower() not in args.extensions:
                continue
            scanned += 1
            try:
                canonical = path.resolve(strict=True)
                if not canonical.is_file():
                    raise ValueError("not a regular file")
                if canonical in seen:
                    duplicates += 1
                    continue
                seen.add(canonical)
                if args.verify_images:
                    verify_image(canonical)
            except Exception as error:
                if not args.verify_images:
                    raise
                invalid += 1
                logger.warning("Skipping %s: %s", path, error)
                continue
            images.append((canonical, path.relative_to(root).as_posix()))
    print(f"scanned={scanned} valid={len(images)} duplicates={duplicates} invalid={invalid} skipped={duplicates + invalid}")
    return images


def import_images(args: argparse.Namespace, images: list[tuple[Path, str]]) -> None:
    # Lazy import keeps scanning/help independent of MongoDB startup.
    import fiftyone as fo

    if fo.dataset_exists(args.dataset_name):
        raise ValueError(f"Dataset already exists: {args.dataset_name}; choose another name")
    dataset = fo.Dataset(name=args.dataset_name, persistent=True)
    tags = list(dict.fromkeys(tag.strip() for tag in args.tags.split(",") if tag.strip()))
    confirmed = 0
    try:
        for start in range(0, len(images), args.batch_size):
            batch = [
                fo.Sample(filepath=str(path), relpath=relpath, tags=tags)
                for path, relpath in images[start:start + args.batch_size]
            ]
            dataset.add_samples(batch)
            confirmed += len(batch)
            logger.info("Imported %s/%s", confirmed, len(images))
        dataset.save()
    except (Exception, KeyboardInterrupt):
        try:
            logger.error("Import failed: dataset=%s persisted_samples=%s; partial dataset retained", args.dataset_name, len(dataset))
        except Exception:
            logger.error("Import failed: dataset=%s confirmed_samples=%s; actual count unavailable, last batch may be partially written", args.dataset_name, confirmed)
        raise
    print(f"imported_dataset={args.dataset_name} samples={len(dataset)}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        root = args.images_root.expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"Not a directory: {root}")
        print(f"mode={'dry-run' if args.dry_run else 'import'} dataset_name={args.dataset_name} images_root={root}")
        images = scan_images(root, args)
        if not images:
            raise ValueError("No valid images found; no dataset created")
        if args.dry_run:
            print("Database not accessed; dataset name availability checked during import")
        else:
            import_images(args, images)
        return 0
    except Exception as error:
        logger.error("%s", error)
        return 1
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
