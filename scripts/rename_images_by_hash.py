"""Rename images in a directory using their SHA-256 content hash."""

from __future__ import annotations

import argparse
import hashlib
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class Rename:
    source: Path
    target: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing the images")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories in place")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without renaming files")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def scan_images(directory: Path, recursive: bool) -> list[Path]:
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path for path in candidates
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_plan(images: list[Path]) -> list[Rename]:
    groups: dict[tuple[Path, str, str], list[Path]] = defaultdict(list)
    for image in images:
        groups[(image.parent, sha256_file(image), image.suffix.lower())].append(image)

    plan: list[Rename] = []
    for (parent, digest, suffix), sources in sorted(groups.items(), key=lambda item: str(item[0])):
        canonical = parent / f"{digest}{suffix}"
        sources.sort(key=lambda path: (path != canonical, path.name))
        for index, source in enumerate(sources, start=1):
            name = f"{digest}{suffix}" if index == 1 else f"{digest}-{index}{suffix}"
            target = parent / name
            if source != target:
                plan.append(Rename(source, target))
    return plan


def apply_plan(plan: list[Rename]) -> None:
    """Rename in two phases so existing names cannot be overwritten."""
    staged: list[tuple[Path, Path, Path]] = []
    try:
        for item in plan:
            temp = item.source.with_name(f".{item.source.name}.hash-rename-{uuid.uuid4().hex}.tmp")
            item.source.rename(temp)
            staged.append((temp, item.source, item.target))
        for index, (temp, source, target) in enumerate(staged):
            temp.rename(target)
            staged[index] = (target, source, target)
    except Exception:
        for current, source, _ in reversed(staged):
            if current.exists() and not source.exists():
                current.rename(source)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        directory = args.directory.expanduser().resolve(strict=True)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        images = scan_images(directory, args.recursive)
        plan = build_plan(images)
        for item in plan:
            print(f"{item.source} -> {item.target}")
        print(f"images={len(images)} renamed={len(plan)} unchanged={len(images) - len(plan)}")
        if args.dry_run:
            print("Dry run: no files renamed.")
        else:
            apply_plan(plan)
        return 0
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Rename failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
