"""Append sample tags in an existing FiftyOne dataset using SHA-256 only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def nonempty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be empty")
    return value.strip()


def tag_list(value: str) -> list[str]:
    tags = list(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not tags:
        raise argparse.ArgumentTypeError("provide at least one tag")
    return tags


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True, type=nonempty)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--tags", required=True, type=tag_list)
    parser.add_argument("--recursive", action="store_true", help="Include subdirectories.")
    parser.add_argument("--dry-run", action="store_true", help="Write report without changing the database.")
    return parser.parse_args(argv)


def list_images(root: Path, recursive: bool) -> list[Path]:
    paths = root.rglob("*") if recursive else root.iterdir()
    return sorted(path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def issue(kind: str, path: str = "", digest: str = "", sample_ids: str = "", detail: str = "") -> dict[str, str]:
    return dict(issue=kind, filepath=path, sha256=digest, sample_ids=sample_ids, detail=detail)


def build_plan(rows, images: list[Path], tags: list[str]):
    """Return unique sample IDs to update, issues, and unique matched sample count."""
    by_hash = defaultdict(list)
    issues = []
    for sample_id, digest, existing_tags in rows:
        if not digest:
            issues.append(issue("missing_sha256", sample_ids=str(sample_id), detail="Run update_media.py first"))
        else:
            by_hash[digest].append((str(sample_id), existing_tags or []))

    matched = set()
    updates = set()
    for index, path in enumerate(images, 1):
        try:
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
        except OSError as error:
            issues.append(issue("read_error", str(path), detail=str(error)))
            continue
        candidates = by_hash.get(digest, [])
        if not candidates:
            issues.append(issue("unmatched", str(path), digest))
        elif len(candidates) > 1:
            issues.append(issue("ambiguous_sha256", str(path), digest, ",".join(item[0] for item in candidates)))
        else:
            sample_id, existing_tags = candidates[0]
            matched.add(sample_id)
            if not set(tags).issubset(existing_tags):
                updates.add(sample_id)
        if index % 1000 == 0:
            logger.info("Hashed images %d/%d", index, len(images))
    return sorted(updates), issues, len(matched)


def write_issues(dataset_name: str, issues: list[dict[str, str]]) -> Path:
    safe_name = re.sub(r"[^\w.-]", "_", dataset_name)
    output = Path.cwd() / "tmp" / f"update_tags_{safe_name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["issue", "filepath", "sha256", "sample_ids", "detail"])
        writer.writeheader()
        writer.writerows(issues)
    return output


def run(args: argparse.Namespace, fo) -> int:
    root = args.images_dir.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    if not fo.dataset_exists(args.dataset_name):
        raise ValueError(f"Dataset does not exist: {args.dataset_name}")
    dataset = fo.load_dataset(args.dataset_name)
    if "sha256" not in dataset.get_field_schema():
        raise ValueError("Dataset has no sha256 field; run scripts/update_media.py first")
    images = list_images(root, args.recursive)
    ids, hashes, tags = dataset.values(["id", "sha256", "tags"])
    updates, issues, matched = build_plan(zip(ids, hashes, tags), images, args.tags)
    report = write_issues(args.dataset_name, issues)
    print(f"mode={'dry-run' if args.dry_run else 'update'}")
    print(f"dataset_name={args.dataset_name}")
    print(f"scanned_images={len(images)}")
    print(f"matched_samples={matched}")
    print(f"to_update={len(updates)}")
    print(f"unchanged={matched - len(updates)}")
    print(f"issues={len(issues)}")
    print(f"csv_path={report}")
    if not args.dry_run:
        for sample_id in updates:
            sample = dataset[sample_id]
            sample.tags = list(dict.fromkeys([*(sample.tags or []), *args.tags]))
            sample.save()
        print(f"updated={len(updates)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    try:
        import fiftyone as fo

        return run(args, fo)
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except Exception as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
