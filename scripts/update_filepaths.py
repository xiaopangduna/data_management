"""Update FiftyOne filepaths by matching image SHA-256 hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import fiftyone as fo

logger = logging.getLogger(__name__)
IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
HASH_CHUNK_SIZE = 1024 * 1024
LOG_INTERVAL = 1000
REPORT_SUBDIR = "tmp"
HASH_WORKERS = min(8, os.cpu_count() or 1)


@dataclass(frozen=True)
class PathUpdate:
    sample_id: str
    old_path: str
    new_path: str


def nonempty(value: str) -> str:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True, type=nonempty)
    parser.add_argument(
        "--images-dir",
        required=True,
        type=Path,
        help="Recursively scan this directory and match files by SHA-256.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without updating FiftyOne.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> tuple[str, Path]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest(), path


def list_images(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def hash_index(images: list[Path]) -> dict[str, list[str]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=HASH_WORKERS) as executor:
        for index, (digest, path) in enumerate(executor.map(sha256_file, images), 1):
            by_hash[digest].append(str(path.resolve()))
            if index % LOG_INTERVAL == 0:
                logger.info("Hashed images %d/%d", index, len(images))
    return by_hash


def build_plan(
    rows: list[tuple[str, str, str | None]],
    by_hash: dict[str, list[str]],
) -> tuple[list[PathUpdate], list[dict[str, str]], int, int]:
    candidates: list[PathUpdate] = []
    issues: list[dict[str, str]] = []
    unchanged = 0
    ignored = 0
    registered_by_path: dict[str, set[str]] = defaultdict(set)
    for sample_id, filepath, _ in rows:
        registered_by_path[filepath].add(sample_id)

    for sample_id, old_path, digest in rows:
        if not digest:
            ignored += 1
            continue
        targets = sorted(set(by_hash.get(digest, [])))
        if not targets:
            ignored += 1
            continue
        if old_path in targets:
            unchanged += 1
            continue
        if len(targets) == 1:
            target = targets[0]
        else:
            same_name = [
                target for target in targets
                if Path(target).name == Path(old_path).name
            ]
            if len(same_name) != 1:
                issues.append(
                    issue("ambiguous_image_hash", sample_id, old_path, ", ".join(targets))
                )
                continue
            target = same_name[0]
        other_ids = registered_by_path.get(target, set()) - {sample_id}
        if other_ids:
            issues.append(
                issue("target_already_registered", sample_id, old_path, f"{target}: {','.join(sorted(other_ids))}")
            )
            continue
        candidates.append(PathUpdate(sample_id, old_path, target))

    target_counts = Counter(item.new_path for item in candidates)
    updates = []
    for item in candidates:
        if target_counts[item.new_path] > 1:
            issues.append(issue("sample_hash_collision", item.sample_id, item.old_path, item.new_path))
        else:
            updates.append(item)
    return updates, issues, unchanged, ignored


def issue(kind: str, sample_id: str, old_path: str, detail: str) -> dict[str, str]:
    return {"issue": kind, "sample_id": sample_id, "old_path": old_path, "detail": detail}


def write_issues(dataset_name: str, rows: list[dict[str, str]]) -> Path:
    output = Path.cwd() / REPORT_SUBDIR / f"update_filepaths_{dataset_name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["issue", "sample_id", "old_path", "detail"]
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["issue"], row["old_path"])))
    return output


def apply_updates(dataset: fo.Dataset, updates: list[PathUpdate]) -> None:
    for index, item in enumerate(updates, 1):
        sample = dataset[item.sample_id]
        sample.filepath = item.new_path
        sample.save()
        if index % LOG_INTERVAL == 0:
            logger.info("Updated paths %d/%d", index, len(updates))
    dataset.save()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)7s] %(name)s:%(lineno)d - %(message)s",
    )
    args = parse_args(argv)
    try:
        if not fo.dataset_exists(args.dataset_name):
            raise ValueError(f"Dataset does not exist: {args.dataset_name}")
        images_dir = args.images_dir.expanduser().resolve(strict=True)
        if not images_dir.is_dir():
            raise ValueError(f"Not a directory: {images_dir}")

        images = list_images(images_dir)
        logger.info("Found %d images; computing SHA-256", len(images))
        by_hash = hash_index(images)
        dataset = fo.load_dataset(args.dataset_name)
        ids, paths, hashes = dataset.values(["id", "filepath", "sha256"])
        rows = [
            (str(sample_id), str(filepath), str(digest) if digest else None)
            for sample_id, filepath, digest in zip(ids, paths, hashes)
        ]
        updates, issues, unchanged, ignored = build_plan(rows, by_hash)
        report = write_issues(args.dataset_name, issues)

        print(f"mode={'dry-run' if args.dry_run else 'update'}")
        print(f"dataset_name={args.dataset_name}")
        print(f"images_dir={images_dir}")
        print(f"dataset_samples={len(rows)}")
        print(f"indexed_images={len(images)}")
        print(f"unique_hashes={len(by_hash)}")
        print(f"to_update={len(updates)}")
        print(f"unchanged={unchanged}")
        print(f"issues={len(issues)}")
        print(f"ignored_not_in_images_dir={ignored}")
        print(f"csv_path={report}")
        if args.dry_run:
            return 0
        apply_updates(dataset, updates)
        print(f"updated={len(updates)}")
        return 0
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except Exception as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
