"""Append sample tags by unique SHA-256, then unique perceptual hash."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
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
    parser.add_argument("--dataset", required=True, type=nonempty)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--sample-tags", required=True, type=tag_list)
    parser.add_argument("--recursive", action="store_true", help="Include subdirectories.")
    parser.add_argument("--dry-run", action="store_true", help="Write report without changing the database.")
    return parser.parse_args(argv)


def list_images(root: Path, recursive: bool) -> list[Path]:
    paths = root.rglob("*") if recursive else root.iterdir()
    return sorted(path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def normalize_hash(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def issue(
    kind: str,
    path: str = "",
    digest: str = "",
    sample_ids: str = "",
    detail: str = "",
    *,
    phash: str = "",
) -> dict[str, str]:
    return dict(issue=kind, filepath=path, sha256=digest, phash=phash, sample_ids=sample_ids, detail=detail)


def load_compute_phash():
    """Load the same pHash used by ``update_media.py``."""
    path = Path(__file__).resolve().parent / "update_media.py"
    spec = importlib.util.spec_from_file_location("update_media_for_tags", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_phash_hex


def build_plan(rows, images: list[Path], tags: list[str], compute_phash):
    """Return sample IDs to update, issues, matched samples, and image hit counts.

    SHA-256 is tried first. A missing SHA-256 falls back to pHash. Either key
    must belong to exactly one sample; duplicates are reported and skipped.
    """
    by_hash = defaultdict(list)
    by_phash = defaultdict(list)
    issues = []
    for sample_id, digest, phash, existing_tags in rows:
        sample_id = str(sample_id)
        digest = normalize_hash(digest)
        phash = normalize_hash(phash)
        stored_tags = existing_tags or []
        if not digest:
            issues.append(issue("missing_sha256", sample_ids=sample_id, phash=phash, detail="Run update_media.py first"))
        else:
            by_hash[digest].append((sample_id, stored_tags))
        if not phash:
            issues.append(issue("missing_phash", sample_ids=sample_id, digest=digest, detail="Run update_media.py first"))
        else:
            by_phash[phash].append((sample_id, stored_tags))

    matched = set()
    updates = set()
    sha256_hits = 0
    phash_hits = 0
    wanted = set(tags)

    def accept(candidates) -> None:
        sample_id, existing_tags = candidates[0]
        matched.add(sample_id)
        if not wanted.issubset(existing_tags):
            updates.add(sample_id)

    for index, path in enumerate(images, 1):
        try:
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
        except OSError as error:
            issues.append(issue("read_error", str(path), detail=str(error)))
            continue
        candidates = by_hash.get(digest, [])
        if len(candidates) > 1:
            issues.append(issue("ambiguous_sha256", str(path), digest, ",".join(item[0] for item in candidates)))
        elif candidates:
            accept(candidates)
            sha256_hits += 1
        else:
            try:
                phash = normalize_hash(compute_phash(path))
            except Exception as error:
                issues.append(issue("phash_error", str(path), digest, detail=str(error)))
                continue
            if not phash:
                issues.append(issue("phash_error", str(path), digest, detail="empty phash"))
                continue
            candidates = by_phash.get(phash, [])
            if len(candidates) > 1:
                issues.append(
                    issue("ambiguous_phash", str(path), digest, ",".join(item[0] for item in candidates), phash=phash)
                )
            elif candidates:
                accept(candidates)
                phash_hits += 1
            else:
                issues.append(issue("unmatched", str(path), digest, phash=phash))
        if index % 1000 == 0:
            logger.info("Hashed images %d/%d", index, len(images))
    return sorted(updates), issues, len(matched), sha256_hits, phash_hits


def write_issues(dataset_name: str, issues: list[dict[str, str]]) -> Path:
    safe_name = re.sub(r"[^\w.-]", "_", dataset_name)
    output = Path.cwd() / "tmp" / f"update_tags_{safe_name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["issue", "filepath", "sha256", "phash", "sample_ids", "detail"]
        )
        writer.writeheader()
        writer.writerows(issues)
    return output


def run(args: argparse.Namespace, fo, compute_phash=None) -> int:
    root = args.images_dir.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    if not fo.dataset_exists(args.dataset):
        raise ValueError(f"Dataset does not exist: {args.dataset}")
    dataset = fo.load_dataset(args.dataset)
    schema = dataset.get_field_schema()
    missing = [name for name in ("sha256", "phash") if name not in schema]
    if missing:
        raise ValueError(f"Dataset has no {', '.join(missing)} field; run scripts/update_media.py first")
    images = list_images(root, args.recursive)
    if compute_phash is None:
        loaded = None

        def compute_phash(path):
            nonlocal loaded
            if loaded is None:
                loaded = load_compute_phash()
            return loaded(path)

    ids, hashes, phashes, tags = dataset.values(["id", "sha256", "phash", "tags"])
    updates, issues, matched, sha256_hits, phash_hits = build_plan(
        zip(ids, hashes, phashes, tags), images, args.sample_tags, compute_phash
    )
    report = write_issues(args.dataset, issues)
    print(f"mode={'dry-run' if args.dry_run else 'update'}")
    print(f"dataset={args.dataset}")
    print(f"scanned_images={len(images)}")
    print(f"sha256_hits={sha256_hits}")
    print(f"phash_hits={phash_hits}")
    print(f"matched_samples={matched}")
    print(f"to_update={len(updates)}")
    print(f"unchanged={matched - len(updates)}")
    print(f"issues={len(issues)}")
    print(f"csv_path={report}")
    if not args.dry_run:
        for sample_id in updates:
            sample = dataset[sample_id]
            sample.tags = list(dict.fromkeys([*(sample.tags or []), *args.sample_tags]))
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
