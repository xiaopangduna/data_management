"""Classify folder images before import: validate, batch dedupe, optional dataset match.

Without ``--dataset``, only checks the folder itself (corrupt + batch sha256
duplicates). With ``--dataset``, also matches FiftyOne by ``sha256`` / ``phash``.
Writes a CSV under ``tmp/`` and copies into
``--out-dir/{new,exact_dup,near_dup,batch_dup,corrupt}/``. Outputs use
``{sha256}{ext}`` flat names by default. Does not modify the dataset.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_SUFFIXES = (".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp")
IMAGE_SUFFIXES = frozenset(DEFAULT_SUFFIXES)
STATUS_NEW = "new"
STATUS_EXACT = "exact_dup"
STATUS_NEAR = "near_dup"
STATUS_BATCH = "batch_dup"
STATUS_CORRUPT = "corrupt"
ALL_STATUSES = frozenset(
    {STATUS_NEW, STATUS_EXACT, STATUS_NEAR, STATUS_BATCH, STATUS_CORRUPT}
)
CSV_COLUMNS = (
    "status",
    "filepath",
    "sha256",
    "phash",
    "sample_ids",
    "sample_filepaths",
    "out_filepath",
    "canonical_sha256_name",
    "issue",
    "detail",
)
FOLDER_REPORT_NAME = "folder"


def nonempty(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("must not be empty")
    return stripped


def nonempty_path(value: str) -> Path:
    return Path(nonempty(value))


def parse_suffixes(value: str) -> frozenset[str]:
    """Parse comma-separated image suffixes; must be a subset of supported images."""
    parts = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("provide at least one suffix")
    suffixes: set[str] = set()
    for part in parts:
        suffix = part if part.startswith(".") else f".{part}"
        if suffix not in IMAGE_SUFFIXES:
            allowed = ",".join(sorted(IMAGE_SUFFIXES))
            raise argparse.ArgumentTypeError(
                f"unsupported suffix {suffix}; allowed: {allowed}"
            )
        suffixes.add(suffix)
    return frozenset(suffixes)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=None,
        type=nonempty,
        help="FiftyOne dataset to match against. Omit to only check the folder itself.",
    )
    parser.add_argument("--images-dir", required=True, type=nonempty_path)
    parser.add_argument("--out-dir", required=True, type=nonempty_path)
    parser.add_argument(
        "--suffixes",
        default=",".join(DEFAULT_SUFFIXES),
        type=parse_suffixes,
        help="Comma-separated image suffixes to scan (default: common image types).",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan subdirectories (default: on). Use --no-recursive for top level only.",
    )
    parser.add_argument(
        "--rename-by-hash",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Name outputs {sha256}{ext} under each status folder (default: on).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write CSV only; do not write media.")
    return parser.parse_args(argv)


def load_update_media():
    """Load ``update_media`` so sha256/phash match the dataset enrichment script."""
    path = Path(__file__).resolve().parent / "update_media.py"
    spec = importlib.util.spec_from_file_location("update_media", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_images(root: Path, recursive: bool, suffixes: frozenset[str]) -> list[Path]:
    paths = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        path for path in paths if path.is_file() and path.suffix.lower() in suffixes
    )


def normalize_hash(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def validate_image(path: Path) -> str | None:
    """Return an error string when the image cannot fully decode, else None."""
    try:
        with Image.open(path) as image:
            image.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        return str(error)
    return None


def build_hash_indexes(
    ids: list[object],
    sha256s: list[object],
    phashes: list[object],
    filepaths: list[object],
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, list[tuple[str, str]]], int, int]:
    """Return sha256/phash indexes and counts of samples missing each field."""
    by_sha: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_phash: dict[str, list[tuple[str, str]]] = defaultdict(list)
    missing_sha256 = 0
    missing_phash = 0
    for sample_id, digest, phash, filepath in zip(ids, sha256s, phashes, filepaths, strict=True):
        sid = str(sample_id)
        path = str(filepath or "")
        sha = normalize_hash(digest)
        pha = normalize_hash(phash)
        if not sha:
            missing_sha256 += 1
        else:
            by_sha[sha].append((sid, path))
        if not pha:
            missing_phash += 1
        else:
            by_phash[pha].append((sid, path))
    return by_sha, by_phash, missing_sha256, missing_phash


def hash_filename(sha256: str, path: Path) -> str:
    """Return ``{sha256}{lowercase-ext}``."""
    return f"{sha256}{path.suffix.lower()}"


def destination_for(
    out_dir: Path,
    path: Path,
    status: str,
    *,
    rename_by_hash: bool,
    sha256: str,
) -> Path:
    """Place under ``out-dir/<status>/`` as a flat file (hash name when enabled)."""
    status_root = out_dir / status
    if rename_by_hash and sha256:
        return status_root / hash_filename(sha256, path)
    return status_root / path.name


def empty_row(
    path: Path,
    *,
    status: str,
    sha256: str = "",
    phash: str = "",
    out_filepath: str = "",
    canonical_sha256_name: str = "",
    issue: str = "",
    detail: str = "",
    sample_ids: str = "",
    sample_filepaths: str = "",
) -> dict[str, str]:
    return {
        "status": status,
        "filepath": str(path),
        "sha256": sha256,
        "phash": phash,
        "sample_ids": sample_ids,
        "sample_filepaths": sample_filepaths,
        "out_filepath": out_filepath,
        "canonical_sha256_name": canonical_sha256_name,
        "issue": issue,
        "detail": detail,
    }


def classify_against_dataset(
    sha256: str,
    phash: str,
    by_sha: dict[str, list[tuple[str, str]]],
    by_phash: dict[str, list[tuple[str, str]]],
    *,
    match_dataset: bool,
) -> tuple[str, list[tuple[str, str]], str]:
    """Return status, hits, and detail for dataset matching."""
    if not match_dataset:
        return STATUS_NEW, [], ""
    exact_hits = by_sha.get(sha256, [])
    if exact_hits:
        detail = "ambiguous_sha256" if len(exact_hits) > 1 else ""
        return STATUS_EXACT, exact_hits, detail
    if phash:
        near_hits = by_phash.get(phash, [])
        if near_hits:
            detail = "ambiguous_phash" if len(near_hits) > 1 else ""
            return STATUS_NEAR, near_hits, detail
    return STATUS_NEW, [], ""


def classify_image(
    path: Path,
    sha256: str,
    phash: str,
    by_sha: dict[str, list[tuple[str, str]]],
    by_phash: dict[str, list[tuple[str, str]]],
    out_dir: Path,
    *,
    rename_by_hash: bool,
    match_dataset: bool,
) -> dict[str, str]:
    """Classify one validated image against the dataset indexes."""
    status, hits, detail = classify_against_dataset(
        sha256, phash, by_sha, by_phash, match_dataset=match_dataset
    )
    canonical = hash_filename(sha256, path) if sha256 else ""
    out_path = destination_for(
        out_dir, path, status, rename_by_hash=rename_by_hash, sha256=sha256
    )
    return empty_row(
        path,
        status=status,
        sha256=sha256,
        phash=phash,
        sample_ids=",".join(item[0] for item in hits),
        sample_filepaths=",".join(item[1] for item in hits),
        out_filepath=str(out_path),
        canonical_sha256_name=canonical,
        issue=detail if detail in {"ambiguous_sha256", "ambiguous_phash"} else "",
        detail=detail,
    )


def hash_and_classify(
    images: list[Path],
    by_sha: dict[str, list[tuple[str, str]]],
    by_phash: dict[str, list[tuple[str, str]]],
    out_dir: Path,
    compute_sha256,
    compute_phash,
    *,
    rename_by_hash: bool = True,
    match_dataset: bool = True,
) -> list[dict[str, str]]:
    """Validate, hash, detect batch dups, then optionally classify against the dataset."""
    rows: list[dict[str, str]] = []
    seen_sha256: dict[str, Path] = {}
    for index, path in enumerate(images, 1):
        sha256 = ""
        try:
            sha256 = normalize_hash(compute_sha256(path))
        except OSError as error:
            rows.append(
                empty_row(
                    path,
                    status=STATUS_CORRUPT,
                    issue="read_error",
                    detail=str(error),
                )
            )
            continue

        validate_error = validate_image(path)
        if validate_error is not None:
            canonical = hash_filename(sha256, path) if sha256 else ""
            out_path = destination_for(
                out_dir,
                path,
                STATUS_CORRUPT,
                rename_by_hash=rename_by_hash,
                sha256=sha256,
            )
            issue = "truncated" if "truncated" in validate_error.lower() else "decode_error"
            rows.append(
                empty_row(
                    path,
                    status=STATUS_CORRUPT,
                    sha256=sha256,
                    out_filepath=str(out_path),
                    canonical_sha256_name=canonical,
                    issue=issue,
                    detail=validate_error,
                )
            )
            continue

        phash = ""
        if match_dataset:
            try:
                phash = normalize_hash(compute_phash(path))
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
                canonical = hash_filename(sha256, path) if sha256 else ""
                out_path = destination_for(
                    out_dir,
                    path,
                    STATUS_CORRUPT,
                    rename_by_hash=rename_by_hash,
                    sha256=sha256,
                )
                rows.append(
                    empty_row(
                        path,
                        status=STATUS_CORRUPT,
                        sha256=sha256,
                        out_filepath=str(out_path),
                        canonical_sha256_name=canonical,
                        issue="phash_error",
                        detail=str(error),
                    )
                )
                continue

        if sha256 in seen_sha256:
            first = seen_sha256[sha256]
            canonical = hash_filename(sha256, path)
            out_path = destination_for(
                out_dir,
                path,
                STATUS_BATCH,
                rename_by_hash=rename_by_hash,
                sha256=sha256,
            )
            rows.append(
                empty_row(
                    path,
                    status=STATUS_BATCH,
                    sha256=sha256,
                    phash=phash,
                    out_filepath=str(out_path),
                    canonical_sha256_name=canonical,
                    issue="batch_dup",
                    detail=str(first),
                )
            )
            continue

        seen_sha256[sha256] = path
        rows.append(
            classify_image(
                path,
                sha256,
                phash,
                by_sha,
                by_phash,
                out_dir,
                rename_by_hash=rename_by_hash,
                match_dataset=match_dataset,
            )
        )
        if index % 1000 == 0:
            logger.info("Hashed images %d/%d", index, len(images))
    return rows


def plan_materialize(
    rows: list[dict[str, str]],
) -> tuple[list[tuple[Path, Path]], list[dict[str, str]], int]:
    """Mark dest collisions on rows; return (source, dest) pairs still to write."""
    to_write: list[tuple[Path, Path]] = []
    skipped = 0
    for row in rows:
        if row["status"] not in ALL_STATUSES or not row["out_filepath"]:
            continue
        source = Path(row["filepath"])
        dest = Path(row["out_filepath"])
        if dest.exists() or dest.is_symlink():
            detail = row["detail"]
            row["detail"] = f"{detail};dest_exists" if detail else "dest_exists"
            if row["issue"]:
                row["issue"] = f"{row['issue']};dest_exists"
            else:
                row["issue"] = "dest_exists"
            skipped += 1
            continue
        to_write.append((source, dest))
    return to_write, rows, skipped


def apply_materialize(to_write: list[tuple[Path, Path]]) -> int:
    """Copy sources to destinations. Returns number written."""
    written = 0
    for source, dest in to_write:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            continue
        shutil.copy2(source, dest)
        written += 1
    return written


def write_report(dataset_name: str, rows: list[dict[str, str]]) -> Path:
    safe_name = re.sub(r"[^\w.-]", "_", dataset_name)
    output = Path.cwd() / "tmp" / f"check_image_dupes_{safe_name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def count_status(rows: list[dict[str, str]], status: str) -> int:
    return sum(1 for row in rows if row["status"] == status)


def run(args: argparse.Namespace, fo=None, compute_sha256=None, compute_phash=None) -> int:
    images_dir = args.images_dir.expanduser().resolve(strict=True)
    if not images_dir.is_dir():
        raise ValueError(f"Not a directory: {images_dir}")
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"out-dir is not a directory: {out_dir}")

    match_dataset = bool(args.dataset)
    missing_sha256 = 0
    missing_phash = 0
    by_sha: dict[str, list[tuple[str, str]]] = {}
    by_phash: dict[str, list[tuple[str, str]]] = {}

    if match_dataset:
        if fo is None:
            raise ValueError("FiftyOne is required when --dataset is set")
        if not fo.dataset_exists(args.dataset):
            raise ValueError(f"Dataset does not exist: {args.dataset}")
        dataset = fo.load_dataset(args.dataset)
        schema = dataset.get_field_schema()
        if "sha256" not in schema or "phash" not in schema:
            raise ValueError("Dataset missing sha256/phash; run scripts/update_media.py first")
        ids, sha256s, phashes, filepaths = dataset.values(
            ["id", "sha256", "phash", "filepath"]
        )
        by_sha, by_phash, missing_sha256, missing_phash = build_hash_indexes(
            ids, sha256s, phashes, filepaths
        )

    if compute_sha256 is None or (match_dataset and compute_phash is None):
        media = load_update_media()
        if compute_sha256 is None:
            compute_sha256 = media.compute_sha256_hex
        if match_dataset and compute_phash is None:
            compute_phash = media.compute_phash_hex
    if compute_phash is None:
        compute_phash = lambda _path: ""

    images = list_images(images_dir, args.recursive, args.suffixes)
    rows = hash_and_classify(
        images,
        by_sha,
        by_phash,
        out_dir,
        compute_sha256,
        compute_phash,
        rename_by_hash=args.rename_by_hash,
        match_dataset=match_dataset,
    )
    to_write, rows, skipped_dest = plan_materialize(rows)
    report_name = args.dataset if match_dataset else FOLDER_REPORT_NAME
    report = write_report(report_name, rows)

    print(f"mode={'dry-run' if args.dry_run else 'write'}")
    print(f"dataset={args.dataset or 'none'}")
    print(f"scanned={len(images)}")
    print(f"new={count_status(rows, STATUS_NEW)}")
    print(f"exact_dup={count_status(rows, STATUS_EXACT)}")
    print(f"near_dup={count_status(rows, STATUS_NEAR)}")
    print(f"batch_dup={count_status(rows, STATUS_BATCH)}")
    print(f"corrupt={count_status(rows, STATUS_CORRUPT)}")
    if match_dataset:
        print(f"missing_sha256={missing_sha256}")
        print(f"missing_phash={missing_phash}")
    print(f"planned_writes={len(to_write)}")
    print(f"skipped_dest_exists={skipped_dest}")
    print(f"csv_path={report}")
    if args.dry_run:
        print("written=0")
    else:
        written = apply_materialize(to_write)
        print(f"written={written}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    try:
        if args.dataset:
            import fiftyone as fo

            return run(args, fo)
        return run(args, fo=None)
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except Exception as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
