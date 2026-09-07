"""Fill media fields on an existing FiftyOne dataset.

Computes sha256, a 64-bit perceptual difference hash (stored as ``phash``),
and FiftyOne ``metadata`` (width, height, size, mime). Does not change
filepaths, detections, or tags. Does not create or delete datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"glob2(\.|$)",
)

import fiftyone as fo
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "coco2017"
SUPPORTED_HASH_TYPES = frozenset({"sha256", "phash"})
SHA256_FIELD = "sha256"
PHASH_FIELD = "phash"
LOG_INTERVAL = 2000
SHA256_CHUNK_SIZE = 1024 * 1024


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for media enrichment.

    Args:
        argv: Optional argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Add hashes and image metadata to an existing FiftyOne dataset."
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Existing FiftyOne dataset name.",
    )
    parser.add_argument(
        "--hashes",
        default="sha256,phash",
        help="Comma-separated: sha256, phash, or none.",
    )
    parser.add_argument(
        "--metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute FiftyOne metadata (use --no-metadata to skip).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute fields that already have values.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count work only; do not write fields.",
    )
    return parser.parse_args(argv)


def parse_hash_types(raw_hash_types: str) -> list[str]:
    """Split and validate ``--hashes``.

    Args:
        raw_hash_types: Comma-separated hash names, or ``none``.

    Returns:
        Hash type names to compute.

    Raises:
        ValueError: If a name is not supported.
    """
    stripped = raw_hash_types.strip()
    if stripped.lower() in {"", "none"}:
        return []
    hash_types = [part.strip() for part in stripped.split(",") if part.strip()]
    unsupported = sorted(set(hash_types) - SUPPORTED_HASH_TYPES)
    if unsupported:
        supported = ", ".join(sorted(SUPPORTED_HASH_TYPES))
        raise ValueError(f"Unsupported --hashes {unsupported}; allowed: {supported}, none")
    return hash_types


def sample_field_is_filled(sample: fo.Sample, field_name: str) -> bool:
    """Return True when a sample already stores a non-empty field.

    Args:
        sample: FiftyOne sample.
        field_name: Field to inspect.

    Returns:
        Whether the field exists and is non-empty.
    """
    if not sample.has_field(field_name):
        return False
    value = sample[field_name]
    return value is not None and value != ""


def metadata_is_filled(sample: fo.Sample) -> bool:
    """Return True when image width/height metadata is present.

    Args:
        sample: FiftyOne sample.

    Returns:
        Whether ``metadata.width`` is available.
    """
    metadata = sample.metadata if sample.has_field("metadata") else None
    return metadata is not None and getattr(metadata, "width", None) is not None


def compute_sha256_hex(file_path: Path) -> str:
    """Hash file bytes with SHA-256.

    Args:
        file_path: Path to the image file.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(SHA256_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def compute_phash_hex(file_path: Path) -> str:
    """Compute a 64-bit difference hash (near-duplicate / perceptual).

    Args:
        file_path: Path to the image file.

    Returns:
        16-character hex string.
    """
    with Image.open(file_path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        row_pixels = pixels[row * 9 : (row + 1) * 9]
        for index in range(8):
            value = (value << 1) | int(row_pixels[index] > row_pixels[index + 1])
    return f"{value:016x}"


def count_enrichment_work(
    dataset: fo.Dataset,
    hash_types: list[str],
    compute_metadata: bool,
    overwrite: bool,
) -> dict[str, int]:
    """Count samples that would be updated.

    Args:
        dataset: Existing dataset.
        hash_types: Hash fields to consider.
        compute_metadata: Whether metadata is requested.
        overwrite: Whether filled fields would be recomputed.

    Returns:
        Counts keyed by field name, plus missing files.
    """
    counts = {
        SHA256_FIELD: 0,
        PHASH_FIELD: 0,
        "metadata": 0,
        "missing_files": 0,
        "samples": len(dataset),
    }
    for sample in dataset.iter_samples():
        if not Path(sample.filepath).is_file():
            counts["missing_files"] += 1
            continue
        if "sha256" in hash_types and (overwrite or not sample_field_is_filled(sample, SHA256_FIELD)):
            counts[SHA256_FIELD] += 1
        if "phash" in hash_types and (overwrite or not sample_field_is_filled(sample, PHASH_FIELD)):
            counts[PHASH_FIELD] += 1
        if compute_metadata and (overwrite or not metadata_is_filled(sample)):
            counts["metadata"] += 1
    return counts


def enrich_sample_hashes(
    sample: fo.Sample,
    file_path: Path,
    hash_types: list[str],
    overwrite: bool,
) -> list[str]:
    """Write hash fields onto one sample.

    Args:
        sample: Sample to update.
        file_path: Image path on disk.
        hash_types: Requested hash kinds.
        overwrite: Recompute existing values.

    Returns:
        Names of fields written.
    """
    written: list[str] = []
    if "sha256" in hash_types and (overwrite or not sample_field_is_filled(sample, SHA256_FIELD)):
        sample[SHA256_FIELD] = compute_sha256_hex(file_path)
        written.append(SHA256_FIELD)
    if "phash" in hash_types and (overwrite or not sample_field_is_filled(sample, PHASH_FIELD)):
        sample[PHASH_FIELD] = compute_phash_hex(file_path)
        written.append(PHASH_FIELD)
    return written


def enrich_dataset(
    dataset: fo.Dataset,
    hash_types: list[str],
    overwrite: bool,
) -> tuple[int, int]:
    """Compute hashes for samples that need them.

    Args:
        dataset: Existing dataset.
        hash_types: Requested hash kinds.
        overwrite: Recompute existing values.

    Returns:
        Tuple of (updated_sample_count, missing_file_count).
    """
    updated_count = 0
    missing_count = 0
    for sample_index, sample in enumerate(dataset.iter_samples(autosave=True), start=1):
        file_path = Path(sample.filepath)
        if not file_path.is_file():
            missing_count += 1
            logger.warning("Missing file: %s", file_path)
            continue
        if enrich_sample_hashes(sample, file_path, hash_types, overwrite):
            updated_count += 1
        if sample_index % LOG_INTERVAL == 0:
            logger.info("Hashed %s/%s samples", sample_index, len(dataset))
    return updated_count, missing_count


def print_enrich_report(
    dataset_name: str,
    hash_types: list[str],
    compute_metadata: bool,
    overwrite: bool,
    dry_run: bool,
    counts: dict[str, int],
) -> None:
    """Print a summary of planned or completed enrichment.

    Args:
        dataset_name: Dataset name.
        hash_types: Hash kinds requested.
        compute_metadata: Whether metadata was requested.
        overwrite: Whether existing values are replaced.
        dry_run: Whether nothing was written.
        counts: Work counters.
    """
    print(f"mode={'dry-run' if dry_run else 'enrich'}")
    print(f"dataset_name={dataset_name}")
    print(f"hashes={','.join(hash_types) if hash_types else 'none'}")
    print(f"metadata={compute_metadata}")
    print(f"overwrite={overwrite}")
    print(f"samples={counts['samples']}")
    print(f"sha256_to_write={counts[SHA256_FIELD]}")
    print(f"phash_to_write={counts[PHASH_FIELD]}")
    print(f"metadata_to_write={counts['metadata']}")
    print(f"missing_files={counts['missing_files']}")


def run_enrich(
    dataset: fo.Dataset,
    hash_types: list[str],
    compute_metadata: bool,
    overwrite: bool,
) -> None:
    """Write metadata and hashes onto ``dataset``.

    Args:
        dataset: Existing dataset.
        hash_types: Hash kinds to compute.
        compute_metadata: Whether to run ``compute_metadata``.
        overwrite: Recompute existing values.
    """
    print(f"mode=enrich")
    print(f"dataset_name={dataset.name}")
    print(f"samples={len(dataset)}")
    if compute_metadata:
        logger.info("Computing metadata overwrite=%s", overwrite)
        dataset.compute_metadata(overwrite=overwrite, skip_failures=True)
    if hash_types:
        updated_count, missing_count = enrich_dataset(dataset, hash_types, overwrite)
        print(f"hash_samples_updated={updated_count}")
        print(f"hash_missing_files={missing_count}")
    dataset.save()
    print("enrich_done=true")


def main(argv: list[str] | None = None) -> int:
    """Run dry-run counting or write hashes and metadata.

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
    try:
        hash_types = parse_hash_types(args.hashes)
    except ValueError as error:
        logger.error("%s", error)
        return 1
    if not fo.dataset_exists(args.dataset_name):
        logger.error("Dataset does not exist: %s", args.dataset_name)
        return 1
    dataset = fo.load_dataset(args.dataset_name)
    if args.dry_run:
        counts = count_enrichment_work(
            dataset, hash_types, args.metadata, args.overwrite
        )
        print_enrich_report(
            args.dataset_name, hash_types, args.metadata, args.overwrite, True, counts
        )
        return 0
    run_enrich(dataset, hash_types, args.metadata, args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
