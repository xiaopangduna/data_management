"""Delete exact duplicates and tag near-duplicates on an existing FiftyOne dataset.

Exact matches share ``sha256``: keep one sample, delete the rest from the
dataset (not from disk). Near matches share a 64-bit pHash within
``--hamming-max``: keep one sample untagged, tag the rest ``dup_near``, and
write ``dup_group`` / ``dup_of``. Every run writes CSVs under ``tmp/`` in the
current working directory (including ``--dry-run``): a full archive, an exact
file, and slim near shards of 200 groups each. Does not create or delete
datasets.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"glob2(\.|$)",
)

import fiftyone as fo

logger = logging.getLogger(__name__)

SHA256_FIELD = "sha256"
PHASH_FIELD = "phash"
DUP_NEAR_TAG = "dup_near"
DUP_GROUP_FIELD = "dup_group"
DUP_OF_FIELD = "dup_of"
DEFAULT_HAMMING_MAX = 2
PHASH_BITS = 64
REPORT_COLUMNS = (
    "dataset_name",
    "kind",
    "action",
    "dup_group",
    "sample_id",
    "filepath",
    "relpath",
    "sha256",
    "phash",
    "kept_sample_id",
    "kept_filepath",
    "kept_relpath",
)
SLIM_COLUMNS = (
    "kind",
    "action",
    "relpath",
    "kept_relpath",
    "dup_group",
)
NEAR_GROUPS_PER_FILE = 200
REPORT_SUBDIR = "tmp"


@dataclass(frozen=True)
class SampleRef:
    """Lightweight sample fields used for grouping."""

    id: str
    filepath: str
    relpath: str
    sha256: str
    phash: str
    area: int
    has_dup_near: bool
    has_dup_group: bool


@dataclass(frozen=True)
class ExactDeleteRow:
    """One deleted exact-duplicate sample."""

    sha256: str
    kept_filepath: str
    kept_relpath: str
    deleted_id: str
    deleted_filepath: str
    deleted_relpath: str


def nonempty(value: str) -> str:
    """Reject blank CLI strings."""
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("must not be empty")
    return stripped


def nonnegative_int(value: str) -> int:
    """Parse an integer that must be >= 0."""
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for dataset deduplication."""
    parser = argparse.ArgumentParser(
        description="Delete exact duplicates and tag near-duplicates on an existing FiftyOne dataset."
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        type=nonempty,
        help="Existing FiftyOne dataset name.",
    )
    parser.add_argument(
        "--hamming-max",
        type=nonnegative_int,
        default=DEFAULT_HAMMING_MAX,
        help="Max pHash Hamming distance for a near group. Default 2.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count and write CSV only; do not delete or tag samples.",
    )
    return parser.parse_args(argv)


def normalize_text(value: object) -> str:
    """Return a stripped string, or empty if missing."""
    if value is None:
        return ""
    return str(value).strip()


def field_values(dataset: fo.Dataset, field_name: str) -> list[object]:
    """Return per-sample values, or None placeholders when the field is absent."""
    if not dataset.has_field(field_name):
        return [None] * len(dataset)
    return dataset.values(field_name)


def collect_sample_refs(dataset: fo.Dataset) -> list[SampleRef]:
    """Load grouping fields in bulk instead of iterating Sample objects."""
    ids = dataset.values("id")
    filepaths = dataset.values("filepath")
    relpaths = field_values(dataset, "relpath")
    sha256s = field_values(dataset, SHA256_FIELD)
    phashes = field_values(dataset, PHASH_FIELD)
    tags_list = field_values(dataset, "tags")
    dup_groups = field_values(dataset, DUP_GROUP_FIELD)
    if dataset.has_field("metadata"):
        widths = dataset.values("metadata.width")
        heights = dataset.values("metadata.height")
    else:
        widths = [None] * len(ids)
        heights = [None] * len(ids)
    refs: list[SampleRef] = []
    for index, sample_id in enumerate(ids):
        tags = tags_list[index] or []
        width = widths[index] or 0
        height = heights[index] or 0
        refs.append(
            SampleRef(
                id=str(sample_id),
                filepath=str(filepaths[index]),
                relpath=normalize_text(relpaths[index]),
                sha256=normalize_text(sha256s[index]).lower(),
                phash=normalize_text(phashes[index]).lower(),
                area=int(width) * int(height),
                has_dup_near=DUP_NEAR_TAG in tags,
                has_dup_group=bool(normalize_text(dup_groups[index])),
            )
        )
    return refs


def keep_sort_key(ref: SampleRef) -> tuple[str, int, str]:
    """Sort key so the first item is the sample to keep."""
    path = ref.relpath or ref.filepath
    return (path, -ref.area, ref.id)


def group_refs_by_key(refs: list[SampleRef], key_name: str) -> dict[str, list[SampleRef]]:
    """Group refs by a non-empty string attribute."""
    grouped: dict[str, list[SampleRef]] = defaultdict(list)
    for ref in refs:
        value = getattr(ref, key_name)
        if value:
            grouped[value].append(ref)
    return grouped


def pick_keeper(group: list[SampleRef]) -> SampleRef:
    """Return the canonical sample for a duplicate group."""
    return sorted(group, key=keep_sort_key)[0]


def exact_groups_from_refs(refs: list[SampleRef]) -> dict[str, list[SampleRef]]:
    """Return sha256 groups that contain more than one sample."""
    return {key: group for key, group in group_refs_by_key(refs, "sha256").items() if len(group) >= 2}


def exact_delete_rows(groups: dict[str, list[SampleRef]]) -> list[ExactDeleteRow]:
    """Build delete records for sha256 groups larger than one."""
    rows: list[ExactDeleteRow] = []
    for sha256, group in groups.items():
        keeper = pick_keeper(group)
        for ref in group:
            if ref.id == keeper.id:
                continue
            rows.append(
                ExactDeleteRow(
                    sha256=sha256,
                    kept_filepath=keeper.filepath,
                    kept_relpath=keeper.relpath,
                    deleted_id=ref.id,
                    deleted_filepath=ref.filepath,
                    deleted_relpath=ref.relpath,
                )
            )
    rows.sort(key=lambda row: (row.sha256, row.deleted_relpath, row.deleted_filepath, row.deleted_id))
    return rows


def parse_phash_int(phash: str) -> int | None:
    """Parse a hex pHash; return None if invalid."""
    try:
        value = int(phash, 16)
    except ValueError:
        return None
    if value < 0:
        return None
    return value


def phash_bit_blocks(hamming_max: int) -> list[tuple[int, int]]:
    """Split 64-bit hashes into ``hamming_max + 1`` blocks (pigeonhole).

    Two hashes with Hamming distance <= ``hamming_max`` must share at least
    one identical block, so only pairs in the same block need comparing.
    """
    block_count = hamming_max + 1
    base, remainder = divmod(PHASH_BITS, block_count)
    blocks: list[tuple[int, int]] = []
    shift = 0
    for index in range(block_count):
        width = base + (1 if index < remainder else 0)
        if width <= 0:
            continue
        blocks.append((shift, (1 << width) - 1))
        shift += width
    return blocks


def cluster_phashes(phashes: list[str], hamming_max: int) -> dict[str, str]:
    """Map each pHash to a canonical group id.

    ``hamming_max == 0`` groups identical strings. Larger values merge hashes
    whose Hamming distance is at most the threshold (connected components),
    using block indexes instead of all-pairs comparison.
    """
    unique = sorted(set(phashes))
    if hamming_max == 0:
        return {item: item for item in unique}
    valid: list[str] = []
    values: list[int] = []
    for item in unique:
        parsed = parse_phash_int(item)
        if parsed is None:
            continue
        valid.append(item)
        values.append(parsed)
    n = len(valid)
    if n == 0:
        return {}
    if hamming_max >= PHASH_BITS:
        root = valid[0]
        return {item: root for item in valid}
    parent = list(range(n))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if valid[root_left] <= valid[root_right]:
            parent[root_right] = root_left
        else:
            parent[root_left] = root_right

    blocks = phash_bit_blocks(hamming_max)
    logger.info(
        "Clustering %s unique phashes hamming_max=%s blocks=%s",
        n,
        hamming_max,
        len(blocks),
    )
    for shift, mask in blocks:
        buckets: dict[int, list[int]] = defaultdict(list)
        for index, value in enumerate(values):
            buckets[(value >> shift) & mask].append(index)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for left_pos, left_index in enumerate(members):
                left_value = values[left_index]
                for right_index in members[left_pos + 1 :]:
                    if (left_value ^ values[right_index]).bit_count() <= hamming_max:
                        union(left_index, right_index)
    return {valid[index]: valid[find(index)] for index in range(n)}


def near_groups(refs: list[SampleRef], hamming_max: int) -> dict[str, list[SampleRef]]:
    """Group remaining samples by pHash Hamming distance."""
    phashes = [ref.phash for ref in refs if ref.phash and parse_phash_int(ref.phash) is not None]
    cluster_of = cluster_phashes(phashes, hamming_max)
    grouped: dict[str, list[SampleRef]] = defaultdict(list)
    for ref in refs:
        cluster = cluster_of.get(ref.phash)
        if cluster is None:
            continue
        grouped[cluster].append(ref)
    return {key: group for key, group in grouped.items() if len(group) >= 2}


def sample_already_marked(ref: SampleRef) -> bool:
    """Return True when near-dup fields should be left alone."""
    return ref.has_dup_near or ref.has_dup_group


def plan_near_updates(
    groups: dict[str, list[SampleRef]],
) -> tuple[dict[str, tuple[str, str, bool]], int]:
    """Return sample id -> (dup_group, dup_of, add_dup_near) and skip count.

    Keepers get ``dup_group`` and empty ``dup_of`` without ``dup_near``.
    Extras get ``dup_near`` and ``dup_of`` pointing at the keeper filepath.
    Already tagged samples are left unchanged.
    """
    updates: dict[str, tuple[str, str, bool]] = {}
    skipped = 0
    for group in groups.values():
        keeper = pick_keeper(group)
        dup_group = keeper.phash
        for ref in group:
            is_extra = ref.id != keeper.id
            if sample_already_marked(ref):
                skipped += 1
                continue
            updates[ref.id] = (dup_group, keeper.filepath if is_extra else "", is_extra)
    return updates, skipped


def add_dup_near_tag(sample: fo.Sample) -> None:
    """Add ``dup_near`` if it is not already present."""
    tags = list(sample.tags or [])
    if DUP_NEAR_TAG not in tags:
        sample.tags = tags + [DUP_NEAR_TAG]


def report_dir() -> Path:
    """Return ``<cwd>/tmp``, creating it if needed."""
    path = Path.cwd() / REPORT_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def csv_row(
    dataset_name: str,
    kind: str,
    action: str,
    dup_group: str,
    ref: SampleRef,
    keeper: SampleRef,
) -> dict[str, str]:
    """Build one report CSV row."""
    return {
        "dataset_name": dataset_name,
        "kind": kind,
        "action": action,
        "dup_group": dup_group,
        "sample_id": ref.id,
        "filepath": ref.filepath,
        "relpath": ref.relpath,
        "sha256": ref.sha256,
        "phash": ref.phash,
        "kept_sample_id": keeper.id,
        "kept_filepath": keeper.filepath,
        "kept_relpath": keeper.relpath,
    }


def build_report_rows(
    dataset_name: str,
    exact_groups: dict[str, list[SampleRef]],
    near: dict[str, list[SampleRef]],
) -> list[dict[str, str]]:
    """List every sample in exact and near groups, keepers included."""
    rows: list[dict[str, str]] = []
    for sha256, group in exact_groups.items():
        keeper = pick_keeper(group)
        for ref in group:
            action = "keep" if ref.id == keeper.id else "delete"
            rows.append(csv_row(dataset_name, "exact", action, sha256, ref, keeper))
    for group in near.values():
        keeper = pick_keeper(group)
        for ref in group:
            action = "keep" if ref.id == keeper.id else "tag_dup_near"
            rows.append(csv_row(dataset_name, "near", action, keeper.phash, ref, keeper))
    rows.sort(
        key=lambda row: (
            row["kind"],
            row["dup_group"],
            0 if row["action"] == "keep" else 1,
            row["relpath"],
            row["filepath"],
            row["sample_id"],
        )
    )
    return rows


def write_report_csv(
    path: Path,
    rows: list[dict[str, str]],
    columns: tuple[str, ...] | None = None,
) -> None:
    """Write a CSV, overwriting a previous file of the same name."""
    fieldnames = list(columns or REPORT_COLUMNS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def slim_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only columns meant for spreadsheet review."""
    return [{column: row[column] for column in SLIM_COLUMNS} for row in rows]


def group_rows_by_dup_group(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    """Split already-sorted rows into contiguous ``dup_group`` lists."""
    groups: list[list[dict[str, str]]] = []
    current_key = None
    current: list[dict[str, str]] = []
    for row in rows:
        key = row["dup_group"]
        if current and key != current_key:
            groups.append(current)
            current = [row]
        else:
            current.append(row)
        current_key = key
    if current:
        groups.append(current)
    return groups


def chunk_groups(
    groups: list[list[dict[str, str]]],
    groups_per_file: int,
) -> list[list[dict[str, str]]]:
    """Pack whole groups into shards of at most ``groups_per_file`` groups."""
    shards: list[list[dict[str, str]]] = []
    for start in range(0, len(groups), groups_per_file):
        shard: list[dict[str, str]] = []
        for group in groups[start : start + groups_per_file]:
            shard.extend(group)
        shards.append(shard)
    return shards


def clear_near_shards(directory: Path, dataset_name: str) -> None:
    """Remove previous near shard files for this dataset."""
    for path in directory.glob(f"dedup_{dataset_name}_near_*.csv"):
        path.unlink()


def write_dedup_csvs(dataset_name: str, rows: list[dict[str, str]]) -> dict[str, object]:
    """Write archive, exact, and slim near-shard CSVs under ``tmp/``."""
    directory = report_dir()
    archive_path = directory / f"dedup_{dataset_name}.csv"
    exact_path = directory / f"dedup_{dataset_name}_exact.csv"
    write_report_csv(archive_path, rows)
    exact_rows = [row for row in rows if row["kind"] == "exact"]
    near_rows = [row for row in rows if row["kind"] == "near"]
    if exact_rows:
        write_report_csv(exact_path, exact_rows)
    elif exact_path.exists():
        exact_path.unlink()
    clear_near_shards(directory, dataset_name)
    near_shards = chunk_groups(
        group_rows_by_dup_group(near_rows),
        NEAR_GROUPS_PER_FILE,
    )
    shard_count = len(near_shards)
    pad = max(2, len(str(shard_count)))
    shard_paths: list[str] = []
    for index, shard in enumerate(near_shards, start=1):
        shard_path = directory / f"dedup_{dataset_name}_near_{index:0{pad}d}.csv"
        write_report_csv(shard_path, slim_rows(shard), SLIM_COLUMNS)
        shard_paths.append(str(shard_path))
    logger.info(
        "Wrote dedup CSVs dir=%s archive_rows=%s exact_rows=%s near_shards=%s",
        directory,
        len(rows),
        len(exact_rows),
        shard_count,
    )
    return {
        "csv_dir": str(directory),
        "csv_archive": str(archive_path),
        "csv_exact": str(exact_path) if exact_rows else "none",
        "near_shard_files": shard_count,
        "csv_rows": len(rows),
    }


def apply_exact_deletes(dataset: fo.Dataset, rows: list[ExactDeleteRow]) -> int:
    """Delete extra exact-duplicate samples. Returns deleted count."""
    ids = [row.deleted_id for row in rows]
    if not ids:
        return 0
    dataset.delete_samples(ids)
    return len(ids)


def apply_near_updates(
    dataset: fo.Dataset,
    updates: dict[str, tuple[str, str, bool]],
) -> int:
    """Write near-dup fields and tags. Returns samples written."""
    if not updates:
        return 0
    written = 0
    for sample in dataset.select(list(updates)).iter_samples(autosave=True):
        dup_group, dup_of, add_tag = updates[str(sample.id)]
        sample[DUP_GROUP_FIELD] = dup_group
        sample[DUP_OF_FIELD] = dup_of
        if add_tag:
            add_dup_near_tag(sample)
        written += 1
    return written


def print_report(values: dict[str, object]) -> None:
    """Print key=value lines in insertion order."""
    for key, value in values.items():
        print(f"{key}={value}")


def run_dedup(
    dataset: fo.Dataset,
    hamming_max: int,
    dry_run: bool,
) -> None:
    """Count or apply exact deletes and near-dup tags."""
    refs = collect_sample_refs(dataset)
    missing_sha256 = sum(1 for ref in refs if not ref.sha256)
    missing_phash = sum(
        1 for ref in refs if not ref.phash or parse_phash_int(ref.phash) is None
    )

    exact_groups = exact_groups_from_refs(refs)
    delete_rows = exact_delete_rows(exact_groups)
    deleted_ids = {row.deleted_id for row in delete_rows}
    remaining = [ref for ref in refs if ref.id not in deleted_ids]
    groups = near_groups(remaining, hamming_max)
    updates, already_tagged = plan_near_updates(groups)

    near_to_tag = sum(1 for _group, dup_of, add_tag in updates.values() if add_tag)
    csv_rows = build_report_rows(dataset.name, exact_groups, groups)
    csv_info = write_dedup_csvs(dataset.name, csv_rows)

    print_report(
        {
            "mode": "dry-run" if dry_run else "dedup",
            "dataset_name": dataset.name,
            "hamming_max": hamming_max,
            "samples": len(refs),
            "missing_sha256": missing_sha256,
            "missing_phash": missing_phash,
            "exact_groups": len(exact_groups),
            "exact_to_delete": len(delete_rows),
            "near_groups": len(groups),
            "near_to_tag": near_to_tag,
            "already_tagged": already_tagged,
            **csv_info,
        }
    )
    if dry_run:
        return

    deleted_count = 0
    if delete_rows:
        deleted_count = apply_exact_deletes(dataset, delete_rows)

    tagged_count = apply_near_updates(dataset, updates)

    dataset.save()
    print_report(
        {
            "exact_deleted": deleted_count,
            "near_tagged": tagged_count,
            "csv_dir": csv_info["csv_dir"],
            "dedup_done": "true",
        }
    )


def main(argv: list[str] | None = None) -> int:
    """Run dry-run counting or apply dedup to an existing dataset."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)7s] %(name)s:%(lineno)d - %(message)s",
    )
    args = parse_args(argv)
    if not fo.dataset_exists(args.dataset_name):
        logger.error("Dataset does not exist: %s", args.dataset_name)
        return 1
    dataset = fo.load_dataset(args.dataset_name)
    run_dedup(dataset, args.hamming_max, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
