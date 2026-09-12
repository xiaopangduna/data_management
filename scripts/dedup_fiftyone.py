"""Tag exact and near duplicates on an existing FiftyOne dataset.

Exact matches share ``sha256``. Every member gets ``dup_repeat``; one sample
gets ``dup_repeat_keep`` and the rest ``dup_repeat_drop``. Near matches share
a 64-bit pHash within ``--hamming-max``: every member gets ``dup_near``.
Samples tagged ``dup_repeat_drop`` are left out of near clustering. Both kinds
write ``dup_group`` / ``dup_of`` (``exact:<sha256>`` or ``near:<phash>``).
Tags only; does not delete samples or files. Every run writes CSVs under
``tmp/`` (including ``--dry-run``). Does not create or delete datasets.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
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
DUP_REPEAT_TAG = "dup_repeat"
DUP_REPEAT_KEEP_TAG = "dup_repeat_keep"
DUP_REPEAT_DROP_TAG = "dup_repeat_drop"
DUP_NEAR_TAG = "dup_near"
DUP_GROUP_FIELD = "dup_group"
DUP_OF_FIELD = "dup_of"
EXACT_GROUP_PREFIX = "exact:"
NEAR_GROUP_PREFIX = "near:"
DUP_ALL_TAGS = (
    DUP_REPEAT_TAG,
    DUP_REPEAT_KEEP_TAG,
    DUP_REPEAT_DROP_TAG,
    DUP_NEAR_TAG,
)
DUP_ALL_TAG_SET = frozenset(DUP_ALL_TAGS)
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
    tags: tuple[str, ...]
    dup_group: str
    dup_of: str

    def has_tag(self, tag: str) -> bool:
        """Return True when ``tag`` is already on the sample."""
        return tag in self.tags


@dataclass(frozen=True)
class GroupDecision:
    """Keep/drop roles for one exact-duplicate group."""

    keeper: SampleRef | None
    roles: dict[str, str]
    keep_count: int
    is_new: bool


@dataclass(frozen=True)
class SampleUpdate:
    """Desired tags and grouping fields for one sample."""

    tags: tuple[str, ...]
    dup_group: str
    dup_of: str


@dataclass
class DedupPlan:
    """Exact/near groups, field updates, and report counts."""

    exact_groups: dict[str, list[SampleRef]]
    exact_decisions: dict[str, GroupDecision]
    near_groups: dict[str, list[SampleRef]]
    near_keepers: dict[str, SampleRef]
    updates: dict[str, SampleUpdate]
    drop_ids: list[str]
    exact_keep: int = 0
    exact_drop: int = 0
    exact_keep_conflict: int = 0
    near_members: int = 0
    stale_cleared: int = 0
    refs: list[SampleRef] = field(default_factory=list)


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
        description="Tag exact and near duplicates on an existing FiftyOne dataset."
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
        help="Count and write CSV only; do not tag samples.",
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


def exact_group_id(sha256: str) -> str:
    """Return the ``dup_group`` value for an exact sha256 group."""
    return f"{EXACT_GROUP_PREFIX}{sha256}"


def near_group_id(phash: str) -> str:
    """Return the ``dup_group`` value for a near pHash cluster."""
    return f"{NEAR_GROUP_PREFIX}{phash}"


def collect_sample_refs(dataset: fo.Dataset) -> list[SampleRef]:
    """Load grouping fields in bulk instead of iterating Sample objects."""
    ids = dataset.values("id")
    filepaths = dataset.values("filepath")
    relpaths = field_values(dataset, "relpath")
    sha256s = field_values(dataset, SHA256_FIELD)
    phashes = field_values(dataset, PHASH_FIELD)
    tags_list = field_values(dataset, "tags")
    dup_groups = field_values(dataset, DUP_GROUP_FIELD)
    dup_ofs = field_values(dataset, DUP_OF_FIELD)
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
                tags=tuple(str(tag) for tag in tags),
                dup_group=normalize_text(dup_groups[index]),
                dup_of=normalize_text(dup_ofs[index]),
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


def resolve_exact_decision(group: list[SampleRef]) -> GroupDecision:
    """Assign keep/drop without reshuffling a group that already has decisions.

    New groups (no keep or drop tags) use ``pick_keeper``. Existing keep/drop
    tags stay as they are. New members of a decided group become drop when a
    keeper already exists. Zero or multiple keeps are logged, not fixed.
    """
    keeps = [ref for ref in group if ref.has_tag(DUP_REPEAT_KEEP_TAG)]
    drops = [ref for ref in group if ref.has_tag(DUP_REPEAT_DROP_TAG)]
    both = [ref for ref in group if ref.has_tag(DUP_REPEAT_KEEP_TAG) and ref.has_tag(DUP_REPEAT_DROP_TAG)]
    if both:
        logger.warning(
            "Exact group sha256=%s has %s sample(s) tagged both keep and drop",
            group[0].sha256,
            len(both),
        )
    decided = bool(keeps or drops)
    roles: dict[str, str] = {}
    if not decided:
        keeper = pick_keeper(group)
        for ref in group:
            roles[ref.id] = "keep" if ref.id == keeper.id else "drop"
        return GroupDecision(keeper=keeper, roles=roles, keep_count=1, is_new=True)

    if len(keeps) == 1:
        keeper = keeps[0]
    elif len(keeps) > 1:
        logger.warning(
            "Exact group sha256=%s has %s keep tags; leaving them unchanged",
            group[0].sha256,
            len(keeps),
        )
        keeper = pick_keeper(keeps)
    else:
        logger.warning(
            "Exact group sha256=%s has drop tags but no keep; not assigning a keeper",
            group[0].sha256,
        )
        keeper = None

    for ref in group:
        if ref.has_tag(DUP_REPEAT_KEEP_TAG):
            roles[ref.id] = "keep"
        elif ref.has_tag(DUP_REPEAT_DROP_TAG):
            roles[ref.id] = "drop"
        elif keeper is not None:
            roles[ref.id] = "drop"
        else:
            roles[ref.id] = ""
    keep_count = sum(1 for role in roles.values() if role == "keep")
    return GroupDecision(keeper=keeper, roles=roles, keep_count=keep_count, is_new=False)


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
    """Group samples by pHash Hamming distance."""
    phashes = [ref.phash for ref in refs if ref.phash and parse_phash_int(ref.phash) is not None]
    cluster_of = cluster_phashes(phashes, hamming_max)
    grouped: dict[str, list[SampleRef]] = defaultdict(list)
    for ref in refs:
        cluster = cluster_of.get(ref.phash)
        if cluster is None:
            continue
        grouped[cluster].append(ref)
    return {key: group for key, group in grouped.items() if len(group) >= 2}


def compose_tags(existing: tuple[str, ...], desired_dup: set[str]) -> tuple[str, ...]:
    """Keep non-dup tags and replace dup tags with ``desired_dup``."""
    kept = [tag for tag in existing if tag not in DUP_ALL_TAG_SET]
    kept.extend(tag for tag in DUP_ALL_TAGS if tag in desired_dup)
    return tuple(kept)


def desired_dup_tags(role: str, in_exact: bool, in_near: bool, existing: tuple[str, ...]) -> set[str]:
    """Return the dup tags a sample should have after this run."""
    tags: set[str] = set()
    if in_exact:
        tags.add(DUP_REPEAT_TAG)
        if role == "keep":
            tags.add(DUP_REPEAT_KEEP_TAG)
        elif role == "drop":
            tags.add(DUP_REPEAT_DROP_TAG)
        if DUP_REPEAT_KEEP_TAG in existing and DUP_REPEAT_DROP_TAG in existing:
            tags.add(DUP_REPEAT_KEEP_TAG)
            tags.add(DUP_REPEAT_DROP_TAG)
    if in_near:
        tags.add(DUP_NEAR_TAG)
    return tags


def sample_changed(ref: SampleRef, update: SampleUpdate) -> bool:
    """Return True when tags or grouping fields differ from the plan."""
    return ref.tags != update.tags or ref.dup_group != update.dup_group or ref.dup_of != update.dup_of


def build_dedup_plan(refs: list[SampleRef], hamming_max: int) -> DedupPlan:
    """Compute exact/near groups, tags, and field updates without writing."""
    exact = exact_groups_from_refs(refs)
    decisions = {sha256: resolve_exact_decision(group) for sha256, group in exact.items()}
    drop_ids = [
        ref.id
        for sha256, group in exact.items()
        for ref in group
        if decisions[sha256].roles.get(ref.id) == "drop"
        and not (ref.has_tag(DUP_REPEAT_KEEP_TAG) and ref.has_tag(DUP_REPEAT_DROP_TAG))
    ]
    drop_id_set = set(drop_ids)
    near_refs = [ref for ref in refs if ref.id not in drop_id_set]
    near = near_groups(near_refs, hamming_max)
    near_keepers = {key: pick_keeper(group) for key, group in near.items()}

    exact_of = {ref.id: sha256 for sha256, group in exact.items() for ref in group}
    near_of = {ref.id: key for key, group in near.items() for ref in group}

    updates: dict[str, SampleUpdate] = {}
    stale_cleared = 0
    for ref in refs:
        sha256 = exact_of.get(ref.id)
        near_key = near_of.get(ref.id)
        in_exact = sha256 is not None
        in_near = near_key is not None
        role = decisions[sha256].roles.get(ref.id, "") if in_exact else ""
        desired = desired_dup_tags(role, in_exact, in_near, ref.tags)
        if in_exact:
            keeper = decisions[sha256].keeper
            group_id = exact_group_id(sha256)
            dup_of = keeper.filepath if role == "drop" and keeper is not None else ""
        elif in_near:
            keeper = near_keepers[near_key]
            group_id = near_group_id(near_key)
            dup_of = keeper.filepath if ref.id != keeper.id else ""
        else:
            group_id = ""
            dup_of = ""
        update = SampleUpdate(tags=compose_tags(ref.tags, desired), dup_group=group_id, dup_of=dup_of)
        if sample_changed(ref, update):
            updates[ref.id] = update
            removed = (set(ref.tags) & DUP_ALL_TAG_SET) - desired
            if removed and not in_exact and not in_near:
                stale_cleared += 1

    exact_keep = sum(1 for decision in decisions.values() for role in decision.roles.values() if role == "keep")
    exact_drop = sum(1 for decision in decisions.values() for role in decision.roles.values() if role == "drop")
    exact_keep_conflict = sum(1 for decision in decisions.values() if decision.keep_count != 1)
    return DedupPlan(
        exact_groups=exact,
        exact_decisions=decisions,
        near_groups=near,
        near_keepers=near_keepers,
        updates=updates,
        drop_ids=drop_ids,
        exact_keep=exact_keep,
        exact_drop=exact_drop,
        exact_keep_conflict=exact_keep_conflict,
        near_members=sum(len(group) for group in near.values()),
        stale_cleared=stale_cleared,
        refs=refs,
    )


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
    keeper: SampleRef | None,
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
        "kept_sample_id": keeper.id if keeper is not None else "",
        "kept_filepath": keeper.filepath if keeper is not None else "",
        "kept_relpath": keeper.relpath if keeper is not None else "",
    }


def exact_action(role: str) -> str:
    """CSV action for an exact-group member."""
    if role == "keep":
        return "keep"
    if role == "drop":
        return "drop"
    return "member"


def build_report_rows(dataset_name: str, plan: DedupPlan) -> list[dict[str, str]]:
    """List every sample in exact and near groups, keepers included."""
    rows: list[dict[str, str]] = []
    for sha256, group in plan.exact_groups.items():
        decision = plan.exact_decisions[sha256]
        for ref in group:
            rows.append(
                csv_row(
                    dataset_name,
                    "exact",
                    exact_action(decision.roles.get(ref.id, "")),
                    exact_group_id(sha256),
                    ref,
                    decision.keeper,
                )
            )
    for key, group in plan.near_groups.items():
        keeper = plan.near_keepers[key]
        for ref in group:
            action = "keep" if ref.id == keeper.id else "tag_dup_near"
            rows.append(csv_row(dataset_name, "near", action, near_group_id(key), ref, keeper))
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


def ensure_dup_fields(dataset: fo.Dataset) -> None:
    """Add grouping fields when the dataset does not already have them."""
    if not dataset.has_field(DUP_GROUP_FIELD):
        dataset.add_sample_field(DUP_GROUP_FIELD, fo.StringField)
    if not dataset.has_field(DUP_OF_FIELD):
        dataset.add_sample_field(DUP_OF_FIELD, fo.StringField)


def apply_updates(dataset: fo.Dataset, updates: dict[str, SampleUpdate]) -> int:
    """Write planned tags and grouping fields. Returns samples written."""
    if not updates:
        return 0
    ensure_dup_fields(dataset)
    written = 0
    for sample in dataset.select(list(updates)).iter_samples(autosave=True):
        update = updates[str(sample.id)]
        sample.tags = list(update.tags)
        sample[DUP_GROUP_FIELD] = update.dup_group
        sample[DUP_OF_FIELD] = update.dup_of
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
    """Count or apply exact/near tags. Does not delete samples."""
    refs = collect_sample_refs(dataset)
    missing_sha256 = sum(1 for ref in refs if not ref.sha256)
    missing_phash = sum(
        1 for ref in refs if not ref.phash or parse_phash_int(ref.phash) is None
    )
    plan = build_dedup_plan(refs, hamming_max)
    csv_rows = build_report_rows(dataset.name, plan)
    csv_info = write_dedup_csvs(dataset.name, csv_rows)

    print_report(
        {
            "mode": "dry-run" if dry_run else "tag",
            "dataset_name": dataset.name,
            "hamming_max": hamming_max,
            "samples": len(refs),
            "missing_sha256": missing_sha256,
            "missing_phash": missing_phash,
            "exact_groups": len(plan.exact_groups),
            "exact_keep": plan.exact_keep,
            "exact_drop": plan.exact_drop,
            "exact_keep_conflict": plan.exact_keep_conflict,
            "near_groups": len(plan.near_groups),
            "near_members": plan.near_members,
            "to_update": len(plan.updates),
            "stale_cleared": plan.stale_cleared,
            **csv_info,
        }
    )
    if dry_run:
        return

    tagged_count = apply_updates(dataset, plan.updates)
    dataset.save()
    print_report(
        {
            "tagged": tagged_count,
            "csv_dir": csv_info["csv_dir"],
            "dedup_done": "true",
        }
    )


def main(argv: list[str] | None = None) -> int:
    """Run dry-run counting or apply dedup tags to an existing dataset."""
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
