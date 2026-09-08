"""Attach YOLO detection txt files onto an existing FiftyOne dataset.

Matches ``labels/<stem>.txt`` to sample ``relpath`` (suffix stripped). Writes
``ground_truth_detect`` and ``label_relpath``. Does not overwrite existing
boxes; conflicts go to ``tmp/attach_labels_<dataset>.csv``. Does not create or
delete datasets, and does not change filepaths, hashes, or tags.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
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

DETECT_FIELD = "ground_truth_detect"
LABEL_RELPATH_FIELD = "label_relpath"
BOX_DECIMALS = 6
REPORT_SUBDIR = "tmp"
ISSUE_COLUMNS = (
    "issue",
    "sample_id",
    "relpath",
    "label_relpath",
    "detail",
)


@dataclass(frozen=True)
class SampleRef:
    """Sample identity used for label matching."""

    id: str
    relpath: str
    key: str


@dataclass(frozen=True)
class ParsedBoxes:
    """YOLO boxes converted to FiftyOne top-left xywh."""

    boxes: tuple[tuple[str, tuple[float, float, float, float]], ...]
    label_relpath: str


@dataclass
class AttachPlan:
    """Counts and actions for one attach run."""

    label_files: int = 0
    matched: int = 0
    unlabeled: int = 0
    to_write: dict[str, ParsedBoxes] = field(default_factory=dict)
    unchanged: int = 0
    issues: list[dict[str, str]] = field(default_factory=list)


def nonempty(value: str) -> str:
    """Reject blank CLI strings."""
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("must not be empty")
    return stripped


def class_names(value: str) -> list[str]:
    """Parse a comma-separated class list; index is YOLO class_id."""
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("provide at least one class name")
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for attaching YOLO labels."""
    parser = argparse.ArgumentParser(
        description="Attach YOLO detection labels to an existing FiftyOne dataset."
    )
    parser.add_argument("--dataset-name", required=True, type=nonempty)
    parser.add_argument(
        "--labels-dir",
        required=True,
        type=Path,
        help="Directory of YOLO txt files (class_id cx cy w h).",
    )
    parser.add_argument(
        "--class-names",
        required=True,
        type=class_names,
        help="Comma-separated names; order is YOLO class_id (0, 1, ...).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and write the issue CSV only; do not change the dataset.",
    )
    return parser.parse_args(argv)


def path_key(relative: str) -> str:
    """Return a POSIX path without the suffix, used as the match key."""
    return Path(relative).with_suffix("").as_posix()


def normalize_text(value: object) -> str:
    """Return a stripped string, or empty if missing."""
    if value is None:
        return ""
    return str(value).strip()


def collect_sample_refs(dataset: fo.Dataset) -> list[SampleRef]:
    """Load id and relpath for every sample."""
    ids = dataset.values("id")
    if dataset.has_field("relpath"):
        relpaths = dataset.values("relpath")
    else:
        relpaths = [None] * len(ids)
    filepaths = dataset.values("filepath")
    refs: list[SampleRef] = []
    for index, sample_id in enumerate(ids):
        relpath = normalize_text(relpaths[index])
        if not relpath:
            relpath = Path(str(filepaths[index])).name
        refs.append(
            SampleRef(
                id=str(sample_id),
                relpath=relpath,
                key=path_key(relpath),
            )
        )
    return refs


def samples_by_key(refs: list[SampleRef]) -> dict[str, list[SampleRef]]:
    """Group samples by filename key."""
    grouped: dict[str, list[SampleRef]] = defaultdict(list)
    for ref in refs:
        grouped[ref.key].append(ref)
    return grouped


def iter_label_paths(labels_dir: Path) -> list[Path]:
    """List YOLO txt files; skip cache files; do not follow dir symlinks."""
    paths: list[Path] = []
    for directory, dir_names, filenames in os.walk(labels_dir, followlinks=False):
        dir_names.sort()
        for filename in filenames:
            if filename.endswith(".cache"):
                continue
            path = Path(directory) / filename
            if path.suffix.lower() != ".txt":
                continue
            paths.append(path)
    paths.sort()
    return paths


def issue_row(
    issue: str,
    sample_id: str = "",
    relpath: str = "",
    label_relpath: str = "",
    detail: str = "",
) -> dict[str, str]:
    """Build one issue CSV row."""
    return {
        "issue": issue,
        "sample_id": sample_id,
        "relpath": relpath,
        "label_relpath": label_relpath,
        "detail": detail,
    }


def parse_yolo_label_file(
    label_path: Path,
    labels_dir: Path,
    names: list[str],
) -> tuple[ParsedBoxes | None, list[dict[str, str]]]:
    """Parse one txt. Returns boxes or None when the file must not be applied."""
    label_relpath = label_path.relative_to(labels_dir).as_posix()
    issues: list[dict[str, str]] = []
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    text = label_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        issues.append(issue_row("empty_label", label_relpath=label_relpath, detail="empty file"))
        return None, issues
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(
                issue_row(
                    "parse_error",
                    label_relpath=label_relpath,
                    detail=f"line {line_number}: expected 5 numbers, got {len(parts)}",
                )
            )
            continue
        try:
            class_id = int(float(parts[0]))
            center_x, center_y, width, height = (float(item) for item in parts[1:])
        except ValueError:
            issues.append(
                issue_row(
                    "parse_error",
                    label_relpath=label_relpath,
                    detail=f"line {line_number}: not numeric",
                )
            )
            continue
        if not 0 <= class_id < len(names):
            issues.append(
                issue_row(
                    "class_id_out_of_range",
                    label_relpath=label_relpath,
                    detail=f"line {line_number}: class_id {class_id} not in --class-names",
                )
            )
            continue
        if width <= 0 or height <= 0:
            issues.append(
                issue_row(
                    "parse_error",
                    label_relpath=label_relpath,
                    detail=f"line {line_number}: non-positive box size",
                )
            )
            continue
        boxes.append(
            (
                names[class_id],
                (
                    round(center_x - width / 2.0, BOX_DECIMALS),
                    round(center_y - height / 2.0, BOX_DECIMALS),
                    round(width, BOX_DECIMALS),
                    round(height, BOX_DECIMALS),
                ),
            )
        )
    if issues:
        return None, issues
    if not boxes:
        issues.append(issue_row("empty_label", label_relpath=label_relpath, detail="no boxes"))
        return None, issues
    boxes.sort(key=lambda item: (item[0], item[1]))
    return ParsedBoxes(boxes=tuple(boxes), label_relpath=label_relpath), issues


def existing_box_tuples(value: object) -> tuple[tuple[str, tuple[float, float, float, float]], ...] | None:
    """Normalize stored detections for comparison; None means unlabeled."""
    if value is None:
        return None
    detections = getattr(value, "detections", None)
    if not detections:
        return None
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for detection in detections:
        raw = list(detection.bounding_box)
        boxes.append(
            (
                str(detection.label),
                (
                    round(float(raw[0]), BOX_DECIMALS),
                    round(float(raw[1]), BOX_DECIMALS),
                    round(float(raw[2]), BOX_DECIMALS),
                    round(float(raw[3]), BOX_DECIMALS),
                ),
            )
        )
    boxes.sort(key=lambda item: (item[0], item[1]))
    return tuple(boxes)


def load_existing_boxes(
    dataset: fo.Dataset, sample_ids: list[str]
) -> dict[str, tuple[tuple[str, tuple[float, float, float, float]], ...] | None]:
    """Load current detection boxes for the given samples."""
    if not sample_ids:
        return {}
    if not dataset.has_field(DETECT_FIELD):
        return {sample_id: None for sample_id in sample_ids}
    view = dataset.select(sample_ids)
    ids = [str(sample_id) for sample_id in view.values("id")]
    values = view.values(DETECT_FIELD)
    return {sample_id: existing_box_tuples(value) for sample_id, value in zip(ids, values)}


def build_attach_plan(
    refs: list[SampleRef],
    labels_dir: Path,
    names: list[str],
    dataset: fo.Dataset,
) -> AttachPlan:
    """Match txt files to samples and decide writes vs issue rows."""
    plan = AttachPlan()
    grouped = samples_by_key(refs)
    label_paths = iter_label_paths(labels_dir)
    plan.label_files = len(label_paths)
    matched_ids: list[str] = []
    pending: list[tuple[SampleRef, ParsedBoxes]] = []
    label_keys: set[str] = set()

    for label_path in label_paths:
        key = path_key(label_path.relative_to(labels_dir).as_posix())
        label_keys.add(key)
        label_relpath = label_path.relative_to(labels_dir).as_posix()
        samples = grouped.get(key, [])
        if not samples:
            plan.issues.append(issue_row("orphan_label", label_relpath=label_relpath))
            continue
        if len(samples) > 1:
            for ref in samples:
                plan.issues.append(
                    issue_row(
                        "stem_collision",
                        sample_id=ref.id,
                        relpath=ref.relpath,
                        label_relpath=label_relpath,
                        detail=f"{len(samples)} samples share key {key}",
                    )
                )
            continue
        parsed, parse_issues = parse_yolo_label_file(label_path, labels_dir, names)
        if parse_issues:
            ref = samples[0]
            for row in parse_issues:
                row["sample_id"] = ref.id
                row["relpath"] = ref.relpath
            plan.issues.extend(parse_issues)
            continue
        if parsed is None:
            continue
        ref = samples[0]
        plan.matched += 1
        matched_ids.append(ref.id)
        pending.append((ref, parsed))

    plan.unlabeled = sum(1 for ref in refs if ref.key not in label_keys)
    existing = load_existing_boxes(dataset, matched_ids)
    for ref, parsed in pending:
        current = existing.get(ref.id)
        if current is None:
            plan.to_write[ref.id] = parsed
            continue
        if current == parsed.boxes:
            plan.unchanged += 1
            continue
        plan.issues.append(
            issue_row(
                "box_mismatch",
                sample_id=ref.id,
                relpath=ref.relpath,
                label_relpath=parsed.label_relpath,
                detail=f"existing={len(current)} new={len(parsed.boxes)}",
            )
        )
    return plan


def issue_csv_path(dataset_name: str) -> Path:
    """Return ``tmp/attach_labels_<dataset>.csv``."""
    directory = Path.cwd() / REPORT_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"attach_labels_{dataset_name}.csv"


def write_issue_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the issue CSV, including a header when there are no issues."""
    rows = sorted(
        rows,
        key=lambda row: (row["issue"], row["label_relpath"], row["relpath"], row["sample_id"]),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ISSUE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def apply_writes(dataset: fo.Dataset, to_write: dict[str, ParsedBoxes]) -> int:
    """Write detections onto samples. Returns the number updated."""
    if not to_write:
        return 0
    written = 0
    for sample in dataset.select(list(to_write)).iter_samples(autosave=True):
        parsed = to_write[str(sample.id)]
        sample[DETECT_FIELD] = fo.Detections(
            detections=[
                fo.Detection(label=label, bounding_box=list(bbox))
                for label, bbox in parsed.boxes
            ]
        )
        sample[LABEL_RELPATH_FIELD] = parsed.label_relpath
        written += 1
    return written


def print_report(values: dict[str, object]) -> None:
    """Print key=value lines in insertion order."""
    for key, value in values.items():
        print(f"{key}={value}")


def run_attach(
    dataset: fo.Dataset,
    labels_dir: Path,
    names: list[str],
    dry_run: bool,
) -> int:
    """Plan, write the issue CSV, and optionally apply detections."""
    refs = collect_sample_refs(dataset)
    plan = build_attach_plan(refs, labels_dir, names, dataset)
    csv_path = issue_csv_path(dataset.name)
    write_issue_csv(csv_path, plan.issues)
    logger.info("Wrote issue CSV %s rows=%s", csv_path, len(plan.issues))
    print_report(
        {
            "mode": "dry-run" if dry_run else "attach",
            "dataset_name": dataset.name,
            "labels_dir": str(labels_dir),
            "class_names": ",".join(names),
            "samples": len(refs),
            "label_files": plan.label_files,
            "matched": plan.matched,
            "unlabeled": plan.unlabeled,
            "to_write": len(plan.to_write),
            "unchanged": plan.unchanged,
            "issues": len(plan.issues),
            "csv_path": str(csv_path),
        }
    )
    if dry_run:
        return 0
    written = apply_writes(dataset, plan.to_write)
    dataset.save()
    print_report({"written": written, "attach_done": "true"})
    return 0


def main(argv: list[str] | None = None) -> int:
    """Attach YOLO labels to an existing dataset."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)7s] %(name)s:%(lineno)d - %(message)s",
    )
    args = parse_args(argv)
    labels_dir = args.labels_dir.expanduser().resolve()
    if not labels_dir.is_dir():
        logger.error("Not a directory: %s", labels_dir)
        return 1
    if not fo.dataset_exists(args.dataset_name):
        logger.error("Dataset does not exist: %s", args.dataset_name)
        return 1
    dataset = fo.load_dataset(args.dataset_name)
    return run_attach(dataset, labels_dir, args.class_names, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
