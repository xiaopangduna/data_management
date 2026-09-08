"""Attach X-AnyLabeling sidecar JSON onto an existing FiftyOne dataset.

Reads ``*.json`` next to image files in a task directory (the export folder).
Matches ``sample_id`` from the JSON (or ``description`` / ``fo_sample_id=``).
Overwrites ``ground_truth_detect`` for matched samples. Does not create or
delete datasets, and does not change filepaths or hashes.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import warnings
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
SAMPLE_ID_PREFIX = "fo_sample_id="
XLABEL_TAG = "xlabel"
XLABEL_CHECKED_TAG = "xlabel_checked"
SKIP_JSON_NAMES = frozenset({"manifest.json"})
SUPPORTED_SHAPES = frozenset({"rectangle", "polygon"})
ISSUE_COLUMNS = (
    "issue",
    "sample_id",
    "relpath",
    "label_relpath",
    "detail",
)


@dataclass(frozen=True)
class ParsedBoxes:
    """XLABEL shapes converted to FiftyOne top-left xywh."""

    sample_id: str
    boxes: tuple[tuple[str, tuple[float, float, float, float]], ...]
    label_relpath: str
    checked: bool


@dataclass
class AttachPlan:
    """Counts and actions for one attach run."""

    label_files: int = 0
    matched: int = 0
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
    """Parse a comma-separated class list."""
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("provide at least one class name")
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for attaching X-AnyLabeling JSON."""
    parser = argparse.ArgumentParser(
        description="Attach X-AnyLabeling JSON from a task folder onto FiftyOne."
    )
    parser.add_argument("--dataset-name", required=True, type=nonempty)
    parser.add_argument(
        "--task-dir",
        required=True,
        type=Path,
        help="Export folder with image symlinks and sidecar JSON.",
    )
    parser.add_argument(
        "--class-names",
        required=True,
        type=class_names,
        help="Allowed labels; other shape labels are rejected.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip samples whose boxes already differ (YOLO-attach behavior).",
    )
    parser.add_argument(
        "--clear-empty",
        action="store_true",
        help="Clear detections when JSON has no valid shapes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and write the issue CSV only; do not change the dataset.",
    )
    return parser.parse_args(argv)


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


def iter_json_paths(task_dir: Path) -> list[Path]:
    """List sidecar JSON files; do not follow directory symlinks."""
    paths: list[Path] = []
    for directory, dir_names, filenames in os.walk(task_dir, followlinks=False):
        dir_names.sort()
        for filename in filenames:
            path = Path(directory) / filename
            if path.suffix.lower() != ".json":
                continue
            if filename.lower() in SKIP_JSON_NAMES:
                continue
            paths.append(path)
    paths.sort()
    return paths


def parse_fo_sample_id(data: dict) -> str:
    """Read sample_id from top-level field or description."""
    sample_id = str(data.get("sample_id") or "").strip()
    if sample_id:
        return sample_id
    description = str(data.get("description") or "")
    for part in description.replace(",", " ").split():
        if part.startswith(SAMPLE_ID_PREFIX):
            return part[len(SAMPLE_ID_PREFIX) :].strip()
    return ""


def image_size_from_json(data: dict) -> tuple[int, int] | None:
    """Return width, height stored in the JSON."""
    try:
        width = int(data.get("imageWidth") or 0)
        height = int(data.get("imageHeight") or 0)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def points_to_pixel_box(points: object) -> tuple[float, float, float, float] | None:
    """Return min/max pixel xyxy from a list of [x, y] points."""
    if not isinstance(points, list) or len(points) < 2:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
        except (TypeError, ValueError):
            return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def pixel_box_to_fo(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    """Convert pixel xyxy to FiftyOne relative top-left xywh."""
    x1, y1, x2, y2 = box
    if width <= 0 or height <= 0:
        return None
    left = round(x1 / width, BOX_DECIMALS)
    top = round(y1 / height, BOX_DECIMALS)
    box_w = round((x2 - x1) / width, BOX_DECIMALS)
    box_h = round((y2 - y1) / height, BOX_DECIMALS)
    if box_w <= 0 or box_h <= 0:
        return None
    return left, top, box_w, box_h


def parse_xlabel_file(
    json_path: Path,
    task_dir: Path,
    names: set[str],
) -> tuple[ParsedBoxes | None, list[dict[str, str]]]:
    """Parse one JSON. Returns boxes or None when the file must not be applied."""
    label_relpath = json_path.relative_to(task_dir).as_posix()
    issues: list[dict[str, str]] = []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        issues.append(issue_row("parse_error", label_relpath=label_relpath, detail=str(error)))
        return None, issues
    if not isinstance(data, dict):
        issues.append(issue_row("parse_error", label_relpath=label_relpath, detail="not an object"))
        return None, issues
    sample_id = parse_fo_sample_id(data)
    size = image_size_from_json(data)
    if size is None:
        issues.append(issue_row("missing_size", sample_id=sample_id, label_relpath=label_relpath))
        return None, issues
    width, height = size
    shapes = data.get("shapes")
    if shapes is None:
        shapes = []
    if not isinstance(shapes, list):
        issues.append(issue_row("parse_error", sample_id=sample_id, label_relpath=label_relpath, detail="shapes"))
        return None, issues
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            issues.append(
                issue_row(
                    "parse_error",
                    sample_id=sample_id,
                    label_relpath=label_relpath,
                    detail=f"shape {index}: not an object",
                )
            )
            continue
        shape_type = str(shape.get("shape_type") or "").strip().lower()
        if shape_type not in SUPPORTED_SHAPES:
            issues.append(
                issue_row(
                    "unsupported_shape",
                    sample_id=sample_id,
                    label_relpath=label_relpath,
                    detail=f"shape {index}: {shape_type or 'missing'}",
                )
            )
            continue
        label = str(shape.get("label") or "").strip()
        if label not in names:
            issues.append(
                issue_row(
                    "unknown_label",
                    sample_id=sample_id,
                    label_relpath=label_relpath,
                    detail=f"shape {index}: {label}",
                )
            )
            continue
        pixel_box = points_to_pixel_box(shape.get("points"))
        if pixel_box is None:
            issues.append(
                issue_row(
                    "parse_error",
                    sample_id=sample_id,
                    label_relpath=label_relpath,
                    detail=f"shape {index}: invalid points",
                )
            )
            continue
        fo_box = pixel_box_to_fo(pixel_box, width, height)
        if fo_box is None:
            issues.append(
                issue_row(
                    "parse_error",
                    sample_id=sample_id,
                    label_relpath=label_relpath,
                    detail=f"shape {index}: non-positive box",
                )
            )
            continue
        boxes.append((label, fo_box))
    if issues:
        return None, issues
    boxes.sort(key=lambda item: (item[0], item[1]))
    checked = bool(data.get("checked"))
    return (
        ParsedBoxes(
            sample_id=sample_id,
            boxes=tuple(boxes),
            label_relpath=label_relpath,
            checked=checked,
        ),
        issues,
    )


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
    dataset: fo.Dataset,
    task_dir: Path,
    names: list[str],
    overwrite: bool,
    clear_empty: bool,
) -> AttachPlan:
    """Match JSON files to samples and decide writes vs issue rows."""
    plan = AttachPlan()
    allowed = set(names)
    json_paths = iter_json_paths(task_dir)
    plan.label_files = len(json_paths)
    dataset_ids = {str(sample_id) for sample_id in dataset.values("id")}
    pending: list[ParsedBoxes] = []
    matched_ids: list[str] = []
    for json_path in json_paths:
        parsed, parse_issues = parse_xlabel_file(json_path, task_dir, allowed)
        if parse_issues:
            plan.issues.extend(parse_issues)
            continue
        if parsed is None:
            continue
        if not parsed.sample_id:
            plan.issues.append(
                issue_row("missing_sample_id", label_relpath=parsed.label_relpath)
            )
            continue
        if parsed.sample_id not in dataset_ids:
            plan.issues.append(
                issue_row(
                    "orphan_label",
                    sample_id=parsed.sample_id,
                    label_relpath=parsed.label_relpath,
                )
            )
            continue
        if not parsed.boxes and not clear_empty:
            plan.issues.append(
                issue_row(
                    "empty_label",
                    sample_id=parsed.sample_id,
                    label_relpath=parsed.label_relpath,
                    detail="no boxes; pass --clear-empty to wipe detections",
                )
            )
            continue
        plan.matched += 1
        matched_ids.append(parsed.sample_id)
        pending.append(parsed)
    existing = load_existing_boxes(dataset, matched_ids)
    for parsed in pending:
        current = existing.get(parsed.sample_id)
        new_boxes = parsed.boxes
        if current == new_boxes and (new_boxes or current is None):
            if new_boxes:
                plan.unchanged += 1
                continue
            if current is None and not new_boxes:
                plan.unchanged += 1
                continue
        if current is not None and current != new_boxes and not overwrite:
            plan.issues.append(
                issue_row(
                    "box_mismatch",
                    sample_id=parsed.sample_id,
                    label_relpath=parsed.label_relpath,
                    detail=f"existing={len(current)} new={len(new_boxes)}",
                )
            )
            continue
        plan.to_write[parsed.sample_id] = parsed
    return plan


def issue_csv_path(dataset_name: str) -> Path:
    """Return ``tmp/attach_xlabel_<dataset>.csv``."""
    directory = Path.cwd() / REPORT_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"attach_xlabel_{dataset_name}.csv"


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
    """Write detections and xlabel tags. Returns the number updated."""
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
        tags = list(sample.tags or [])
        if XLABEL_TAG not in tags:
            tags.append(XLABEL_TAG)
        if parsed.checked and XLABEL_CHECKED_TAG not in tags:
            tags.append(XLABEL_CHECKED_TAG)
        sample.tags = tags
        written += 1
    return written


def print_report(values: dict[str, object]) -> None:
    """Print key=value lines in insertion order."""
    for key, value in values.items():
        print(f"{key}={value}")


def run_attach(
    dataset: fo.Dataset,
    task_dir: Path,
    names: list[str],
    overwrite: bool,
    clear_empty: bool,
    dry_run: bool,
) -> int:
    """Plan, write the issue CSV, and optionally apply detections."""
    plan = build_attach_plan(dataset, task_dir, names, overwrite, clear_empty)
    csv_path = issue_csv_path(dataset.name)
    write_issue_csv(csv_path, plan.issues)
    logger.info("Wrote issue CSV %s rows=%s", csv_path, len(plan.issues))
    print_report(
        {
            "mode": "dry-run" if dry_run else "attach",
            "dataset_name": dataset.name,
            "task_dir": str(task_dir),
            "class_names": ",".join(names),
            "overwrite": overwrite,
            "clear_empty": clear_empty,
            "label_files": plan.label_files,
            "matched": plan.matched,
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
    """Attach X-AnyLabeling JSON to an existing dataset."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)7s] %(name)s:%(lineno)d - %(message)s",
    )
    args = parse_args(argv)
    task_dir = args.task_dir.expanduser().resolve()
    if not task_dir.is_dir():
        logger.error("Not a directory: %s", task_dir)
        return 1
    if not fo.dataset_exists(args.dataset_name):
        logger.error("Dataset does not exist: %s", args.dataset_name)
        return 1
    dataset = fo.load_dataset(args.dataset_name)
    return run_attach(
        dataset,
        task_dir,
        args.class_names,
        overwrite=not args.no_overwrite,
        clear_empty=args.clear_empty,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
