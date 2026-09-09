"""Attach X-AnyLabeling sidecar JSON onto an existing FiftyOne dataset.

Reads ``*.json`` under ``--label-dir``. Matches ``sample_id`` from the JSON
(or ``description`` / ``fo_sample_id=``), falling back to absolute image paths
with ``--images-dir`` when the ID is missing. Overwrites detections whose labels are in ``--class-names`` (drop old boxes
of those classes, then write JSON boxes). Other classes on the sample are
kept. Empty JSON for those classes clears them. Adds ``--tags``. Does not
create or delete datasets, and does not change filepaths or hashes.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"glob2(\.|$)",
)

from bson import ObjectId

import fiftyone as fo

logger = logging.getLogger(__name__)

DEFAULT_LABEL_FIELD = "ground_truth"
LABEL_RELPATH_FIELD = "label_relpath"
BOX_DECIMALS = 6
REPORT_SUBDIR = "tmp"
SAMPLE_ID_PREFIX = "fo_sample_id="
XLABEL_CHECKED_TAG = "xlabel_checked"
CHANGED_TAG = "changed"
QUERY_BATCH_SIZE = 1000
PIXEL_TOL = 2
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
    width: int
    height: int
    image_path: str = ""


@dataclass
class AttachPlan:
    """Counts and actions for one attach run."""

    label_files: int = 0
    matched: int = 0
    to_write: dict[str, ParsedBoxes] = field(default_factory=dict)
    to_tag: dict[str, ParsedBoxes] = field(default_factory=dict)
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


def sample_tags(value: str) -> list[str]:
    """Parse required batch tags added to updated samples."""
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("provide at least one tag")
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for attaching X-AnyLabeling JSON."""
    parser = argparse.ArgumentParser(
        description="Attach X-AnyLabeling JSON from a label folder onto FiftyOne."
    )
    parser.add_argument("--dataset-name", required=True, type=nonempty)
    parser.add_argument(
        "--label-dir",
        required=True,
        type=Path,
        help="Directory of sidecar JSON (e.g. tmp/images/val2017). Walks subfolders.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Image root for path matching when sample_id is missing; mirrors label-dir subfolders.",
    )
    parser.add_argument(
        "--class-names",
        required=True,
        type=class_names,
        help="Classes to replace from JSON; other detection labels on the sample are kept.",
    )
    parser.add_argument(
        "--label-field",
        default=DEFAULT_LABEL_FIELD,
        type=nonempty,
        help=f"FiftyOne Detections field to update (default: {DEFAULT_LABEL_FIELD}).",
    )
    parser.add_argument(
        "--tags",
        required=True,
        type=sample_tags,
        help="Batch tag, e.g. label_person_260909. Changed samples also get tag changed.",
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


def iter_json_paths(label_dir: Path) -> list[Path]:
    """List sidecar JSON files; do not follow directory symlinks."""
    paths: list[Path] = []
    for directory, dir_names, filenames in os.walk(label_dir, followlinks=False):
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
    label_dir: Path,
    names: set[str],
) -> tuple[ParsedBoxes | None, list[dict[str, str]]]:
    """Parse one JSON. Returns boxes or None when the file must not be applied."""
    label_relpath = json_path.relative_to(label_dir).as_posix()
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
            width=width,
            height=height,
            image_path=str(data.get("imagePath") or "").strip(),
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
    dataset: fo.Dataset,
    sample_ids: list[str],
    label_field: str = DEFAULT_LABEL_FIELD,
) -> dict[str, tuple[tuple[str, tuple[float, float, float, float]], ...] | None]:
    """Load current detection boxes for the given samples."""
    if not sample_ids:
        return {}
    if not dataset.has_field(label_field):
        return {sample_id: None for sample_id in sample_ids}
    existing = {}
    for offset in range(0, len(sample_ids), QUERY_BATCH_SIZE):
        view = dataset.select(sample_ids[offset:offset + QUERY_BATCH_SIZE])
        ids, values = view.values(["id", label_field])
        existing.update({str(sid): existing_box_tuples(value) for sid, value in zip(ids, values)})
        logger.info("Read existing boxes %d/%d", min(offset + QUERY_BATCH_SIZE, len(sample_ids)), len(sample_ids))
    return existing


def boxes_for_classes(
    boxes: tuple[tuple[str, tuple[float, float, float, float]], ...] | None,
    names: set[str],
) -> tuple[tuple[str, tuple[float, float, float, float]], ...]:
    """Return stored boxes whose label is in ``names``."""
    if not boxes:
        return ()
    return tuple(item for item in boxes if item[0] in names)


def pixel_xyxy(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Map relative xywh to integer pixel xyxy."""
    left, top, box_w, box_h = bbox
    x1 = round(left * width)
    y1 = round(top * height)
    x2 = round((left + box_w) * width)
    y2 = round((top + box_h) * height)
    return (x1, y1, x2, y2)


def max_corner_delta(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> int:
    """Largest absolute difference among the four corners."""
    return max(abs(left[index] - right[index]) for index in range(4))


def groups_match(
    old_boxes: list[tuple[int, int, int, int]],
    new_boxes: list[tuple[int, int, int, int]],
    tol: int,
) -> bool:
    """Greedy match: every old box has a unique new box within ``tol`` pixels."""
    if len(old_boxes) != len(new_boxes):
        return False
    used = [False] * len(new_boxes)
    for old in old_boxes:
        best_index = -1
        best_delta = None
        for index, new in enumerate(new_boxes):
            if used[index]:
                continue
            delta = max_corner_delta(old, new)
            if delta <= tol and (best_delta is None or delta < best_delta):
                best_index = index
                best_delta = delta
        if best_index < 0:
            return False
        used[best_index] = True
    return True


def boxes_equal(
    current: tuple[tuple[str, tuple[float, float, float, float]], ...] | None,
    new_boxes: tuple[tuple[str, tuple[float, float, float, float]], ...],
    names: set[str],
    width: int,
    height: int,
) -> bool:
    """True when class-name boxes match within PIXEL_TOL of each corner."""
    old_by_label: dict[str, list[tuple[int, int, int, int]]] = {}
    new_by_label: dict[str, list[tuple[int, int, int, int]]] = {}
    for label, bbox in boxes_for_classes(current, names):
        old_by_label.setdefault(label, []).append(pixel_xyxy(bbox, width, height))
    for label, bbox in new_boxes:
        new_by_label.setdefault(label, []).append(pixel_xyxy(bbox, width, height))
    if set(old_by_label) != set(new_by_label):
        return False
    labels = set(old_by_label) | set(new_by_label)
    return all(
        groups_match(old_by_label.get(label, []), new_by_label.get(label, []), PIXEL_TOL)
        for label in labels
    )


def build_attach_plan(
    dataset: fo.Dataset,
    label_dir: Path,
    names: list[str],
    images_dir: Path | None = None,
    label_field: str = DEFAULT_LABEL_FIELD,
) -> AttachPlan:
    """Match JSON files to samples and decide writes vs issue rows."""
    plan = AttachPlan()
    allowed = set(names)
    json_paths = iter_json_paths(label_dir)
    plan.label_files = len(json_paths)
    started = time.monotonic()
    logger.info("Found %d JSON files; parsing labels", len(json_paths))
    parsed_files = []
    requested_ids = set()
    requested_paths = set()
    paths_by_label = {}
    for index, json_path in enumerate(json_paths, 1):
        parsed, parse_issues = parse_xlabel_file(json_path, label_dir, allowed)
        plan.issues.extend(parse_issues)
        if parsed is not None:
            parsed_files.append(parsed)
            if parsed.sample_id:
                if ObjectId.is_valid(parsed.sample_id):
                    requested_ids.add(parsed.sample_id)
            elif images_dir is not None and parsed.image_path:
                path = Path(parsed.image_path).expanduser()
                if not path.is_absolute():
                    path = images_dir / Path(parsed.label_relpath).parent / path
                # Query only this batch's lexical and canonical paths, never resolve the entire dataset.
                candidates = {os.path.abspath(path), str(path.resolve())}
                paths_by_label[parsed.label_relpath] = candidates
                requested_paths.update(candidates)
        if index % QUERY_BATCH_SIZE == 0:
            logger.info("Parsed JSON %d/%d", index, len(json_paths))
    logger.info("Parsed labels in %.2fs; querying %d IDs and %d paths",
                time.monotonic() - started, len(requested_ids), len(requested_paths))
    dataset_ids = set()
    path_ids: dict[str, set[str]] = {}
    ids = sorted(requested_ids)
    for offset in range(0, len(ids), QUERY_BATCH_SIZE):
        found = dataset.select(ids[offset:offset + QUERY_BATCH_SIZE]).values("id")
        dataset_ids.update(str(sid) for sid in found)
        logger.info("Queried IDs %d/%d", min(offset + QUERY_BATCH_SIZE, len(ids)), len(ids))
    paths = sorted(requested_paths)
    for offset in range(0, len(paths), QUERY_BATCH_SIZE):
        view = dataset.match({"filepath": {"$in": paths[offset:offset + QUERY_BATCH_SIZE]}})
        found_ids, found_paths = view.values(["id", "filepath"])
        for sid, filepath in zip(found_ids, found_paths):
            sid = str(sid)
            path_ids.setdefault(filepath, set()).add(sid)
            dataset_ids.add(sid)
        logger.info("Queried paths %d/%d", min(offset + QUERY_BATCH_SIZE, len(paths)), len(paths))
    pending: list[ParsedBoxes] = []
    matched_ids: list[str] = []
    seen_ids: dict[str, str] = {}
    for parsed in parsed_files:
        if not parsed.sample_id and images_dir is not None:
            if not parsed.image_path:
                plan.issues.append(issue_row("missing_image_path", label_relpath=parsed.label_relpath))
                continue
            paths = paths_by_label[parsed.label_relpath]
            candidates = set().union(*(path_ids.get(path, set()) for path in paths))
            if len(candidates) != 1:
                plan.issues.append(issue_row(
                    "orphan_image_path" if not candidates else "ambiguous_image_path",
                    label_relpath=parsed.label_relpath,
                    detail=", ".join(sorted(paths)),
                ))
                continue
            parsed = replace(parsed, sample_id=next(iter(candidates)))
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
        if parsed.sample_id in seen_ids:
            plan.issues.append(
                issue_row(
                    "sample_id_collision",
                    sample_id=parsed.sample_id,
                    label_relpath=parsed.label_relpath,
                    detail=f"already matched {seen_ids[parsed.sample_id]}",
                )
            )
            continue
        seen_ids[parsed.sample_id] = parsed.label_relpath
        plan.matched += 1
        matched_ids.append(parsed.sample_id)
        pending.append(parsed)
    logger.info("Matched %d samples; loading existing boxes", plan.matched)
    existing = load_existing_boxes(dataset, matched_ids, label_field)
    for parsed in pending:
        current = existing.get(parsed.sample_id)
        new_boxes = parsed.boxes
        if boxes_equal(current, new_boxes, allowed, parsed.width, parsed.height):
            plan.unchanged += 1
            plan.to_tag[parsed.sample_id] = parsed
            continue
        plan.to_write[parsed.sample_id] = parsed
    logger.info("Plan finished in %.2fs: changed=%d unchanged=%d issues=%d",
                time.monotonic() - started, len(plan.to_write), plan.unchanged, len(plan.issues))
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


def merge_tags(
    existing: object,
    extra: list[str],
    checked: bool,
    *,
    boxes_changed: bool,
) -> list[str]:
    """Tag only changed samples with batch tags and ``changed``; strip them otherwise."""
    tags = [str(tag) for tag in (existing or [])]
    drop = set(extra)
    drop.add(CHANGED_TAG)
    if boxes_changed:
        tags = [tag for tag in tags if tag not in drop]
        tags.extend(extra)
        tags.append(CHANGED_TAG)
        if checked and XLABEL_CHECKED_TAG not in tags:
            tags.append(XLABEL_CHECKED_TAG)
        return tags
    return [tag for tag in tags if tag not in drop]


def replace_class_detections(
    existing: object,
    new_boxes: tuple[tuple[str, tuple[float, float, float, float]], ...],
    names: set[str],
) -> fo.Detections:
    """Drop detections in ``names``, then append JSON boxes for those classes."""
    kept: list = []
    detections = getattr(existing, "detections", None) or []
    for detection in detections:
        if str(detection.label) not in names:
            kept.append(detection)
    kept.extend(
        fo.Detection(label=label, bounding_box=list(bbox)) for label, bbox in new_boxes
    )
    return fo.Detections(detections=kept)


def apply_writes(
    dataset: fo.Dataset,
    to_write: dict[str, ParsedBoxes],
    extra_tags: list[str],
    names: set[str],
    label_field: str = DEFAULT_LABEL_FIELD,
) -> int:
    """Replace --class-names boxes and add batch tags. Returns samples updated."""
    if not to_write:
        return 0
    written = 0
    for sample in dataset.select(list(to_write)).iter_samples(autosave=True):
        parsed = to_write[str(sample.id)]
        current = sample[label_field] if dataset.has_field(label_field) else None
        sample[label_field] = replace_class_detections(current, parsed.boxes, names)
        sample[LABEL_RELPATH_FIELD] = parsed.label_relpath
        sample.tags = merge_tags(
            sample.tags, extra_tags, parsed.checked, boxes_changed=True
        )
        written += 1
    return written


def apply_tags(
    dataset: fo.Dataset,
    to_tag: dict[str, ParsedBoxes],
    extra_tags: list[str],
) -> int:
    """Strip batch tags when boxes did not change. Returns samples updated."""
    if not to_tag:
        return 0
    tagged = 0
    for sample in dataset.select(list(to_tag)).iter_samples(autosave=True):
        parsed = to_tag[str(sample.id)]
        sample.tags = merge_tags(
            sample.tags, extra_tags, parsed.checked, boxes_changed=False
        )
        tagged += 1
    return tagged


def print_report(values: dict[str, object]) -> None:
    """Print key=value lines in insertion order."""
    for key, value in values.items():
        print(f"{key}={value}")


def run_attach(
    dataset: fo.Dataset,
    label_dir: Path,
    names: list[str],
    extra_tags: list[str],
    dry_run: bool,
    images_dir: Path | None = None,
    label_field: str = DEFAULT_LABEL_FIELD,
) -> int:
    """Plan, write the issue CSV, and optionally apply detections."""
    plan = build_attach_plan(dataset, label_dir, names, images_dir, label_field)
    csv_path = issue_csv_path(dataset.name)
    write_issue_csv(csv_path, plan.issues)
    logger.info("Wrote issue CSV %s rows=%s", csv_path, len(plan.issues))
    print_report(
        {
            "mode": "dry-run" if dry_run else "attach",
            "dataset_name": dataset.name,
            "label_dir": str(label_dir),
            "class_names": ",".join(names),
            "label_field": label_field,
            "tags": ",".join(extra_tags),
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
    written = apply_writes(
        dataset, plan.to_write, extra_tags, set(names), label_field
    )
    tagged = apply_tags(dataset, plan.to_tag, extra_tags)
    dataset.save()
    print_report({"written": written, "unchanged_tags_cleared": tagged, "attach_done": "true"})
    return 0


def main(argv: list[str] | None = None) -> int:
    """Attach X-AnyLabeling JSON to an existing dataset."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)7s] %(name)s:%(lineno)d - %(message)s",
    )
    args = parse_args(argv)
    label_dir = args.label_dir.expanduser().resolve()
    if not label_dir.is_dir():
        logger.error("Not a directory: %s", label_dir)
        return 1
    images_dir = args.images_dir.expanduser().resolve() if args.images_dir else None
    if images_dir is not None and not images_dir.is_dir():
        logger.error("Not a directory: %s", images_dir)
        return 1
    if not fo.dataset_exists(args.dataset_name):
        logger.error("Dataset does not exist: %s", args.dataset_name)
        return 1
    dataset = fo.load_dataset(args.dataset_name)
    return run_attach(
        dataset,
        label_dir,
        args.class_names,
        extra_tags=args.tags,
        dry_run=args.dry_run,
        images_dir=images_dir,
        label_field=args.label_field,
    )


if __name__ == "__main__":
    sys.exit(main())
