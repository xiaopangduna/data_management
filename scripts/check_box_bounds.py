"""Fix abnormal detection boxes and tag the sample.

FiftyOne stores boxes as relative top-left xywh. Overflow within
``--edge-tol`` counts as touching the border. A box past that tolerance is:

- clipped back inside the image when its width and height do not exceed the
  image and it sticks out by at most ``--clip-overflow``
- deleted when it misses the image, sticks out farther, or its width or
  height already exceeds the image
- left in place when the only issue is a same-label IoU at or above ``--iou``

The whole sample is tagged ``bad_box`` when any box is abnormal, and
``changed_box`` when this run clips or deletes a box. Other sample tags are
kept. A later run removes those two tags when they no longer apply.

Writes ``tmp/check_box_bounds_<dataset>.csv``. ``--dry-run`` writes the CSV
only and does not change the database.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_LABEL_FIELD = "ground_truth"
DEFAULT_EDGE_TOL = 0.001
DEFAULT_CLIP_OVERFLOW = 0.02
DEFAULT_IOU = 0.9
ISSUE_INVALID = "invalid"
ISSUE_NO_OVERLAP = "no_overlap"
ISSUE_PARTIAL = "partial"
ISSUE_HIGH_IOU = "high_iou"
ACTION_CLIP = "clip"
ACTION_DELETE = "delete"
ACTION_KEEP = "keep"
TAG_BAD = "bad_box"
TAG_CHANGED = "changed_box"
MANAGED_TAGS = (TAG_BAD, TAG_CHANGED)
REPORT_COLUMNS = (
    "issue",
    "action",
    "sample_id",
    "filepath",
    "tags",
    "det_index",
    "label",
    "x",
    "y",
    "w",
    "h",
    "overlap",
    "max_overflow",
    "iou",
    "pair_index",
)


def nonempty(value: str) -> str:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def tag_list(value: str) -> list[str]:
    tags = list(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not tags:
        raise argparse.ArgumentTypeError("provide at least one tag")
    return tags


def edge_tol(value: str) -> float:
    """Accept a border tolerance in ``[0, 0.5)``."""
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(number) or number < 0 or number >= 0.5:
        raise argparse.ArgumentTypeError("must be in [0, 0.5)")
    return number


def clip_overflow(value: str) -> float:
    """Accept a clip limit in ``(0, 0.5)``."""
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(number) or number <= 0 or number >= 0.5:
        raise argparse.ArgumentTypeError("must be in (0, 0.5)")
    return number


def iou_threshold(value: str) -> float:
    """Accept an IoU threshold in ``(0, 1]``."""
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(number) or number <= 0 or number > 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=nonempty)
    parser.add_argument("--label-field", default=DEFAULT_LABEL_FIELD, type=nonempty)
    parser.add_argument(
        "--sample-tags",
        type=tag_list,
        help="Only scan samples that have any of these tags.",
    )
    parser.add_argument(
        "--edge-tol",
        default=DEFAULT_EDGE_TOL,
        type=edge_tol,
        help=f"Ignore overflow within this fraction of the image (default: {DEFAULT_EDGE_TOL}).",
    )
    parser.add_argument(
        "--clip-overflow",
        default=DEFAULT_CLIP_OVERFLOW,
        type=clip_overflow,
        help=(
            "Clip boxes that stick out by at most this fraction when width and height "
            f"stay within the image (default: {DEFAULT_CLIP_OVERFLOW}). Larger overflow is deleted."
        ),
    )
    parser.add_argument(
        "--iou",
        default=DEFAULT_IOU,
        type=iou_threshold,
        help=f"Flag same-label pairs at or above this IoU (default: {DEFAULT_IOU}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the CSV only. Do not change the database.",
    )
    return parser.parse_args(argv)


def parse_box(box: object) -> tuple[float, float, float, float] | None:
    """Return finite xywh, or None when the value is not four finite numbers."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        values = tuple(float(part) for part in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(part) for part in values):
        return None
    return values  # type: ignore[return-value]


def overlap_fraction(x: float, y: float, width: float, height: float) -> float:
    """Return the share of the box area that lies inside the unit image."""
    area = width * height
    if area <= 0:
        return 0.0
    overlap_w = min(x + width, 1.0) - max(x, 0.0)
    overlap_h = min(y + height, 1.0) - max(y, 0.0)
    if overlap_w <= 0 or overlap_h <= 0:
        return 0.0
    return (overlap_w * overlap_h) / area


def max_overflow(x: float, y: float, width: float, height: float) -> float:
    """Return the largest distance the box extends past the unit image."""
    return max(0.0, -x, -y, x + width - 1.0, y + height - 1.0)


def box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    """Return intersection-over-union for two xywh boxes."""
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    overlap_w = min(lx + lw, rx + rw) - max(lx, rx)
    overlap_h = min(ly + lh, ry + rh) - max(ly, ry)
    if overlap_w <= 0 or overlap_h <= 0:
        return 0.0
    intersection = overlap_w * overlap_h
    union = lw * lh + rw * rh - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def classify_box(box: object, tol: float = DEFAULT_EDGE_TOL) -> str | None:
    """Return a bounds issue name, or None when the box is inside the tolerance."""
    parsed = parse_box(box)
    if parsed is None:
        return ISSUE_INVALID
    x, y, width, height = parsed
    if width <= 0 or height <= 0:
        return ISSUE_INVALID
    if x >= -tol and y >= -tol and x + width <= 1.0 + tol and y + height <= 1.0 + tol:
        return None
    if x >= 1.0 or y >= 1.0 or x + width <= 0.0 or y + height <= 0.0:
        return ISSUE_NO_OVERLAP
    return ISSUE_PARTIAL


def high_iou_partners(
    labels: list[object],
    boxes: list[object],
    iou_min: float,
) -> dict[int, tuple[int, float]]:
    """Map each box index to the same-label partner with the highest IoU."""
    grouped: dict[str, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for index, box in enumerate(boxes):
        label = labels[index] if index < len(labels) and labels[index] is not None else ""
        label = str(label)
        parsed = parse_box(box)
        if not label or parsed is None or parsed[2] <= 0 or parsed[3] <= 0:
            continue
        grouped.setdefault(label, []).append((index, parsed))
    partners: dict[int, tuple[int, float]] = {}
    for group in grouped.values():
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                score = box_iou(group[left][1], group[right][1])
                if score < iou_min:
                    continue
                left_index = group[left][0]
                right_index = group[right][0]
                for index, partner in ((left_index, right_index), (right_index, left_index)):
                    current = partners.get(index)
                    if current is None or score > current[1] or (score == current[1] and partner < current[0]):
                        partners[index] = (partner, score)
    return partners


@dataclass(frozen=True)
class BoxEdit:
    """One clip or delete, addressed by the original detection index."""

    index: int
    action: str
    box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class SamplePlan:
    """Issues, sample-tag replacement, and box edits for one sample."""

    issues: tuple[tuple[str, int, str, str, str, str], ...]
    tags: list[str] | None
    edits: tuple[BoxEdit, ...]


def clip_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    """Return the box clamped to the unit image, or None when nothing remains."""
    x, y, width, height = box
    x1 = min(max(x, 0.0), 1.0)
    y1 = min(max(y, 0.0), 1.0)
    x2 = min(max(x + width, 0.0), 1.0)
    y2 = min(max(y + height, 0.0), 1.0)
    clipped_w = round(x2 - x1, 6)
    clipped_h = round(y2 - y1, 6)
    if clipped_w <= 0 or clipped_h <= 0:
        return None
    return (round(x1, 6), round(y1, 6), clipped_w, clipped_h)


def bounds_action(
    box: object,
    tol: float = DEFAULT_EDGE_TOL,
    clip_limit: float = DEFAULT_CLIP_OVERFLOW,
) -> tuple[str, str] | None:
    """Return ``(issue, clip|delete)``, or None when the box is inside tolerance."""
    issue = classify_box(box, tol)
    if issue is None:
        return None
    if issue in {ISSUE_INVALID, ISSUE_NO_OVERLAP}:
        return issue, ACTION_DELETE
    parsed = parse_box(box)
    assert parsed is not None
    _x, _y, width, height = parsed
    overflow = max_overflow(*parsed)
    if width > 1 or height > 1 or overflow > clip_limit or clip_box(parsed) is None:
        return issue, ACTION_DELETE
    return issue, ACTION_CLIP


def merge_sample_tags(existing: object, *, bad: bool, changed: bool) -> list[str] | None:
    """Return sample tags with ``bad_box`` and ``changed_box`` set, or None if unchanged."""
    current = [str(tag) for tag in (existing or [])]
    desired = [tag for tag, enabled in ((TAG_BAD, bad), (TAG_CHANGED, changed)) if enabled]
    owned = set(MANAGED_TAGS)
    if [tag for tag in current if tag in owned] == desired:
        return None
    return [tag for tag in current if tag not in owned] + desired


def format_number(value: float) -> str:
    return f"{value:.6f}"


def coordinate_fields(box: object) -> tuple[str, str, str, str, str, str]:
    """Return x, y, w, h, overlap, max_overflow strings for one box."""
    parsed = parse_box(box)
    if parsed is None:
        return ("", "", "", "", "", "")
    coords = tuple(format_number(part) for part in parsed)
    if parsed[2] <= 0 or parsed[3] <= 0:
        return (*coords, "", "")
    return (
        *coords,
        format_number(overlap_fraction(*parsed)),
        format_number(max_overflow(*parsed)),
    )


def plan_sample(
    labels: object,
    boxes: object,
    sample_tags: object,
    tol: float,
    iou_min: float = DEFAULT_IOU,
    clip_limit: float = DEFAULT_CLIP_OVERFLOW,
) -> SamplePlan:
    """Decide sample tags and which boxes to clip or delete."""
    label_list = list(labels or [])
    box_list = list(boxes or [])
    partners = high_iou_partners(label_list, box_list, iou_min)
    issues: list[tuple[str, int, str, str, str, str]] = []
    edits: list[BoxEdit] = []
    for index, box in enumerate(box_list):
        label = label_list[index] if index < len(label_list) and label_list[index] is not None else ""
        label = str(label)
        action = ACTION_KEEP
        decided = bounds_action(box, tol, clip_limit)
        if decided is not None:
            issue, action = decided
            issues.append((issue, index, label, "", "", action))
            if action == ACTION_CLIP:
                parsed = parse_box(box)
                assert parsed is not None
                clipped = clip_box(parsed)
                edits.append(BoxEdit(index, ACTION_CLIP, clipped))
            else:
                edits.append(BoxEdit(index, ACTION_DELETE))
        partner = partners.get(index)
        if partner is not None:
            issues.append((
                ISSUE_HIGH_IOU,
                index,
                label,
                format_number(partner[1]),
                str(partner[0]),
                action,
            ))
    tags = merge_sample_tags(sample_tags, bad=bool(issues), changed=bool(edits))
    return SamplePlan(tuple(issues), tags, tuple(edits))


def rows_for_sample(
    sample_id: str,
    filepath: str,
    tags: object,
    labels: object,
    boxes: object,
    tol: float,
    iou_min: float = DEFAULT_IOU,
    clip_limit: float = DEFAULT_CLIP_OVERFLOW,
) -> list[dict[str, str]]:
    """Return one CSV row for each abnormal box issue on a sample."""
    tag_text = ",".join(str(tag) for tag in (tags or []))
    box_list = list(boxes or [])
    plan = plan_sample(labels, boxes, tags, tol, iou_min, clip_limit)
    rows: list[dict[str, str]] = []
    for issue, index, label, iou, pair_index, action in plan.issues:
        x, y, width, height, overlap, overflow = coordinate_fields(
            box_list[index] if index < len(box_list) else None
        )
        rows.append({
            "issue": issue,
            "action": action,
            "sample_id": sample_id,
            "filepath": filepath or "",
            "tags": tag_text,
            "det_index": str(index),
            "label": label,
            "x": x,
            "y": y,
            "w": width,
            "h": height,
            "overlap": overlap,
            "max_overflow": overflow,
            "iou": iou,
            "pair_index": pair_index,
        })
    return rows


def report_path(dataset_name: str) -> Path:
    """Return ``tmp/check_box_bounds_<dataset>.csv``."""
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in dataset_name).strip("._")
    directory = Path.cwd() / "tmp"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"check_box_bounds_{safe or 'dataset'}.csv"


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REPORT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def collect_findings(
    ids: list[str],
    filepaths: list[str],
    tags: list[object],
    labels: list[object],
    boxes: list[object],
    tol: float,
    iou_min: float,
    clip_limit: float,
) -> tuple[list[dict[str, str]], dict[str, SamplePlan]]:
    """Classify samples. Returns CSV rows and plans that change boxes or sample tags."""
    rows: list[dict[str, str]] = []
    updates: dict[str, SamplePlan] = {}
    for sample_id, filepath, sample_tags, sample_labels, sample_boxes in zip(
        ids, filepaths, tags, labels, boxes, strict=True
    ):
        plan = plan_sample(sample_labels, sample_boxes, sample_tags, tol, iou_min, clip_limit)
        if plan.tags is not None or plan.edits:
            updates[str(sample_id)] = plan
        if not plan.issues:
            continue
        tag_text = ",".join(str(tag) for tag in (sample_tags or []))
        box_list = list(sample_boxes or [])
        for issue, index, label, iou, pair_index, action in plan.issues:
            x, y, width, height, overlap, overflow = coordinate_fields(
                box_list[index] if index < len(box_list) else None
            )
            rows.append({
                "issue": issue,
                "action": action,
                "sample_id": str(sample_id),
                "filepath": filepath or "",
                "tags": tag_text,
                "det_index": str(index),
                "label": label,
                "x": x,
                "y": y,
                "w": width,
                "h": height,
                "overlap": overlap,
                "max_overflow": overflow,
                "iou": iou,
                "pair_index": pair_index,
            })
    return rows, updates


def summarize(rows: list[dict[str, str]], updates: dict[str, SamplePlan]) -> dict[str, int]:
    """Count issues, box edits, and samples that receive each sample tag."""
    names = (ISSUE_INVALID, ISSUE_NO_OVERLAP, ISSUE_PARTIAL, ISSUE_HIGH_IOU)
    counts = {name: 0 for name in names}
    actions = {ACTION_CLIP: set(), ACTION_DELETE: set()}
    for row in rows:
        issue = row["issue"]
        counts[issue] = counts.get(issue, 0) + 1
        key = (row["sample_id"], row["det_index"])
        if row["action"] in actions:
            actions[row["action"]].add(key)
    bad_samples = {row["sample_id"] for row in rows}
    changed_samples = {
        sample_id for sample_id, plan in updates.items() if plan.edits
    }
    return {
        "issue_boxes": len({(row["sample_id"], row["det_index"]) for row in rows}),
        "issue_samples": len(bad_samples),
        "invalid": counts.get(ISSUE_INVALID, 0),
        "no_overlap": counts.get(ISSUE_NO_OVERLAP, 0),
        "partial": counts.get(ISSUE_PARTIAL, 0),
        "high_iou": counts.get(ISSUE_HIGH_IOU, 0),
        "clip_boxes": len(actions[ACTION_CLIP]),
        "delete_boxes": len(actions[ACTION_DELETE]),
        "bad_box_samples": len(bad_samples),
        "changed_box_samples": len(changed_samples),
    }


def apply_plans(dataset, label_field: str, updates: dict[str, SamplePlan]) -> int:
    """Clip or delete boxes and write sample tags. Returns samples saved."""
    if not updates:
        return 0
    saved = 0
    for sample in dataset.select(list(updates)).iter_samples(autosave=True):
        plan = updates.get(str(sample.id))
        if plan is None:
            continue
        if plan.edits:
            label = sample[label_field] if sample.has_field(label_field) else None
            if label is None:
                logger.warning("Skip %s: missing %s", sample.id, label_field)
                continue
            detections = list(getattr(label, "detections", None) or [])
            if any(edit.index >= len(detections) for edit in plan.edits):
                logger.warning("Skip %s: detection count changed", sample.id)
                continue
            by_index = {edit.index: edit for edit in plan.edits}
            kept = []
            for index, detection in enumerate(detections):
                edit = by_index.get(index)
                if edit is None:
                    kept.append(detection)
                elif edit.action == ACTION_CLIP and edit.box is not None:
                    detection.bounding_box = list(edit.box)
                    kept.append(detection)
            label.detections = kept
            sample[label_field] = label
        if plan.tags is not None:
            sample.tags = plan.tags
        saved += 1
        if saved % 1000 == 0:
            logger.info("Updated samples %d/%d", saved, len(updates))
    return saved


def scan_dataset(args: argparse.Namespace) -> dict[str, object]:
    import fiftyone as fo

    if not fo.dataset_exists(args.dataset):
        raise ValueError(f"Dataset does not exist: {args.dataset}")
    dataset = fo.load_dataset(args.dataset)
    field = dataset.get_field_schema().get(args.label_field)
    if not isinstance(field, fo.EmbeddedDocumentField) or field.document_type != fo.Detections:
        raise ValueError(f"Label field must be fo.Detections: {args.label_field}")
    view = dataset.match_tags(args.sample_tags) if args.sample_tags else dataset
    label_key = f"{args.label_field}.detections.label"
    box_key = f"{args.label_field}.detections.bounding_box"
    ids, filepaths, tags, labels, boxes = view.values(
        ["id", "filepath", "tags", label_key, box_key]
    )
    rows, updates = collect_findings(
        ids, filepaths, tags, labels, boxes, args.edge_tol, args.iou, args.clip_overflow
    )
    csv_path = report_path(args.dataset)
    write_report(csv_path, rows)
    summary: dict[str, object] = {
        "dataset": args.dataset,
        "label_field": args.label_field,
        "edge_tol": args.edge_tol,
        "clip_overflow": args.clip_overflow,
        "iou": args.iou,
        "mode": "dry-run" if args.dry_run else "update",
        "samples": len(ids),
        "to_update_samples": len(updates),
        "csv_path": str(csv_path),
    }
    if args.sample_tags:
        summary["sample_tags"] = ",".join(args.sample_tags)
    summary.update(summarize(rows, updates))
    if not args.dry_run:
        summary["updated_samples"] = apply_plans(dataset, args.label_field, updates)
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        summary = scan_dataset(args)
    except Exception as exc:
        print(f"Check failed: {exc}", file=sys.stderr)
        return 1
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
