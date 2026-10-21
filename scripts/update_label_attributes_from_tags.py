"""Copy label tags such as ``age_2`` onto detection fields.

``--fields age:0-2`` writes field ``age`` with values ``0``, ``1``, or ``2``.
When several ``age_*`` tags are present, the highest number in range is kept.
A box with none of those tags gets the low end of the range. Tags outside the
range are reported and that field is left unset. Existing label tags are not
removed. Replacing a value already stored on the field adds the sample tag
``changed`` and records the old and new values. Filling a missing field does not.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import fiftyone as fo

logger = logging.getLogger(__name__)

CHANGED_TAG = "changed"
DEFAULT_LABEL_FIELD = "ground_truth"
FIELD_SPEC_PATTERN = re.compile(r"([A-Za-z][A-Za-z0-9_]*):(\d+)-(\d+)")
ISSUE_COLUMNS = ("issue", "sample_id", "label", "detection_id", "field", "old", "new", "detail")


@dataclass(frozen=True)
class FieldSpec:
    """One detection field and the inclusive integer values it may hold."""

    name: str
    low: int
    high: int

    @property
    def values(self) -> list[str]:
        """Allowed field values, as strings."""
        return [str(item) for item in range(self.low, self.high + 1)]


def nonempty(value: str) -> str:
    """Reject blank CLI strings."""
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("must not be empty")
    return stripped


def name_list(value: str) -> list[str]:
    """Parse a comma-separated list, preserving first-seen order."""
    names = list(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not names:
        raise argparse.ArgumentTypeError("provide at least one name")
    return names


def field_spec(value: str) -> FieldSpec:
    """Parse ``name:low-high``, for example ``age:0-2``."""
    match = FIELD_SPEC_PATTERN.fullmatch(value.strip())
    if match is None:
        raise argparse.ArgumentTypeError("use name:low-high, for example age:0-2")
    low = int(match.group(2))
    high = int(match.group(3))
    if low > high:
        raise argparse.ArgumentTypeError("range start must be <= end")
    return FieldSpec(match.group(1), low, high)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for copying label tags onto fields."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=nonempty)
    parser.add_argument(
        "--sample-tags",
        required=True,
        type=name_list,
        help="Require ALL of these sample tags.",
    )
    parser.add_argument(
        "--classes",
        required=True,
        type=name_list,
        help="Detection labels whose tags are copied onto fields.",
    )
    parser.add_argument(
        "--fields",
        required=True,
        action="append",
        type=field_spec,
        help="Field written from tags: name:low-high. Repeat to write more fields.",
    )
    parser.add_argument(
        "--label-field",
        default=DEFAULT_LABEL_FIELD,
        type=nonempty,
        help=f"Detections field to update (default: {DEFAULT_LABEL_FIELD}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and write the CSV without changing fields or sample tags.",
    )
    return parser.parse_args(argv)


def family_codes(tags: object, name: str) -> list[tuple[str, int]]:
    """Return ``name_<digits>`` tags and their numbers."""
    prefix = f"{name}_"
    found = []
    for tag in tags or []:
        if not isinstance(tag, str) or not tag.startswith(prefix):
            continue
        suffix = tag[len(prefix):]
        if suffix.isdigit():
            found.append((tag, int(suffix)))
    return found


def attribute_from_tags(tags: object, spec: FieldSpec) -> tuple[str | None, list[str]]:
    """Return the field value, or None when a family tag falls outside the range."""
    found = family_codes(tags, spec.name)
    outside = [tag for tag, code in found if code < spec.low or code > spec.high]
    if outside:
        return None, outside
    inside = [code for _, code in found]
    if not inside:
        return str(spec.low), []
    return str(max(inside)), []


def issue_row(
    sample_id: str,
    label: str,
    detection_id: str,
    field: str,
    detail: str,
) -> dict[str, str]:
    """One CSV row for a box whose field was not written."""
    return {
        "issue": "out_of_range",
        "sample_id": sample_id,
        "label": label,
        "detection_id": detection_id,
        "field": field,
        "old": "",
        "new": "",
        "detail": detail,
    }


def overwrite_row(
    label: str,
    detection_id: str,
    field: str,
    old: str,
    new: str,
) -> dict[str, str]:
    """One CSV row for a field whose stored value was replaced."""
    return {
        "issue": "overwritten",
        "sample_id": "",
        "label": label,
        "detection_id": detection_id,
        "field": field,
        "old": old,
        "new": new,
        "detail": "",
    }


def stored_value(current: object) -> str | None:
    """Return the stored field text, or None when the field is missing."""
    if current is None:
        return None
    text = str(current).strip()
    if not text:
        return None
    return text


def append_changed_tag(existing: object) -> list[str]:
    """Append ``changed`` without dropping existing sample tags."""
    tags = [str(tag) for tag in (existing or [])]
    if CHANGED_TAG not in tags:
        tags.append(CHANGED_TAG)
    return tags


def plan_detection(
    detection: fo.Detection,
    names: set[str],
    specs: list[FieldSpec],
) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
    """Return field writes, issue rows, and overwrite rows. Sample id is filled later."""
    if str(detection.label) not in names:
        return {}, [], []
    writes: dict[str, str] = {}
    issues: list[dict[str, str]] = []
    overwrites: list[dict[str, str]] = []
    detection_id = str(getattr(detection, "id", "") or "")
    label = str(detection.label)
    for spec in specs:
        value, outside = attribute_from_tags(detection.tags, spec)
        if outside:
            issues.append(issue_row("", label, detection_id, spec.name, ",".join(outside)))
            continue
        current = stored_value(detection.get_attribute_value(spec.name, None))
        if current is None:
            writes[spec.name] = value
        elif current != value:
            writes[spec.name] = value
            overwrites.append(overwrite_row(label, detection_id, spec.name, current, value))
    return writes, issues, overwrites


def apply_attributes(
    dataset: fo.Dataset,
    sample_ids: list[str],
    label_field: str,
    names: set[str],
    specs: list[FieldSpec],
    dry_run: bool,
) -> tuple[int, int, list[dict[str, str]], list[dict[str, str]]]:
    """Write planned fields. Returns boxes updated, boxes unchanged, issues, and overwrites."""
    updated = 0
    unchanged = 0
    issues: list[dict[str, str]] = []
    overwrites: list[dict[str, str]] = []
    for sample in dataset.select(sample_ids).iter_samples(autosave=not dry_run):
        label = sample[label_field] if dataset.has_field(label_field) else None
        detections = list(getattr(label, "detections", None) or [])
        sample_changed = False
        replaced = False
        for detection in detections:
            if str(detection.label) not in names:
                continue
            writes, rows, replaced_rows = plan_detection(detection, names, specs)
            for row in rows + replaced_rows:
                row["sample_id"] = str(sample.id)
            issues.extend(rows)
            overwrites.extend(replaced_rows)
            if not writes:
                unchanged += 1
                continue
            updated += 1
            replaced = replaced or bool(replaced_rows)
            if dry_run:
                continue
            for field_name, value in writes.items():
                detection.set_attribute_value(field_name, value)
            sample_changed = True
        if sample_changed:
            sample[label_field] = label
            if replaced:
                sample.tags = append_changed_tag(sample.tags)
    if not dry_run:
        dataset.save()
    return updated, unchanged, issues, overwrites


def write_issues(dataset_name: str, issues: list[dict[str, str]]) -> Path:
    """Write issue rows under ``tmp/``."""
    safe_name = re.sub(r"[^\w.-]", "_", dataset_name)
    output = Path.cwd() / "tmp" / f"update_label_attributes_{safe_name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ISSUE_COLUMNS))
        writer.writeheader()
        writer.writerows(issues)
    return output


def require_specs(specs: list[FieldSpec]) -> None:
    """Reject a repeated field name."""
    seen = set()
    for spec in specs:
        if spec.name in seen:
            raise ValueError(f"Duplicate field: {spec.name}")
        seen.add(spec.name)


def print_report(values: dict[str, object]) -> None:
    """Print key=value lines in insertion order."""
    for key, value in values.items():
        print(f"{key}={value}")


def run(args: argparse.Namespace) -> int:
    """Copy label tags onto detection fields for the selected samples."""
    require_specs(args.fields)
    if not fo.dataset_exists(args.dataset):
        raise ValueError(f"Dataset does not exist: {args.dataset}")
    dataset = fo.load_dataset(args.dataset)
    if args.label_field not in dataset.get_field_schema():
        raise ValueError(f"Dataset has no field: {args.label_field}")
    field = dataset.get_field(args.label_field)
    if not issubclass(field.document_type, fo.Detections):
        raise ValueError(f"Field '{args.label_field}' is not Detections")
    view = dataset.match_tags(args.sample_tags, bool=True, all=True)
    sample_ids = [str(sample_id) for sample_id in view.values("id")]
    if not sample_ids:
        raise ValueError("No samples matched --sample-tags.")
    updated, unchanged, issues, overwrites = apply_attributes(
        dataset,
        sample_ids,
        args.label_field,
        set(args.classes),
        args.fields,
        args.dry_run,
    )
    report = write_issues(args.dataset, issues + overwrites)
    print_report(
        {
            "mode": "dry-run" if args.dry_run else "update",
            "dataset": args.dataset,
            "label_field": args.label_field,
            "classes": ",".join(args.classes),
            "fields": ",".join(f"{spec.name}:{spec.low}-{spec.high}" for spec in args.fields),
            "samples": len(sample_ids),
            "updated_boxes": updated,
            "unchanged_boxes": unchanged,
            "overwritten": len(overwrites),
            "issues": len(issues),
            "csv_path": report,
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Copy label tags onto detection fields."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except Exception as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
