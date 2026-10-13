"""Rewrite label tags from detection fields after a CVAT load.

``--fields age:0-2`` reads field ``age`` and replaces that box's ``age_*`` tags
with one tag, ``age_<value>``. Other tags stay. A sample is tagged with
``--sample-tags`` and ``changed`` when any requested box's tag family differs
from its field. A missing or unexpected field value is reported and that box's
tags are left as they are.
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
FIELD_SPEC_PATTERN = re.compile(r"([A-Za-z][A-Za-z0-9_]*):(\d+)-(\d+)")


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
    """Parse CLI arguments for rewriting label tags from fields."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=nonempty)
    parser.add_argument("--anno-key", required=True, type=nonempty)
    parser.add_argument(
        "--classes",
        required=True,
        type=name_list,
        help="Detection labels whose tags are rewritten from fields.",
    )
    parser.add_argument(
        "--fields",
        required=True,
        action="append",
        type=field_spec,
        help="Field read back into tags: name:low-high. Repeat to read more fields.",
    )
    parser.add_argument(
        "--sample-tags",
        required=True,
        type=name_list,
        help="Tags added, with 'changed', on samples whose field and tags disagree.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and write the issue CSV without changing tags.",
    )
    return parser.parse_args(argv)


def label_field_from_run(info: object) -> str:
    """Return the single label field recorded on an annotation run."""
    config = getattr(info, "config", None)
    schema = getattr(config, "label_schema", None) or {}
    if len(schema) != 1:
        raise ValueError("Annotation run must contain exactly one label field")
    return str(next(iter(schema)))


def tags_for_value(tags: object, spec: FieldSpec, value: str) -> tuple[list[str], bool]:
    """Replace one tag family with ``name_<value>``. Other tags stay in place."""
    prefix = f"{spec.name}_"
    expected = f"{spec.name}_{value}"
    kept = []
    family = []
    for tag in list(tags or []):
        if isinstance(tag, str) and tag.startswith(prefix) and tag[len(prefix):].isdigit():
            family.append(tag)
        else:
            kept.append(tag)
    if family == [expected]:
        return list(tags or []), False
    return [*kept, expected], True


def issue_row(
    kind: str,
    sample_id: str,
    label: str,
    detection_id: str,
    field: str,
    detail: str,
) -> dict[str, str]:
    """One CSV row for a box whose tags were not rewritten."""
    return {
        "issue": kind,
        "sample_id": sample_id,
        "label": label,
        "detection_id": detection_id,
        "field": field,
        "detail": detail,
    }


def plan_detection(
    detection: fo.Detection,
    names: set[str],
    specs: list[FieldSpec],
) -> tuple[list[str] | None, list[dict[str, str]]]:
    """Return replacement tags when they differ, plus issue rows."""
    if str(detection.label) not in names:
        return None, []
    tags = list(detection.tags or [])
    issues: list[dict[str, str]] = []
    detection_id = str(getattr(detection, "id", "") or "")
    changed = False
    for spec in specs:
        raw = detection.get_attribute_value(spec.name, None)
        if raw is None or str(raw).strip() == "":
            issues.append(issue_row("missing_attribute", "", str(detection.label), detection_id, spec.name, ""))
            continue
        value = str(raw).strip()
        if value not in spec.values:
            issues.append(
                issue_row("unexpected_value", "", str(detection.label), detection_id, spec.name, value)
            )
            continue
        tags, field_changed = tags_for_value(tags, spec, value)
        changed = changed or field_changed
    if not changed:
        return None, issues
    return tags, issues


def append_changed_tags(existing: object, extra: list[str]) -> list[str]:
    """Append batch tags and ``changed`` without dropping existing tags."""
    tags = [str(tag) for tag in (existing or [])]
    for tag in [*extra, CHANGED_TAG]:
        if tag not in tags:
            tags.append(tag)
    return tags


def apply_tag_sync(
    dataset: fo.Dataset,
    sample_ids: list[str],
    label_field: str,
    names: set[str],
    specs: list[FieldSpec],
    extra_tags: list[str],
    dry_run: bool,
) -> tuple[int, int, list[dict[str, str]]]:
    """Rewrite tags from fields. Returns changed samples, unchanged samples, and issues."""
    changed = 0
    unchanged = 0
    issues: list[dict[str, str]] = []
    for sample in dataset.select(sample_ids).iter_samples(autosave=not dry_run):
        label = sample[label_field] if dataset.has_field(label_field) else None
        detections = list(getattr(label, "detections", None) or [])
        sample_changed = False
        for detection in detections:
            if str(detection.label) not in names:
                continue
            new_tags, rows = plan_detection(detection, names, specs)
            for row in rows:
                row["sample_id"] = str(sample.id)
            issues.extend(rows)
            if new_tags is None:
                continue
            sample_changed = True
            if not dry_run:
                detection.tags = new_tags
        if sample_changed:
            changed += 1
            if not dry_run:
                sample[label_field] = label
                sample.tags = append_changed_tags(sample.tags, extra_tags)
        else:
            unchanged += 1
    if not dry_run:
        dataset.save()
    return changed, unchanged, issues


def write_issues(dataset_name: str, issues: list[dict[str, str]]) -> Path:
    """Write issue rows under ``tmp/``."""
    safe_name = re.sub(r"[^\w.-]", "_", dataset_name)
    output = Path.cwd() / "tmp" / f"update_tags_from_attributes_{safe_name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["issue", "sample_id", "label", "detection_id", "field", "detail"],
        )
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
    """Rewrite label tags from fields on one annotation run."""
    require_specs(args.fields)
    if not fo.dataset_exists(args.dataset):
        raise ValueError(f"Dataset does not exist: {args.dataset}")
    dataset = fo.load_dataset(args.dataset)
    if args.anno_key not in dataset.list_annotation_runs():
        raise ValueError(f"Annotation run does not exist: {args.anno_key}")
    label_field = label_field_from_run(dataset.get_annotation_info(args.anno_key))
    view = dataset.load_annotation_view(args.anno_key)
    sample_ids = [str(sample_id) for sample_id in view.values("id")]
    if not sample_ids:
        raise ValueError(f"Annotation run '{args.anno_key}' has no samples.")
    changed, unchanged, issues = apply_tag_sync(
        dataset,
        sample_ids,
        label_field,
        set(args.classes),
        args.fields,
        args.sample_tags,
        args.dry_run,
    )
    report = write_issues(args.dataset, issues)
    print_report(
        {
            "mode": "dry-run" if args.dry_run else "update",
            "dataset": args.dataset,
            "anno_key": args.anno_key,
            "label_field": label_field,
            "classes": ",".join(args.classes),
            "fields": ",".join(f"{spec.name}:{spec.low}-{spec.high}" for spec in args.fields),
            "sample_tags": ",".join(args.sample_tags),
            "samples": len(sample_ids),
            "changed": changed,
            "unchanged": unchanged,
            "issues": len(issues),
            "csv_path": report,
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Rewrite label tags from detection fields."""
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
