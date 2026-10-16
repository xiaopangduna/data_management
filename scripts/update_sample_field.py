"""Write one sample field from mutually exclusive sample tags.

``--field vendor --sample-tags aopeng,boyue`` sets ``vendor`` to whichever one
of those tags the sample has. A sample with none of them is left alone. Two or
more, or an existing different value, are reported and not written. Tags stay.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

import fiftyone as fo

logger = logging.getLogger(__name__)

FIELD_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
RESERVED_FIELDS = frozenset({"id", "filepath", "tags", "metadata", "created_at", "last_modified_at"})
ISSUE_COLUMNS = ("issue", "sample_id", "field", "current", "detail")


def nonempty(value: str) -> str:
    """Reject blank CLI strings."""
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("must not be empty")
    return stripped


def field_name(value: str) -> str:
    """Accept a FiftyOne sample field name."""
    name = nonempty(value)
    if FIELD_NAME_PATTERN.fullmatch(name) is None:
        raise argparse.ArgumentTypeError("use a letter followed by letters, digits, or underscores")
    if name in RESERVED_FIELDS:
        raise argparse.ArgumentTypeError(f"{name} is reserved")
    return name


def name_list(value: str) -> list[str]:
    """Parse a comma-separated list, preserving first-seen order."""
    names = list(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not names:
        raise argparse.ArgumentTypeError("provide at least one name")
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for copying sample tags onto a field."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=nonempty)
    parser.add_argument("--field", required=True, type=field_name, help="Sample field to write.")
    parser.add_argument(
        "--sample-tags",
        required=True,
        type=name_list,
        help="Mutually exclusive tags. Exactly one is copied into --field as the value.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and write the issue CSV without changing the dataset.",
    )
    return parser.parse_args(argv)


def matched_tags(tags: object, source: list[str]) -> list[str]:
    """Return source tags present on the sample, in source order."""
    present = {tag for tag in tags or [] if isinstance(tag, str)}
    return [tag for tag in source if tag in present]


def plan_sample(
    tags: object,
    source: list[str],
    field: str,
    current: object,
) -> tuple[str | None, dict[str, str] | None]:
    """Return the value to write, or an issue row. ``(None, None)`` means skip."""
    found = matched_tags(tags, source)
    if not found:
        return None, None
    if len(found) > 1:
        return None, {
            "issue": "multiple_tags",
            "sample_id": "",
            "field": field,
            "current": "" if current is None else str(current),
            "detail": ",".join(found),
        }
    value = found[0]
    if current is None or current == "":
        return value, None
    if str(current) == value:
        return None, None
    return None, {
        "issue": "field_mismatch",
        "sample_id": "",
        "field": field,
        "current": str(current),
        "detail": value,
    }


def current_value(sample: fo.Sample, field: str, dataset_has_field: bool) -> object:
    """Return the sample field, or None when it is absent."""
    if not dataset_has_field or not sample.has_field(field):
        return None
    return sample[field]


def apply_field(
    dataset: fo.Dataset,
    sample_ids: list[str],
    field: str,
    source: list[str],
    dry_run: bool,
) -> tuple[int, int, list[dict[str, str]]]:
    """Write the field. Returns samples updated, samples already set, and issues."""
    updated = 0
    unchanged = 0
    issues: list[dict[str, str]] = []
    dataset_has_field = dataset.has_field(field)
    for sample in dataset.select(sample_ids).iter_samples(autosave=not dry_run):
        current = current_value(sample, field, dataset_has_field)
        value, row = plan_sample(sample.tags, source, field, current)
        if row is not None:
            row["sample_id"] = str(sample.id)
            issues.append(row)
            continue
        if value is None:
            if matched_tags(sample.tags, source):
                unchanged += 1
            continue
        updated += 1
        if dry_run:
            continue
        if not dataset_has_field:
            dataset.add_sample_field(field, fo.StringField)
            dataset_has_field = True
        sample[field] = value
    if not dry_run:
        dataset.save()
    return updated, unchanged, issues


def write_issues(dataset_name: str, issues: list[dict[str, str]]) -> Path:
    """Write issue rows under ``tmp/``."""
    safe_name = re.sub(r"[^\w.-]", "_", dataset_name)
    output = Path.cwd() / "tmp" / f"update_sample_field_{safe_name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ISSUE_COLUMNS))
        writer.writeheader()
        writer.writerows(issues)
    return output


def require_string_field(dataset: fo.Dataset, field: str) -> None:
    """Reject an existing field that cannot store the tag string."""
    if not dataset.has_field(field):
        return
    existing = dataset.get_field(field)
    if not isinstance(existing, fo.StringField):
        raise ValueError(f"Field '{field}' is not a string field")


def print_report(values: dict[str, object]) -> None:
    """Print key=value lines in insertion order."""
    for key, value in values.items():
        print(f"{key}={value}")


def run(args: argparse.Namespace) -> int:
    """Copy mutually exclusive sample tags onto one sample field."""
    if not fo.dataset_exists(args.dataset):
        raise ValueError(f"Dataset does not exist: {args.dataset}")
    dataset = fo.load_dataset(args.dataset)
    require_string_field(dataset, args.field)
    view = dataset.match_tags(args.sample_tags, bool=True, all=False)
    sample_ids = [str(sample_id) for sample_id in view.values("id")]
    if not sample_ids:
        raise ValueError("No samples matched --sample-tags.")
    updated, unchanged, issues = apply_field(
        dataset,
        sample_ids,
        args.field,
        args.sample_tags,
        args.dry_run,
    )
    report = write_issues(args.dataset, issues)
    print_report(
        {
            "mode": "dry-run" if args.dry_run else "update",
            "dataset": args.dataset,
            "field": args.field,
            "sample_tags": ",".join(args.sample_tags),
            "samples": len(sample_ids),
            "updated": updated,
            "unchanged": unchanged,
            "issues": len(issues),
            "csv_path": report,
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Copy mutually exclusive sample tags onto one sample field."""
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
