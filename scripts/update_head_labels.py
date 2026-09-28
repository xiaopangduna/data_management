"""Rename baby_head and adult_head boxes to head, keeping age label tags."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LABEL_TO_AGE = {
    "baby_head": "age_0",
    "adult_head": "age_2",
}
REPORT_COLUMNS = ("sample_id", "filepath", "det_index", "label", "tags", "extra_age")


def nonempty(value: str) -> str:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=nonempty)
    parser.add_argument("--label-field", default="ground_truth", type=nonempty)
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing the database.")
    return parser.parse_args(argv)


def renamed_tags(label: str, tags: list[str] | None) -> tuple[str, list[str]] | None:
    """Return ``(head, tags)`` for a renamed box. Other labels return None."""
    age = LABEL_TO_AGE.get(label)
    if age is None:
        return None
    new_tags = list(tags or [])
    if age not in new_tags:
        new_tags.append(age)
    return "head", new_tags


def extra_age_tags(tags: list[str] | None, expected: str) -> list[str]:
    """Return other ``age_<n>`` tags on a box that is about to be renamed."""
    extras = []
    for tag in tags or []:
        if not isinstance(tag, str) or not tag.startswith("age_"):
            continue
        suffix = tag[len("age_"):]
        if suffix.isdigit() and tag != expected:
            extras.append(tag)
    return extras


def rename_pipeline(label_field: str, now: datetime) -> list[dict]:
    """Mongo update pipeline. Copies every detection and renames only the two source labels."""
    tags = {"$ifNull": ["$$det.tags", []]}
    branches = []
    for label, age in LABEL_TO_AGE.items():
        branches.append({
            "case": {"$eq": ["$$det.label", label]},
            "then": {
                "$mergeObjects": [
                    "$$det",
                    {
                        "label": "head",
                        "tags": {
                            "$cond": {
                                "if": {"$in": [age, tags]},
                                "then": tags,
                                "else": {"$concatArrays": [tags, [age]]},
                            }
                        },
                    },
                ]
            },
        })
    return [{
        "$set": {
            "last_modified_at": now,
            f"{label_field}.detections": {
                "$map": {
                    "input": {"$ifNull": [f"${label_field}.detections", []]},
                    "as": "det",
                    "in": {"$switch": {"branches": branches, "default": "$$det"}},
                }
            },
        }
    }]


def report_path(dataset_name: str) -> Path:
    """Return ``tmp/update_head_labels_<dataset>.csv``."""
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in dataset_name).strip("._")
    directory = Path.cwd() / "tmp"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"update_head_labels_{safe or 'dataset'}.csv"


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REPORT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def collect_conflicts(collection, label_field: str) -> tuple[dict[str, int], list[dict[str, str]]]:
    """Count source boxes and list boxes that already carry another age tag."""
    counts = {label: 0 for label in LABEL_TO_AGE}
    counts["tags_added"] = 0
    rows: list[dict[str, str]] = []
    pipeline = [
        {"$match": {f"{label_field}.detections.label": {"$in": list(LABEL_TO_AGE)}}},
        {"$unwind": {"path": f"${label_field}.detections", "includeArrayIndex": "det_index"}},
        {"$match": {f"{label_field}.detections.label": {"$in": list(LABEL_TO_AGE)}}},
        {"$project": {
            "_id": 0,
            "sample_id": {"$toString": "$_id"},
            "filepath": {"$ifNull": ["$filepath", ""]},
            "det_index": 1,
            "label": f"${label_field}.detections.label",
            "tags": {"$ifNull": [f"${label_field}.detections.tags", []]},
        }},
    ]
    for doc in collection.aggregate(pipeline, allowDiskUse=True):
        label = doc["label"]
        tags = list(doc["tags"])
        counts[label] += 1
        renamed = renamed_tags(label, tags)
        if renamed is not None and renamed[1] != tags:
            counts["tags_added"] += 1
        extras = extra_age_tags(tags, LABEL_TO_AGE[label])
        if extras:
            rows.append({
                "sample_id": doc["sample_id"],
                "filepath": doc["filepath"],
                "det_index": str(doc["det_index"]),
                "label": label,
                "tags": ",".join(tags),
                "extra_age": ",".join(extras),
            })
    return counts, rows


def apply_rename(collection, label_field: str, now: datetime):
    """Rename matching boxes in place. Other detections on the same sample stay put."""
    return collection.update_many(
        {f"{label_field}.detections.label": {"$in": list(LABEL_TO_AGE)}},
        rename_pipeline(label_field, now),
    )


def update_dataset(args: argparse.Namespace) -> dict:
    import fiftyone as fo

    dataset = fo.load_dataset(args.dataset)
    field = dataset.get_field_schema().get(args.label_field)
    if not isinstance(field, fo.EmbeddedDocumentField) or field.document_type != fo.Detections:
        raise ValueError(f"Label field must be fo.Detections: {args.label_field}")
    collection = dataset._sample_collection
    counts, conflicts = collect_conflicts(collection, args.label_field)
    csv_path = report_path(args.dataset)
    write_report(csv_path, conflicts)
    summary = {
        "dataset": args.dataset,
        "label_field": args.label_field,
        "mode": "dry-run" if args.dry_run else "update",
        "samples": collection.count_documents(
            {f"{args.label_field}.detections.label": {"$in": list(LABEL_TO_AGE)}}
        ),
        "baby_head": counts["baby_head"],
        "adult_head": counts["adult_head"],
        "tags_added": counts["tags_added"],
        "extra_age_boxes": len(conflicts),
        "csv_path": str(csv_path),
    }
    if args.dry_run:
        return summary
    if summary["samples"] == 0:
        return summary
    result = apply_rename(collection, args.label_field, datetime.now(timezone.utc).replace(tzinfo=None))
    dataset.reload()
    summary["matched"] = result.matched_count
    summary["modified"] = result.modified_count
    summary["labels_after"] = dataset.count_values(f"{args.label_field}.detections.label")
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        summary = update_dataset(args)
    except Exception as exc:
        print(f"Update failed: {exc}", file=sys.stderr)
        return 1
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
