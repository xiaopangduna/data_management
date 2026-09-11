"""Append sample tags to a FiftyOne dataset by matching stems from a CSV."""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True, type=nonempty)
    parser.add_argument("--csv-path", required=True, type=Path)
    parser.add_argument("--column", default="filename", help="CSV filename/path column")
    parser.add_argument("--tags", required=True, type=tag_list, help="Comma-separated tags")
    parser.add_argument(
        "--allow-ambiguous",
        action="store_true",
        help="Tag all samples when a stem occurs more than once in the dataset",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report without changing FiftyOne")
    return parser.parse_args(argv)


def value_stem(value: str) -> str:
    filename = Path(value.strip().replace("\\", "/")).name
    return Path(filename).stem


def read_csv_stems(csv_path: Path, column: str) -> tuple[set[str], int]:
    with csv_path.expanduser().resolve(strict=True).open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        if column not in reader.fieldnames:
            raise ValueError(f"CSV has no column {column!r}; columns: {', '.join(reader.fieldnames)}")
        values = [row[column] for row in reader if row.get(column, "").strip()]
    stems = {value_stem(value) for value in values}
    return stems, len(values)


def build_plan(rows, csv_stems: set[str], tags: list[str], allow_ambiguous: bool):
    by_stem = defaultdict(list)
    for sample_id, filepath, existing_tags in rows:
        by_stem[Path(filepath).stem].append((str(sample_id), existing_tags or [], filepath))

    updates = set()
    matched_stems = set()
    issues = []
    matched_samples = 0
    unchanged = 0
    for stem in sorted(csv_stems):
        candidates = by_stem.get(stem, [])
        if not candidates:
            issues.append({"issue": "unmatched_csv_stem", "stem": stem, "sample_ids": "", "filepaths": ""})
            continue
        if len(candidates) > 1 and not allow_ambiguous:
            issues.append(
                {
                    "issue": "ambiguous_dataset_stem",
                    "stem": stem,
                    "sample_ids": ",".join(item[0] for item in candidates),
                    "filepaths": " | ".join(item[2] for item in candidates),
                }
            )
            continue
        matched_stems.add(stem)
        matched_samples += len(candidates)
        for sample_id, existing_tags, _ in candidates:
            if set(tags).issubset(existing_tags):
                unchanged += 1
            else:
                updates.add(sample_id)
    return sorted(updates), issues, len(matched_stems), matched_samples, unchanged


def write_issues(dataset_name: str, issues: list[dict[str, str]]) -> Path:
    safe_name = re.sub(r"[^\w.-]", "_", dataset_name)
    output = Path.cwd() / "tmp" / f"update_tags_from_csv_{safe_name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["issue", "stem", "sample_ids", "filepaths"]
        )
        writer.writeheader()
        writer.writerows(issues)
    return output


def run(args: argparse.Namespace, fo) -> int:
    if not fo.dataset_exists(args.dataset_name):
        raise ValueError(f"Dataset does not exist: {args.dataset_name}")

    csv_stems, csv_rows = read_csv_stems(args.csv_path, args.column)
    dataset = fo.load_dataset(args.dataset_name)
    ids, filepaths, existing_tags = dataset.values(["id", "filepath", "tags"])
    updates, issues, matched_stems, matched_samples, unchanged = build_plan(
        zip(ids, filepaths, existing_tags), csv_stems, args.tags, args.allow_ambiguous
    )
    report = write_issues(args.dataset_name, issues)

    print(f"mode={'dry-run' if args.dry_run else 'update'}")
    print(f"dataset_name={args.dataset_name}")
    print(f"csv_rows={csv_rows}")
    print(f"unique_csv_stems={len(csv_stems)}")
    print(f"matched_stems={matched_stems}")
    print(f"matched_samples={matched_samples}")
    print(f"to_update={len(updates)}")
    print(f"unchanged={unchanged}")
    print(f"issues={len(issues)}")
    print(f"issues_csv={report}")

    if not args.dry_run:
        for sample in dataset.select(updates).iter_samples(autosave=True):
            sample.tags = list(dict.fromkeys([*(sample.tags or []), *args.tags]))
        print(f"updated={len(updates)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    try:
        import fiftyone as fo

        return run(args, fo)
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except Exception as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
