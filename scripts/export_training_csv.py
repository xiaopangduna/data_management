"""Export a training manifest from an existing FiftyOne dataset.

Writes ``tmp/export_<dataset>.csv`` (one row per image, labels in one column)
and readable shards ``tmp/export_<dataset>_part_XX.csv``. Selects samples that
have every ``--include-tags`` tag (intersection), then drops ``dup_near``
unless ``--exclude-tags none``. Includes images with no boxes (negatives).
Labels keep class names; ``csv_to_yolo.py`` assigns YOLO class ids. Does not
copy images or write YOLO txt files.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"glob2(\.|$)",
)

import fiftyone as fo

logger = logging.getLogger(__name__)

DETECT_FIELD = "ground_truth_detect"
BOX_DECIMALS = 6
REPORT_SUBDIR = "tmp"
DEFAULT_EXCLUDE_TAGS = "dup_near"
IMAGES_PER_PART = 5000
LABEL_SEPARATOR = ";"
EXPORT_COLUMNS = (
    "sample_id",
    "filepath",
    "relpath",
    "tags",
    "box_count",
    "labels",
)
ISSUE_COLUMNS = (
    "issue",
    "sample_id",
    "relpath",
    "detail",
)


def nonempty(value: str) -> str:
    """Reject blank CLI strings."""
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("must not be empty")
    return stripped


def include_tags(value: str) -> list[str]:
    """Parse required include tags; at least one name. Matching is intersection."""
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("provide at least one tag")
    return names


def tag_list(value: str) -> list[str]:
    """Parse exclude tags; ``none`` or empty means exclude nothing."""
    stripped = value.strip()
    if stripped.lower() in {"", "none"}:
        return []
    return [part.strip() for part in stripped.split(",") if part.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for training CSV export."""
    parser = argparse.ArgumentParser(
        description="Export a training CSV for samples that have every include tag."
    )
    parser.add_argument("--dataset-name", required=True, type=nonempty)
    parser.add_argument(
        "--include-tags",
        required=True,
        type=include_tags,
        help="Export samples that have all of these tags (comma-separated, intersection).",
    )
    parser.add_argument(
        "--exclude-tags",
        default=DEFAULT_EXCLUDE_TAGS,
        type=tag_list,
        help="Samples with any of these tags are omitted. Default: dup_near. Use none to keep them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write CSVs from the filtered view; does not change the dataset.",
    )
    return parser.parse_args(argv)


def normalize_text(value: object) -> str:
    """Return a stripped string, or empty if missing."""
    if value is None:
        return ""
    return str(value).strip()


def report_dir() -> Path:
    """Return ``<cwd>/tmp``, creating it if needed."""
    path = Path.cwd() / REPORT_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def filtered_view(
    dataset: fo.Dataset,
    include_tags: list[str],
    exclude_tags: list[str],
) -> fo.DatasetView:
    """Samples that have every include tag, minus any excluded tag."""
    view = dataset.match_tags(include_tags, bool=True, all=True)
    for tag in exclude_tags:
        view = view.match_tags(tag, bool=False)
    return view


def to_label_line(class_name: str, bounding_box: list[float]) -> str | None:
    """Convert FiftyOne top-left xywh to ``name cx cy w h`` (center-normalized)."""
    left, top, width, height = (float(item) for item in bounding_box)
    if width <= 0 or height <= 0:
        return None
    cx = round(left + width / 2.0, BOX_DECIMALS)
    cy = round(top + height / 2.0, BOX_DECIMALS)
    return (
        f"{class_name} {cx:.{BOX_DECIMALS}f} {cy:.{BOX_DECIMALS}f} "
        f"{round(width, BOX_DECIMALS):.{BOX_DECIMALS}f} {round(height, BOX_DECIMALS):.{BOX_DECIMALS}f}"
    )


def collect_export_rows(
    view: fo.DatasetView,
) -> tuple[list[dict[str, str]], list[dict[str, str]], int]:
    """Build one row per image and any issue rows. Empty labels are negatives."""
    ids = [str(sample_id) for sample_id in view.values("id")]
    filepaths = view.values("filepath")
    if view.has_field("relpath"):
        relpaths = view.values("relpath")
    else:
        relpaths = [None] * len(ids)
    tags_list = view.values("tags") if view.has_field("tags") else [None] * len(ids)
    if view.has_field(DETECT_FIELD):
        detections_list = view.values(DETECT_FIELD)
    else:
        detections_list = [None] * len(ids)
    rows: list[dict[str, str]] = []
    issues: list[dict[str, str]] = []
    seen_relpath: dict[str, str] = {}
    box_count = 0
    for index, sample_id in enumerate(ids):
        relpath = normalize_text(relpaths[index])
        if not relpath:
            relpath = Path(str(filepaths[index])).name
        if relpath in seen_relpath:
            issues.append(
                {
                    "issue": "relpath_collision",
                    "sample_id": sample_id,
                    "relpath": relpath,
                    "detail": f"already used by {seen_relpath[relpath]}",
                }
            )
            continue
        detections = detections_list[index]
        det_items = getattr(detections, "detections", None) or []
        lines: list[str] = []
        for detection in det_items:
            class_name = str(detection.label).strip()
            if not class_name:
                issues.append(
                    {
                        "issue": "invalid_box",
                        "sample_id": sample_id,
                        "relpath": relpath,
                        "detail": "empty class name",
                    }
                )
                continue
            formatted = to_label_line(class_name, list(detection.bounding_box))
            if formatted is None:
                issues.append(
                    {
                        "issue": "invalid_box",
                        "sample_id": sample_id,
                        "relpath": relpath,
                        "detail": class_name,
                    }
                )
                continue
            lines.append(formatted)
        if det_items and not lines:
            issues.append(
                {
                    "issue": "no_usable_boxes",
                    "sample_id": sample_id,
                    "relpath": relpath,
                    "detail": "detections present but none had valid geometry",
                }
            )
            continue
        seen_relpath[relpath] = sample_id
        tags = tags_list[index] or []
        box_count += len(lines)
        rows.append(
            {
                "sample_id": sample_id,
                "filepath": str(filepaths[index]),
                "relpath": relpath,
                "tags": ",".join(str(tag) for tag in tags),
                "box_count": str(len(lines)),
                "labels": LABEL_SEPARATOR.join(lines),
            }
        )
    return rows, issues, box_count


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    """Write a CSV with a stable header."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def clear_previous_export_files(directory: Path, dataset_name: str) -> None:
    """Remove previous shards and legacy two-file CSVs for this dataset."""
    for path in directory.glob(f"export_{dataset_name}_part_*.csv"):
        path.unlink()
    for suffix in ("_images.csv", "_boxes.csv"):
        legacy = directory / f"export_{dataset_name}{suffix}"
        if legacy.exists():
            legacy.unlink()


def write_export_csvs(
    dataset_name: str,
    rows: list[dict[str, str]],
    issues: list[dict[str, str]],
) -> dict[str, object]:
    """Write the master CSV, optional shards, and issues only when needed."""
    directory = report_dir()
    clear_previous_export_files(directory, dataset_name)
    master_path = directory / f"export_{dataset_name}.csv"
    write_csv(master_path, EXPORT_COLUMNS, rows)
    shard_count = 0
    if rows:
        pad = max(2, len(str((len(rows) + IMAGES_PER_PART - 1) // IMAGES_PER_PART)))
        for start in range(0, len(rows), IMAGES_PER_PART):
            shard_count += 1
            shard_path = directory / f"export_{dataset_name}_part_{shard_count:0{pad}d}.csv"
            write_csv(shard_path, EXPORT_COLUMNS, rows[start : start + IMAGES_PER_PART])
    issues_path = directory / f"export_{dataset_name}_issues.csv"
    if issues:
        write_csv(issues_path, ISSUE_COLUMNS, issues)
    elif issues_path.exists():
        issues_path.unlink()
    logger.info(
        "Wrote export CSV %s rows=%s parts=%s issues=%s",
        master_path,
        len(rows),
        shard_count,
        len(issues),
    )
    return {
        "csv_path": str(master_path),
        "part_files": shard_count,
        "issues_csv": str(issues_path) if issues else "none",
    }


def print_report(values: dict[str, object]) -> None:
    """Print key=value lines in insertion order."""
    for key, value in values.items():
        print(f"{key}={value}")


def main(argv: list[str] | None = None) -> int:
    """Export a master training CSV and readable parts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)7s] %(name)s:%(lineno)d - %(message)s",
    )
    args = parse_args(argv)
    if not fo.dataset_exists(args.dataset_name):
        logger.error("Dataset does not exist: %s", args.dataset_name)
        return 1
    dataset = fo.load_dataset(args.dataset_name)
    view = filtered_view(dataset, args.include_tags, args.exclude_tags)
    rows, issues, box_count = collect_export_rows(view)
    csv_info = write_export_csvs(dataset.name, rows, issues)
    positives = sum(1 for row in rows if int(row["box_count"]) > 0)
    print_report(
        {
            "mode": "dry-run" if args.dry_run else "export",
            "dataset_name": dataset.name,
            "include_tags": ",".join(args.include_tags),
            "exclude_tags": ",".join(args.exclude_tags) if args.exclude_tags else "none",
            "view_samples": len(view),
            "images": len(rows),
            "positives": positives,
            "negatives": len(rows) - positives,
            "boxes": box_count,
            "issues": len(issues),
            **csv_info,
            "export_done": "true",
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
