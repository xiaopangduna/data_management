"""Build a YOLO dataset directory from the exported master training CSV.

Creates ``images/train`` symlinks to original files and writes ``labels/train``
txt files plus ``data.yaml``. Reads only the master CSV, not part shards.
Does not connect to FiftyOne.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

LABEL_SEPARATOR = ";"


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
    """Parse CLI arguments for CSV-to-YOLO conversion."""
    parser = argparse.ArgumentParser(
        description="Create a YOLO dataset with image symlinks from the master export CSV."
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="Master export CSV (tmp/export_<dataset>.csv), not a part shard.",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--class-names",
        required=True,
        type=class_names,
        help="Comma-separated names written to data.yaml; must match export.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count work only; do not create directories, links, or files.",
    )
    return parser.parse_args(argv)


def read_dicts(path: Path) -> list[dict[str, str]]:
    """Load a CSV as a list of row dicts."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_label_lines(cell: str, names: list[str]) -> list[str] | None:
    """Split the labels column into YOLO txt lines, or None if invalid."""
    lines: list[str] = []
    for part in cell.split(LABEL_SEPARATOR):
        text = part.strip()
        if not text:
            continue
        bits = text.split()
        if len(bits) != 5:
            return None
        try:
            class_id = int(bits[0])
            cx, cy, width, height = (float(item) for item in bits[1:])
        except ValueError:
            return None
        if not 0 <= class_id < len(names):
            return None
        if width <= 0 or height <= 0:
            return None
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
    return lines or None


def write_data_yaml(path: Path, dataset_root: Path, names: list[str]) -> None:
    """Write Ultralytics data.yaml. val points at train until a split exists."""
    lines = [
        f"path: {dataset_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/train",
        "names:",
    ]
    for index, name in enumerate(names):
        lines.append(f"  {index}: {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_symlink(destination: Path, source: Path) -> str | None:
    """Create or reuse a symlink. Return an issue string on conflict."""
    source = source.resolve()
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source:
            return None
        return "dest_exists"
    destination.symlink_to(source)
    return None


def print_report(values: dict[str, object]) -> None:
    """Print key=value lines in insertion order."""
    for key, value in values.items():
        print(f"{key}={value}")


def run_build(
    rows: list[dict[str, str]],
    out_dir: Path,
    names: list[str],
    dry_run: bool,
) -> dict[str, int]:
    """Create symlinks and label files. Returns counters."""
    counts = {
        "images": 0,
        "labels": 0,
        "missing_file": 0,
        "no_boxes": 0,
        "invalid_box": 0,
        "dest_exists": 0,
        "relpath_collision": 0,
    }
    seen: set[str] = set()
    image_root = out_dir / "images" / "train"
    label_root = out_dir / "labels" / "train"
    if not dry_run:
        image_root.mkdir(parents=True, exist_ok=True)
        label_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        relpath = row.get("relpath", "").strip()
        filepath = row.get("filepath", "").strip()
        if not relpath or relpath in seen:
            counts["relpath_collision"] += 1
            continue
        seen.add(relpath)
        source = Path(filepath)
        if not source.is_file():
            counts["missing_file"] += 1
            continue
        lines = parse_label_lines(row.get("labels", ""), names)
        if lines is None:
            counts["invalid_box"] += 1
            continue
        if not lines:
            counts["no_boxes"] += 1
            continue
        destination = image_root / relpath
        label_path = label_root / Path(relpath).with_suffix(".txt")
        if dry_run:
            counts["images"] += 1
            counts["labels"] += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        conflict = ensure_symlink(destination, source)
        if conflict:
            counts["dest_exists"] += 1
            continue
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts["images"] += 1
        counts["labels"] += 1
    if not dry_run:
        write_data_yaml(out_dir / "data.yaml", out_dir, names)
    return counts


def main(argv: list[str] | None = None) -> int:
    """Build a YOLO directory from the master export CSV."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)7s] %(name)s:%(lineno)d - %(message)s",
    )
    args = parse_args(argv)
    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        logger.error("Export CSV not found: %s", csv_path)
        return 1
    out_dir = args.out_dir.expanduser()
    rows = read_dicts(csv_path)
    counts = run_build(rows, out_dir, args.class_names, args.dry_run)
    print_report(
        {
            "mode": "dry-run" if args.dry_run else "build",
            "csv": str(csv_path),
            "out_dir": str(out_dir.resolve()),
            "class_names": ",".join(args.class_names),
            "csv_images": len(rows),
            **counts,
            "yolo_done": "true",
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
