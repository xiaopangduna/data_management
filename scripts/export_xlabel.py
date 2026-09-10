"""Export a relabel folder of images and X-AnyLabeling JSON.

Writes ``<out-dir>/<relpath>`` (copy by default, optionally a symlink) and a sidecar
``<out-dir>/<relpath-with-.json>``. Does not change FiftyOne.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
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
from PIL import Image

logger = logging.getLogger(__name__)

DETECT_FIELD = "ground_truth"
BOX_DECIMALS = 6
REPORT_SUBDIR = "tmp"
XLABEL_VERSION = "2.5.0"
SAMPLE_ID_PREFIX = "fo_sample_id="
MANIFEST_NAME = "manifest.csv"
MANIFEST_COLUMNS = (
    "sample_id",
    "filepath",
    "relpath",
    "json_relpath",
    "box_count",
)
ISSUE_COLUMNS = (
    "issue",
    "sample_id",
    "relpath",
    "detail",
)


@dataclass
class ExportItem:
    """One sample ready to write into the task directory."""

    sample_id: str
    filepath: Path
    relpath: str
    json_relpath: str
    width: int
    height: int
    shapes: list[dict]


@dataclass
class ExportPlan:
    """Counts and files for one export run."""

    view_samples: int = 0
    to_write: list[ExportItem] = field(default_factory=list)
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
    """Parse CLI arguments for X-AnyLabeling export."""
    parser = argparse.ArgumentParser(
        description="Export images and X-AnyLabeling JSON into one folder."
    )
    parser.add_argument("--dataset-name", required=True, type=nonempty)
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="New task directory (images + sidecar JSON). Not the original images folder.",
    )
    parser.add_argument(
        "--export-media",
        choices=("copy", "symlink"),
        default="copy",
        help="Copy images (default), or create symlinks to the originals.",
    )
    parser.add_argument(
        "--sample-tags", required=True, type=class_names,
        help="Require ALL sample tags (comma-separated); intersect with --labels.",
    )
    parser.add_argument(
        "--label-field", default=DETECT_FIELD, type=nonempty,
        help="Detections field used for both filtering and export (default: ground_truth).",
    )
    parser.add_argument(
        "--labels", default=None, type=class_names,
        help="Require ALL labels in --label-field (comma-separated); does not remove boxes.",
    )
    parser.add_argument(
        "--export-labels", default=None, type=class_names,
        help="Export only these box labels after sample selection; default: all labels.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and write the issue CSV only; do not create the task directory.",
    )
    return parser.parse_args(argv)


def normalize_text(value: object) -> str:
    """Return a stripped string, or empty if missing."""
    if value is None:
        return ""
    return str(value).strip()


def issue_row(issue: str, sample_id: str = "", relpath: str = "", detail: str = "") -> dict[str, str]:
    """Build one issue CSV row."""
    return {"issue": issue, "sample_id": sample_id, "relpath": relpath, "detail": detail}


def json_relpath_for(relpath: str) -> str:
    """Return the sidecar JSON path for an image relpath."""
    return Path(relpath).with_suffix(".json").as_posix()


def fo_box_to_points(
    bounding_box: list[float], width: int, height: int
) -> list[list[float]] | None:
    """Convert FiftyOne top-left xywh to XLABEL rectangle corners in pixels."""
    left, top, box_w, box_h = (float(item) for item in bounding_box)
    if box_w <= 0 or box_h <= 0 or width <= 0 or height <= 0:
        return None
    x1 = round(left * width, BOX_DECIMALS)
    y1 = round(top * height, BOX_DECIMALS)
    x2 = round((left + box_w) * width, BOX_DECIMALS)
    y2 = round((top + box_h) * height, BOX_DECIMALS)
    if x2 <= x1 or y2 <= y1:
        return None
    return [[x1, y1], [x2, y2]]


def detections_to_shapes(
    detections: object,
    width: int,
    height: int,
    allowed: set[str] | None,
) -> tuple[list[dict], list[str]]:
    """Build XLABEL rectangle shapes. Returns shapes and unknown class names."""
    items = getattr(detections, "detections", None) or []
    shapes: list[dict] = []
    unknown: list[str] = []
    for detection in items:
        label = str(detection.label)
        if allowed is not None and label not in allowed:
            unknown.append(label)
            continue
        points = fo_box_to_points(list(detection.bounding_box), width, height)
        if points is None:
            continue
        shapes.append(
            {
                "label": label,
                "score": None,
                "points": points,
                "group_id": None,
                "description": None,
                "difficult": False,
                "shape_type": "rectangle",
                "flags": None,
                "attributes": {},
            }
        )
    return shapes, unknown


def read_image_size(filepath: Path, metadata: object) -> tuple[int, int] | None:
    """Return width, height from FiftyOne metadata or the image file."""
    if metadata is not None:
        width = getattr(metadata, "width", None)
        height = getattr(metadata, "height", None)
        if width and height:
            return int(width), int(height)
    try:
        with Image.open(filepath) as image:
            width, height = image.size
    except OSError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def filtered_view(
    dataset: fo.Dataset,
    sample_tags: list[str],
    label_field: str = DETECT_FIELD,
    labels: list[str] | None = None,
) -> fo.DatasetView:
    """Intersect all sample tags and all labels without filtering any boxes."""
    if not dataset.has_field(label_field):
        raise ValueError(f"Label field does not exist: {label_field}")
    field = dataset.get_field(label_field)
    if not isinstance(field, fo.EmbeddedDocumentField) or field.document_type is not fo.Detections:
        raise ValueError(f"Label field must contain Detections: {label_field}")
    view = dataset.match_tags(sample_tags, bool=True, all=True)
    if labels:
        view = view.match({f"{label_field}.detections.label": {"$all": labels}})
    return view


def build_xlabel_document(item: ExportItem) -> dict:
    """Return one XLABEL JSON object, including fo_sample_id for round-trip."""
    image_name = Path(item.relpath).name
    return {
        "version": XLABEL_VERSION,
        "flags": {},
        "shapes": item.shapes,
        "description": f"{SAMPLE_ID_PREFIX}{item.sample_id}",
        "checked": False,
        "sample_id": item.sample_id,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": item.height,
        "imageWidth": item.width,
    }


def collect_export_plan(
    view: fo.DatasetView,
    allowed: set[str] | None,
    label_field: str = DETECT_FIELD,
) -> ExportPlan:
    """Collect image/JSON work and issue rows."""
    plan = ExportPlan(view_samples=len(view))
    ids = [str(sample_id) for sample_id in view.values("id")]
    filepaths = view.values("filepath")
    if view.has_field("relpath"):
        relpaths = view.values("relpath")
    else:
        relpaths = [None] * len(ids)
    if view.has_field(label_field):
        detections_list = view.values(label_field)
    else:
        detections_list = [None] * len(ids)
    if view.has_field("metadata"):
        metadata_list = view.values("metadata")
    else:
        metadata_list = [None] * len(ids)
    seen_relpath: dict[str, str] = {}
    for index, sample_id in enumerate(ids):
        relpath = normalize_text(relpaths[index])
        if not relpath:
            relpath = Path(str(filepaths[index])).name
        if relpath in seen_relpath:
            plan.issues.append(
                issue_row(
                    "relpath_collision",
                    sample_id,
                    relpath,
                    f"already used by {seen_relpath[relpath]}",
                )
            )
            continue
        source = Path(str(filepaths[index]))
        if not source.is_file():
            plan.issues.append(issue_row("missing_file", sample_id, relpath, str(source)))
            continue
        size = read_image_size(source, metadata_list[index])
        if size is None:
            plan.issues.append(issue_row("missing_size", sample_id, relpath, str(source)))
            continue
        width, height = size
        shapes, unknown = detections_to_shapes(
            detections_list[index], width, height, allowed
        )
        for label in unknown:
            plan.issues.append(issue_row("unknown_class", sample_id, relpath, label))
        if detections_list[index] is not None and not shapes and unknown:
            continue
        seen_relpath[relpath] = sample_id
        plan.to_write.append(
            ExportItem(
                sample_id=sample_id,
                filepath=source,
                relpath=relpath,
                json_relpath=json_relpath_for(relpath),
                width=width,
                height=height,
                shapes=shapes,
            )
        )
    return plan


def ensure_symlink(destination: Path, source: Path) -> str | None:
    """Create or reuse a symlink. Return an issue string on conflict."""
    source = source.resolve()
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source:
            return None
        return "dest_exists"
    destination.symlink_to(source)
    return None


def write_json(path: Path, document: dict) -> None:
    """Write pretty-printed XLABEL JSON."""
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    """Write a CSV with a stable header."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def issue_csv_path(dataset_name: str) -> Path:
    """Return ``tmp/export_xlabel_<dataset>.csv``."""
    directory = Path.cwd() / REPORT_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"export_xlabel_{dataset_name}.csv"


def apply_export(
    out_dir: Path, items: list[ExportItem], export_media: str = "copy"
) -> tuple[int, int, list[dict[str, str]]]:
    """Write images and JSON. Returns written images, json files, extra issues."""
    if export_media not in {"copy", "symlink"}:
        raise ValueError(f"Unsupported export media mode: {export_media}")
    images = 0
    json_files = 0
    issues: list[dict[str, str]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        destination = out_dir / item.relpath
        json_path = out_dir / item.json_relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        if export_media == "symlink":
            conflict = ensure_symlink(destination, item.filepath)
        elif destination.exists() or destination.is_symlink():
            conflict = "dest_exists"
        else:
            shutil.copy2(item.filepath, destination)
            conflict = None
        if conflict:
            issues.append(
                issue_row("dest_exists", item.sample_id, item.relpath, str(destination))
            )
            continue
        write_json(json_path, build_xlabel_document(item))
        images += 1
        json_files += 1
    manifest_rows = [
        {
            "sample_id": item.sample_id,
            "filepath": str(item.filepath.resolve()),
            "relpath": item.relpath,
            "json_relpath": item.json_relpath,
            "box_count": str(len(item.shapes)),
        }
        for item in items
        if (out_dir / item.json_relpath).is_file()
    ]
    write_csv(out_dir / MANIFEST_NAME, MANIFEST_COLUMNS, manifest_rows)
    return images, json_files, issues


def print_report(values: dict[str, object]) -> None:
    """Print key=value lines in insertion order."""
    for key, value in values.items():
        print(f"{key}={value}")


def main(argv: list[str] | None = None) -> int:
    """Export a relabel folder from FiftyOne."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)7s] %(name)s:%(lineno)d - %(message)s",
    )
    args = parse_args(argv)
    if not fo.dataset_exists(args.dataset_name):
        logger.error("Dataset does not exist: %s", args.dataset_name)
        return 1
    dataset = fo.load_dataset(args.dataset_name)
    try:
        view = filtered_view(dataset, args.sample_tags, args.label_field, args.labels)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    allowed = set(args.export_labels) if args.export_labels else None
    plan = collect_export_plan(view, allowed, args.label_field)
    csv_path = issue_csv_path(dataset.name)
    write_csv(csv_path, ISSUE_COLUMNS, plan.issues)
    out_dir = args.out_dir.expanduser()
    print_report(
        {
            "mode": "dry-run" if args.dry_run else "export",
            "dataset_name": dataset.name,
            "out_dir": str(out_dir.resolve()),
            "export_media": args.export_media,
            "sample_tags": ",".join(args.sample_tags),
            "label_field": args.label_field,
            "labels": ",".join(args.labels) if args.labels else "none",
            "export_labels": ",".join(args.export_labels) if args.export_labels else "all",
            "view_samples": plan.view_samples,
            "to_write": len(plan.to_write),
            "issues": len(plan.issues),
            "csv_path": str(csv_path),
        }
    )
    if args.dry_run:
        return 0
    images, json_files, extra = apply_export(out_dir, plan.to_write, args.export_media)
    if extra:
        plan.issues.extend(extra)
        write_csv(csv_path, ISSUE_COLUMNS, plan.issues)
    print_report(
        {
            "images": images,
            "json_files": json_files,
            "issues": len(plan.issues),
            "export_done": "true",
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
