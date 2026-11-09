"""Export paired face keypoints as a COCO JSON for CocoWholeBodyFaceDataset.

One run writes one split. Images go in ``<out-dir>/images`` and the annotation
file is ``<out-dir>/instance_<split>.json``. ``--keypoints-field`` is ``keypoints``
(42 points) or ``keypoints_68`` (68 points). Point order and edges come from
that field's skeleton. ``bbox`` and ``face_box`` are both the paired box in
absolute xywh pixels. Visibility is written as stored: 0 unlabeled, 1 occluded,
2 visible. An unlabeled point is written as ``0 0 0``. Samples with no pair,
a broken pair, or no image size are skipped and listed in
``tmp/export_coco_keypoints_<dataset>.csv``. Duplicate image file names stop
the run. Images with no face are not exported.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


def _load_shapes():
    path = Path(__file__).with_name("cvat_keypoints.py")
    spec = importlib.util.spec_from_file_location("cvat_keypoints", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shapes = _load_shapes()

REPORT_SUBDIR = "tmp"
COORD_DECIMALS = 6
ISSUE_COLUMNS = ("issue", "sample_id", "filepath", "detail")
CATEGORY_ID = 1


def nonempty(value: str) -> str:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def names(value: str) -> list[str]:
    items = [part.strip() for part in value.split(",")]
    if not all(items):
        raise argparse.ArgumentTypeError("provide comma-separated nonempty names")
    if len(set(items)) != len(items):
        raise argparse.ArgumentTypeError("duplicate names are not allowed")
    return items


def split_name(value: str) -> str:
    value = nonempty(value)
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise argparse.ArgumentTypeError("must be a single folder name")
    return value


def keypoints_field_name(value: str) -> str:
    value = nonempty(value)
    try:
        shapes.point_count_for_field(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=nonempty)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split", default="train", type=split_name,
                        help="JSON name instance_<split>.json. Default: train.")
    parser.add_argument("--sample-tags", required=True, type=names,
                        help="Comma-separated sample tags; match ANY tag (union).")
    parser.add_argument("--keypoints-field", required=True, type=keypoints_field_name)
    parser.add_argument("--label-field", default="ground_truth", type=nonempty)
    parser.add_argument("--label", default=shapes.FACE_LABEL, type=nonempty)
    parser.add_argument("--exclude-tags", type=names, default=["dup_repeat_drop", "dup_near_drop"],
                        help="Exclude samples with ANY of these tags. Default: dup_repeat_drop,dup_near_drop.")
    parser.add_argument("--export-media", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and print the plan without writing files.")
    return parser.parse_args(argv)


@dataclass
class PlannedFace:
    """One face in absolute pixels, before COCO ids are assigned."""

    bbox: list[float]
    keypoints: list[float]
    num_keypoints: int


@dataclass
class PlannedImage:
    """One image that has at least one paired face."""

    sample_id: str
    source: Path
    filename: str
    width: int
    height: int
    faces: list[PlannedFace] = field(default_factory=list)


def annotation_path(output: Path, split: str) -> Path:
    return output / f"instance_{split}.json"


def check_output(path: Path, split: str) -> None:
    """Require an empty image directory and no existing annotation file."""
    if path.is_symlink():
        raise ValueError(f"Output directory must not be a symlink: {path}")
    if path.exists() and not path.is_dir():
        raise ValueError(f"Output path must be a directory: {path}")
    images = path / "images"
    if images.is_symlink() or (images.exists() and not images.is_dir()):
        raise ValueError(f"Image directory must be a real directory: {images}")
    if images.exists() and any(images.iterdir()):
        raise ValueError(f"Image directory must be empty or absent: {images}")
    annotation = annotation_path(path, split)
    if annotation.exists():
        raise ValueError(f"Annotation file already exists: {annotation}")


def issue_row(issue: str, sample_id: str = "", filepath: str = "", detail: str = "") -> dict[str, str]:
    return {"issue": issue, "sample_id": sample_id, "filepath": filepath, "detail": detail}


def issue_csv_path(dataset_name: str, report_dir: Path) -> Path:
    """Return ``tmp/export_coco_keypoints_<dataset>.csv``."""
    safe = "".join(
        char if char.isalnum() or char in "._-" else "_" for char in dataset_name
    ).strip("._")
    return report_dir / f"export_coco_keypoints_{safe or 'dataset'}.csv"


def write_issues(path: Path, issues: list[dict[str, str]]) -> None:
    """Write skip rows, or remove a stale report when this run has none."""
    if not issues:
        if path.is_file():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ISSUE_COLUMNS))
        writer.writeheader()
        writer.writerows(issues)


def image_size(metadata: object) -> tuple[int, int] | None:
    """Return ``(width, height)`` when both metadata values are present."""
    if metadata is None:
        return None
    width = getattr(metadata, "width", None)
    height = getattr(metadata, "height", None)
    if isinstance(width, bool) or isinstance(height, bool):
        return None
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        return None
    if width <= 0 or height <= 0:
        return None
    return int(width), int(height)


def visibility_flags(keypoint: object) -> list | None:
    """Return the stored visibility list, or None when the sample has none."""
    if not hasattr(keypoint, "has_attribute") or not keypoint.has_attribute("visibility"):
        return None
    value = keypoint.get_attribute_value("visibility")
    if not isinstance(value, (list, tuple)):
        return None
    return list(value)


def as_flag(value: object) -> int | None:
    """Return 0, 1, or 2. Other values, including bools, are rejected."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        flag = value
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number) or math.isinf(number) or not number.is_integer():
            return None
        flag = int(number)
    if flag not in (0, 1, 2):
        return None
    return flag


def point_xy(point: object) -> tuple[float, float] | None:
    try:
        x = float(point[0])
        y = float(point[1])
    except (TypeError, ValueError, IndexError):
        return None
    if math.isinf(x) or math.isinf(y):
        return None
    return x, y


def coco_triple(point: tuple[float, float], flag: int, width: int, height: int) -> tuple[float, float, int]:
    """Return absolute ``(x, y, v)``. Unlabeled and non-finite points become ``(0, 0, 0)``."""
    x, y = point
    if flag == 0 or math.isnan(x) or math.isnan(y):
        return 0.0, 0.0, 0
    return (
        round(x * width, COORD_DECIMALS),
        round(y * height, COORD_DECIMALS),
        flag,
    )


def pixel_box(bbox: object, width: int, height: int) -> list[float] | None:
    """Return absolute xywh, or None when the box is missing or empty."""
    try:
        values = [float(item) for item in list(bbox)]
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or any(math.isnan(item) or math.isinf(item) for item in values):
        return None
    left, top, box_w, box_h = values
    if box_w <= 0 or box_h <= 0:
        return None
    return [
        round(left * width, COORD_DECIMALS),
        round(top * height, COORD_DECIMALS),
        round(box_w * width, COORD_DECIMALS),
        round(box_h * height, COORD_DECIMALS),
    ]


def category_from_skeleton(skeleton: object, point_count: int) -> dict:
    """Return the single COCO category stored on the keypoint field."""
    labels = [str(name) for name in list(getattr(skeleton, "labels", None) or [])]
    if len(labels) != point_count or not all(labels):
        found = len(labels)
        raise ValueError(f"Skeleton has {found} point names, expected {point_count}")
    edges = []
    for edge in list(getattr(skeleton, "edges", None) or []):
        try:
            indexes = [int(item) for item in list(edge)]
        except (TypeError, ValueError):
            raise ValueError(f"Skeleton edge is invalid: {edge}") from None
        if len(indexes) < 2 or any(item < 0 or item >= point_count for item in indexes):
            raise ValueError(f"Skeleton edge is invalid: {edge}")
        edges.append(indexes)
    return {
        "id": CATEGORY_ID,
        "name": "face",
        "supercategory": "face",
        "keypoints": labels,
        "skeleton": edges,
    }


def flags_for_points(points: list[tuple[float, float]], visibility: list | None) -> list[int] | None:
    """Return one flag per point. A missing list treats finite points as visible."""
    if visibility is None:
        return [
            0 if math.isnan(x) or math.isnan(y) else 2
            for x, y in points
        ]
    if len(visibility) != len(points):
        return None
    flags = []
    for value in visibility:
        flag = as_flag(value)
        if flag is None:
            return None
        flags.append(flag)
    return flags


def face_from_pair(detection: object, keypoint: object, width: int, height: int, point_count: int) -> tuple[PlannedFace | None, str]:
    """Convert one paired face. The reason is empty when the face is usable."""
    points = list(getattr(keypoint, "points", None) or [])
    parsed: list[tuple[float, float]] = []
    for point in points:
        xy = point_xy(point)
        if xy is None:
            return None, "bad_keypoints"
        parsed.append(xy)
    if len(parsed) != point_count:
        return None, "bad_keypoints"
    flags = flags_for_points(parsed, visibility_flags(keypoint))
    if flags is None:
        return None, "bad_visibility"
    flat: list[float] = []
    visible = 0
    for point, flag in zip(parsed, flags, strict=True):
        x, y, written = coco_triple(point, flag, width, height)
        flat.extend((x, y, written))
        visible += written > 0
    box = pixel_box(getattr(detection, "bounding_box", None), width, height)
    if box is None:
        return None, "bad_box"
    return PlannedFace(bbox=box, keypoints=flat, num_keypoints=visible), ""


def plan_sample(
    sample: object,
    label_field: str,
    keypoints_field: str,
    label: str,
    point_count: int,
) -> tuple[PlannedImage | None, dict[str, str] | None]:
    """Return one exportable image, or one skip row."""
    sample_id = str(getattr(sample, "id", ""))
    filepath = str(getattr(sample, "filepath", ""))
    source = Path(filepath)
    if not source.is_file():
        raise ValueError(f"Missing image for sample {sample_id}: {filepath}")
    size = image_size(getattr(sample, "metadata", None))
    if size is None:
        return None, issue_row("missing_size", sample_id, filepath)
    width, height = size
    detections = sample[label_field] if hasattr(sample, "__getitem__") else None
    keypoints = sample[keypoints_field] if hasattr(sample, "__getitem__") else None
    try:
        pairs = shapes.face_pairs(detections, keypoints, label)
    except ValueError as exc:
        return None, issue_row("unpaired", sample_id, filepath, str(exc))
    if not pairs:
        return None, issue_row("no_face", sample_id, filepath)
    faces: list[PlannedFace] = []
    for detection, keypoint in pairs:
        face, reason = face_from_pair(detection, keypoint, width, height, point_count)
        if face is None:
            return None, issue_row(reason, sample_id, filepath)
        faces.append(face)
    filename = source.name
    if not filename or filename in {".", ".."}:
        return None, issue_row("bad_filename", sample_id, filepath, filename)
    return PlannedImage(sample_id, source.resolve(), filename, width, height, faces), None


def duplicate_rows(planned: list[PlannedImage]) -> list[dict[str, str]]:
    """Return one row per sample whose image file name is shared."""
    by_name: dict[str, list[PlannedImage]] = defaultdict(list)
    for item in planned:
        by_name[item.filename].append(item)
    rows = []
    for filename, items in by_name.items():
        if len(items) < 2:
            continue
        detail = f"{filename} x{len(items)}"
        for item in items:
            rows.append(issue_row("duplicate_name", item.sample_id, str(item.source), detail))
    return rows


def coco_document(planned: list[PlannedImage], category: dict) -> dict:
    """Assign image and annotation ids and return the COCO object."""
    images = []
    annotations = []
    next_ann = 1
    for image_id, item in enumerate(planned, start=1):
        images.append({
            "id": image_id,
            "file_name": item.filename,
            "width": item.width,
            "height": item.height,
        })
        for face in item.faces:
            box_w, box_h = face.bbox[2], face.bbox[3]
            annotations.append({
                "id": next_ann,
                "image_id": image_id,
                "category_id": CATEGORY_ID,
                "bbox": face.bbox,
                "face_box": list(face.bbox),
                "keypoints": face.keypoints,
                "num_keypoints": face.num_keypoints,
                "iscrowd": 0,
                "area": round(box_w * box_h, COORD_DECIMALS),
                "segmentation": [],
            })
            next_ann += 1
    return {
        "info": {"description": "face keypoints"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [category],
    }


def write_media(planned: list[PlannedImage], images_dir: Path, export_media: str) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    for item in planned:
        destination = images_dir / item.filename
        if export_media == "copy":
            shutil.copy2(item.source, destination)
        else:
            destination.symlink_to(item.source)


def export_dataset(args: argparse.Namespace, report_dir: Path | None = None) -> dict:
    import fiftyone as fo

    output = args.out_dir.expanduser().absolute()
    check_output(output, args.split)
    dataset = fo.load_dataset(args.dataset)
    if dataset.media_type != "image":
        raise ValueError("Only image datasets are supported")
    point_count = shapes.point_count_for_field(args.keypoints_field)
    for field_name, document_type in (
        (args.label_field, fo.Detections),
        (args.keypoints_field, fo.Keypoints),
    ):
        field = dataset.get_field_schema().get(field_name)
        if not isinstance(field, fo.EmbeddedDocumentField) or field.document_type != document_type:
            raise ValueError(f"Label field has the wrong type: {field_name}")
    skeleton = (dataset.skeletons or {}).get(args.keypoints_field)
    if skeleton is None:
        raise ValueError(f"Field '{args.keypoints_field}' has no skeleton")
    category = category_from_skeleton(skeleton, point_count)
    missing_tags = sorted(set(args.sample_tags) - set(dataset.distinct("tags")))
    if missing_tags:
        raise ValueError(f"Unknown sample tags: {', '.join(missing_tags)}")
    view = dataset.match_tags(args.sample_tags, all=False)
    if args.exclude_tags:
        view = view.exclude(dataset.match_tags(args.exclude_tags, all=False))
    if not len(view):
        raise ValueError("No samples match the requested tags")

    planned: list[PlannedImage] = []
    issues: list[dict[str, str]] = []
    for sample in view.iter_samples(progress=False, autosave=False):
        item, issue = plan_sample(
            sample, args.label_field, args.keypoints_field, args.label, point_count,
        )
        if issue is not None:
            issues.append(issue)
        elif item is not None:
            planned.append(item)
    duplicates = duplicate_rows(planned)
    report_dir = report_dir or (Path.cwd() / REPORT_SUBDIR)
    csv_path = issue_csv_path(dataset.name or args.dataset, report_dir)
    if duplicates:
        write_issues(csv_path, [*issues, *duplicates])
        names = sorted({row["detail"] for row in duplicates})
        raise ValueError(f"Duplicate image names: {', '.join(names)}. See {csv_path}")
    if not planned:
        write_issues(csv_path, issues)
        raise ValueError(f"No faces to export. See {csv_path}")
    write_issues(csv_path, issues)

    faces = sum(len(item.faces) for item in planned)
    summary = {
        "dataset": args.dataset,
        "label_field": args.label_field,
        "keypoints_field": args.keypoints_field,
        "point_count": point_count,
        "label": args.label,
        "tags": args.sample_tags,
        "exclude_tags": args.exclude_tags,
        "tag_matching": "union",
        "split": args.split,
        "output_dir": str(output),
        "annotation": str(annotation_path(output, args.split)),
        "export_media": args.export_media,
        "view_samples": len(view),
        "images": len(planned),
        "faces": faces,
        "skipped": len(issues),
        "skip_counts": dict(Counter(row["issue"] for row in issues)),
    }
    if issues:
        summary["csv_path"] = str(csv_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry run: no files written.")
        return summary

    check_output(output, args.split)
    document = coco_document(planned, category)
    write_media(planned, output / "images", args.export_media)
    annotation_path(output, args.split).write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Export complete: {output}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        export_dataset(args)
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
