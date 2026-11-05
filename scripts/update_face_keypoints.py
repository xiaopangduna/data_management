"""Attach COCO face keypoints onto samples already in a FiftyOne dataset.

Matches each image's SHA-256 to the sample ``sha256`` field. Images that
miss fall back to an exact ``phash`` match. File names are not used.
``--image-dir`` only locates the files named in the JSON. The run does not
change ``filepath`` and does not create samples. The dataset must already
have ``sha256`` from ``update_media.py``. A valid ``face_box`` is
appended to the detections field as ``--label`` (default ``kpt_face``). When
``face_box`` is absent, the COCO ``bbox`` — the visible-point extent — is
stored under the same label. A 42-point file is written to ``keypoints``.
A 68-point file is written to ``keypoints_68``. Both use the box label
``kpt_face``. When the other point field already has that face, the existing
box is left in place. The 68-to-42 mapping stays in the training code. Other
detection labels are kept. A rerun replaces only this label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

logger = logging.getLogger(__name__)

POINT_COUNT = 42
POINT_COUNT_68 = 68
ALLOWED_POINT_COUNTS = (POINT_COUNT, POINT_COUNT_68)
DEFAULT_LABEL = "kpt_face"
DEFAULT_LABEL_FIELD = "ground_truth"
DEFAULT_KEYPOINTS_FIELD = "keypoints"
KEYPOINTS_68_FIELD = "keypoints_68"
COORD_DECIMALS = 6
COORD_TOLERANCE = 1e-5
REPORT_SUBDIR = "tmp"
ISSUE_COLUMNS = ("issue", "filename", "sample_id", "filepath", "sha256", "phash", "detail")
SHA256_CHUNK_SIZE = 1024 * 1024
MISSING_SHA256_MSG = "Dataset missing sha256; run scripts/update_media.py first"


@dataclass(frozen=True)
class KeypointSchema:
    """Point names and 0-based skeleton edges from the COCO category."""

    labels: tuple[str, ...]
    edges: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class FaceAnnotation:
    """One face box plus its keypoints, in FiftyOne relative coordinates."""

    filename: str
    image_path: str
    width: int
    height: int
    box_px: tuple[float, float, float, float]
    points_px: tuple[tuple[float, float], ...]
    box: tuple[float, float, float, float]
    points: tuple[tuple[float, float], ...]
    visibility: tuple[int, ...]

    def for_size(self, width: int, height: int) -> FaceAnnotation:
        """Return a copy whose relative coordinates use this image size."""
        if (width, height) == (self.width, self.height):
            return self
        box = _relative_xywh(self.box_px, width, height)
        points = tuple(
            (float("nan"), float("nan")) if math.isnan(x) or math.isnan(y) else (
                round(x / width, COORD_DECIMALS), round(y / height, COORD_DECIMALS)
            )
            for x, y in self.points_px
        )
        return replace(self, width=width, height=height, box=box, points=points)


@dataclass
class UpdateResult:
    """Counts for one update run."""

    json_images: int = 0
    matched: int = 0
    to_update: int = 0
    unchanged: int = 0
    written: int = 0
    phash_matched: int = 0
    issues: list[dict[str, str]] = field(default_factory=list)
    csv_path: str = ""


def nonempty(value: str) -> str:
    """Reject a blank CLI string."""
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("must not be empty")
    return stripped


def tag_list(value: str) -> list[str]:
    """Parse a comma-separated tag list, preserving order and dropping duplicates."""
    tags = [part.strip() for part in value.split(",") if part.strip()]
    if not tags:
        raise argparse.ArgumentTypeError("provide at least one tag")
    return list(dict.fromkeys(tags))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for attaching one COCO keypoint JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=nonempty)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path, dest="json_path")
    parser.add_argument("--sample-tags", required=True, type=tag_list)
    parser.add_argument("--label-field", default=DEFAULT_LABEL_FIELD, type=nonempty)
    parser.add_argument("--keypoints-field", default=DEFAULT_KEYPOINTS_FIELD, type=nonempty)
    parser.add_argument("--label", default=DEFAULT_LABEL, type=nonempty)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.label_field == args.keypoints_field:
        parser.error("--label-field and --keypoints-field must be different")
    return args


def issue_row(
    kind: str,
    filename: str = "",
    sample_id: str = "",
    filepath: str = "",
    detail: str = "",
    sha256: str = "",
    phash: str = "",
) -> dict[str, str]:
    """Build one CSV issue row."""
    return {
        "issue": kind,
        "filename": filename,
        "sample_id": sample_id,
        "filepath": filepath,
        "sha256": sha256,
        "phash": phash,
        "detail": detail,
    }


def compute_phash_hex(file_path: Path) -> str:
    """Hash pixels the same way as ``update_media.compute_phash_hex``."""
    impl = compute_phash_hex.__dict__.get("impl")
    if impl is None:
        media_path = Path(__file__).with_name("update_media.py")
        spec = importlib.util.spec_from_file_location("update_media_phash", media_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {media_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        impl = module.compute_phash_hex
        compute_phash_hex.__dict__["impl"] = impl
    return impl(file_path)


def compute_sha256_hex(file_path: Path) -> str:
    """Hash file bytes the same way as ``update_media.compute_sha256_hex``."""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(SHA256_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _is_index(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _as_visibility(value: object, point_count: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    allowed = (0, 1, 2, 3, 4) if point_count == POINT_COUNT_68 else (0, 1)
    if value not in allowed:
        return None
    return int(value)


def _point_hidden(point_count: int, x: float, y: float, flag: int) -> bool:
    """Hide a missing slot. 68-point files keep a point whenever it has coordinates."""
    if point_count == POINT_COUNT_68:
        return x == 0 and y == 0
    return flag == 0


def load_schema(document: dict) -> KeypointSchema:
    """Read the single face category. Edges stay 0-based, as stored in this JSON."""
    categories = document.get("categories")
    if not isinstance(categories, list) or len(categories) != 1 or not isinstance(categories[0], dict):
        found = len(categories) if isinstance(categories, list) else 0
        raise ValueError(f"Expected one category, found {found}")
    category = categories[0]
    names = category.get("keypoints")
    if (
        not isinstance(names, list)
        or len(names) not in ALLOWED_POINT_COUNTS
        or not all(isinstance(name, str) and name for name in names)
    ):
        found = len(names) if isinstance(names, list) else 0
        raise ValueError(f"Expected 42 or 68 keypoint names, found {found}")
    point_count = len(names)
    raw_edges = category.get("skeleton")
    if not isinstance(raw_edges, list) or not raw_edges:
        raise ValueError("Category is missing skeleton edges")
    edges: list[tuple[int, ...]] = []
    for edge in raw_edges:
        if not isinstance(edge, list) or len(edge) < 2 or not all(_is_index(item) for item in edge):
            raise ValueError(f"Skeleton edge is invalid: {edge}")
        if any(item < 0 or item >= point_count for item in edge):
            raise ValueError(f"Skeleton index out of range: {edge}")
        edges.append(tuple(edge))
    return KeypointSchema(labels=tuple(names), edges=tuple(edges))


def other_keypoints_field(keypoints_field: str) -> str | None:
    """Return the point field that must not share a skeleton with this one."""
    if keypoints_field == DEFAULT_KEYPOINTS_FIELD:
        return KEYPOINTS_68_FIELD
    if keypoints_field == KEYPOINTS_68_FIELD:
        return DEFAULT_KEYPOINTS_FIELD
    return None


def targets_for_schema(schema: KeypointSchema, keypoints_field: str, label: str) -> tuple[str, str]:
    """Send 68-point faces to their own field. Both schemas use one box label."""
    point_count = len(schema.labels)
    if point_count == POINT_COUNT_68:
        if keypoints_field not in {DEFAULT_KEYPOINTS_FIELD, KEYPOINTS_68_FIELD}:
            raise ValueError(f"68-point keypoints belong in field '{KEYPOINTS_68_FIELD}'")
        if label != DEFAULT_LABEL:
            raise ValueError(f"Face boxes use label '{DEFAULT_LABEL}'")
        return KEYPOINTS_68_FIELD, DEFAULT_LABEL
    if keypoints_field == KEYPOINTS_68_FIELD:
        raise ValueError(f"Field '{KEYPOINTS_68_FIELD}' only stores 68-point faces")
    return keypoints_field, label


def face_box_locked(other_keypoints: object, label_name: str) -> bool:
    """Keep the current box when the other point field already has this face."""
    return bool(_matching_labels(_iter_field(other_keypoints, "keypoints"), label_name))


def annotation_box(annotation: dict) -> tuple[float, float, float, float] | None:
    """Use ``face_box`` when it is a real box. Otherwise use the COCO ``bbox`` extent."""
    face_box = absolute_box(annotation.get("face_box"))
    if face_box is not None:
        return face_box
    return absolute_box(annotation.get("bbox"))


def absolute_box(raw_box: object) -> tuple[float, float, float, float] | None:
    """Return an absolute COCO xywh box, or None when it is missing or empty."""
    if not isinstance(raw_box, list) or len(raw_box) != 4:
        return None
    values = [_as_float(item) for item in raw_box]
    if any(item is None for item in values):
        return None
    _left, _top, box_w, box_h = values
    if box_w <= 0 or box_h <= 0:
        return None
    return tuple(values)


def _relative_xywh(
    box_px: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    left, top, box_w, box_h = box_px
    return tuple(
        round(item, COORD_DECIMALS)
        for item in (left / width, top / height, box_w / width, box_h / height)
    )


def absolute_points(
    raw_points: object,
    point_count: int,
) -> tuple[tuple[tuple[float, float], ...], tuple[int, ...]] | None:
    """Convert COCO ``[x, y, v]`` points. A hidden point stays a NaN slot."""
    if not isinstance(raw_points, list) or len(raw_points) != point_count * 3:
        return None
    points: list[tuple[float, float]] = []
    visibility: list[int] = []
    for offset in range(0, len(raw_points), 3):
        x = _as_float(raw_points[offset])
        y = _as_float(raw_points[offset + 1])
        flag = _as_visibility(raw_points[offset + 2], point_count)
        if x is None or y is None or flag is None:
            return None
        visibility.append(flag)
        points.append((float("nan"), float("nan")) if _point_hidden(point_count, x, y, flag) else (x, y))
    return tuple(points), tuple(visibility)


def coordinates_fit(annotation: FaceAnnotation, width: int, height: int) -> bool:
    """Return whether the absolute box and visible points lie inside this image."""
    left, top, box_w, box_h = annotation.box_px
    if min(left, top) < -1 or left + box_w > width + 1 or top + box_h > height + 1:
        return False
    for x, y in annotation.points_px:
        if math.isnan(x) or math.isnan(y):
            continue
        if x < -1 or y < -1 or x > width + 1 or y > height + 1:
            return False
    return True


def parse_coco(
    document: dict, image_dir: Path
) -> tuple[KeypointSchema, dict[str, FaceAnnotation], list[dict[str, str]], int]:
    """Parse one COCO keypoint document into annotations keyed by file name."""
    schema = load_schema(document)
    images = document.get("images")
    annotations = document.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO JSON is missing images or annotations")
    by_image: dict[object, list[dict]] = defaultdict(list)
    for annotation in annotations:
        if isinstance(annotation, dict):
            by_image[annotation.get("image_id")].append(annotation)

    filenames: list[str] = []
    prepared: list[tuple[str, FaceAnnotation | None, dict[str, str] | None]] = []
    for image in images:
        if not isinstance(image, dict):
            prepared.append(("", None, issue_row("bad_image", detail="image entry is not an object")))
            continue
        filename = image.get("file_name")
        if not isinstance(filename, str) or Path(filename).name != filename or filename in {".", ".."}:
            prepared.append(("", None, issue_row("bad_filename", detail=str(filename))))
            continue
        filenames.append(filename)
        width = _as_positive_int(image.get("width"))
        height = _as_positive_int(image.get("height"))
        image_path = str(image_dir / filename)
        if width is None or height is None:
            prepared.append((filename, None, issue_row("bad_image_size", filename, filepath=image_path)))
            continue
        if not (image_dir / filename).is_file():
            prepared.append((filename, None, issue_row("missing_file", filename, filepath=image_path)))
            continue
        anns = by_image.get(image.get("id"), [])
        if len(anns) != 1:
            kind = "missing_annotation" if not anns else "multiple_annotations"
            prepared.append((
                filename,
                None,
                issue_row(kind, filename, filepath=image_path, detail=str(len(anns))),
            ))
            continue
        box_px = annotation_box(anns[0])
        points = absolute_points(anns[0].get("keypoints"), len(schema.labels))
        if box_px is None:
            prepared.append((filename, None, issue_row("bad_box", filename, filepath=image_path)))
            continue
        if points is None:
            prepared.append((filename, None, issue_row("bad_keypoints", filename, filepath=image_path)))
            continue
        points_px, visibility = points
        prepared.append((
            filename,
            FaceAnnotation(
                filename,
                image_path,
                width,
                height,
                box_px,
                points_px,
                _relative_xywh(box_px, width, height),
                tuple(
                    (float("nan"), float("nan")) if math.isnan(x) or math.isnan(y) else (
                        round(x / width, COORD_DECIMALS),
                        round(y / height, COORD_DECIMALS),
                    )
                    for x, y in points_px
                ),
                visibility,
            ),
            None,
        ))

    counts = Counter(filenames)
    labels: dict[str, FaceAnnotation] = {}
    issues: list[dict[str, str]] = []
    reported_duplicates: set[str] = set()
    for filename, annotation, row in prepared:
        if filename and counts[filename] > 1:
            if filename not in reported_duplicates:
                reported_duplicates.add(filename)
                issues.append(issue_row("duplicate_filename", filename, detail=str(counts[filename])))
            continue
        if row is not None:
            issues.append(row)
            continue
        if annotation is not None:
            labels[filename] = annotation
    return schema, labels, issues, len(images)


def index_sha256(
    sample_ids: list[str], hashes: list[object], filepaths: list[str]
) -> dict[str, list[tuple[str, str]]]:
    """Map a lowercase SHA-256 digest to ``(sample id, filepath)`` pairs."""
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for sample_id, digest, filepath in zip(sample_ids, hashes, filepaths, strict=True):
        if not isinstance(digest, str) or not digest.strip():
            continue
        grouped[digest.strip().lower()].append((str(sample_id), filepath))
    return grouped


def hash_annotations(
    labels: dict[str, FaceAnnotation],
) -> tuple[list[tuple[FaceAnnotation, str]], list[dict[str, str]]]:
    """Hash each parsed image. Unreadable files become ``hash_error`` rows."""
    hashed: list[tuple[FaceAnnotation, str]] = []
    issues: list[dict[str, str]] = []
    total = len(labels)
    for index, annotation in enumerate(labels.values(), start=1):
        try:
            digest = compute_sha256_hex(Path(annotation.image_path))
        except OSError as error:
            issues.append(issue_row(
                "hash_error", annotation.filename, filepath=annotation.image_path, detail=str(error)
            ))
            continue
        hashed.append((annotation, digest))
        if index % 2000 == 0 or index == total:
            logger.info("Hashed %s/%s images", index, total)
    return hashed, issues


def merge_tags(tags: object, sample_tags: list[str]) -> tuple[list[str], bool]:
    """Append missing sample tags. Existing tags stay in their original order."""
    merged = [tag for tag in tags or [] if isinstance(tag, str)]
    changed = False
    for tag in sample_tags:
        if tag not in merged:
            merged.append(tag)
            changed = True
    return merged, changed


def _boxes_equal(left: object, right: tuple[float, float, float, float]) -> bool:
    if not isinstance(left, (list, tuple)) or len(left) != 4:
        return False
    return all(abs(float(item) - expected) <= COORD_TOLERANCE for item, expected in zip(left, right, strict=True))


def _points_equal(
    left: object, right: tuple[tuple[float, float], ...]
) -> bool:
    if not isinstance(left, (list, tuple)) or len(left) != len(right):
        return False
    for item, expected in zip(left, right, strict=True):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return False
        for value, target in zip(item, expected, strict=True):
            value = float(value)
            if math.isnan(target):
                if not math.isnan(value):
                    return False
                continue
            if math.isnan(value) or abs(value - target) > COORD_TOLERANCE:
                return False
    return True


def _visibility_of(keypoint: object) -> tuple[int, ...] | None:
    if not hasattr(keypoint, "has_attribute") or not keypoint.has_attribute("visibility"):
        return None
    value = keypoint.get_attribute_value("visibility")
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return None


def _instance_id(label: object) -> str | None:
    instance = getattr(label, "instance", None)
    if instance is None or getattr(instance, "id", None) is None:
        return None
    return str(instance.id)


def _matching_labels(labels: list, label_name: str) -> list:
    return [label for label in labels if getattr(label, "label", None) == label_name]


def label_is_current(
    detections: object,
    keypoints: object,
    label_name: str,
    annotation: FaceAnnotation,
) -> bool:
    """Return whether this sample already has the same box, points, and visibility."""
    detection_list = list(getattr(detections, "detections", None) or [])
    keypoint_list = list(getattr(keypoints, "keypoints", None) or [])
    matched_boxes = _matching_labels(detection_list, label_name)
    matched_points = _matching_labels(keypoint_list, label_name)
    if len(matched_boxes) != 1 or len(matched_points) != 1:
        return False
    if not _boxes_equal(matched_boxes[0].bounding_box, annotation.box):
        return False
    if not _points_equal(matched_points[0].points, annotation.points):
        return False
    return _visibility_of(matched_points[0]) == annotation.visibility


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
    return int(width), int(height)


def _import_fiftyone():
    import fiftyone as fo

    return fo


def _iter_field(label: object, field_name: str) -> list:
    if label is None:
        return []
    return list(getattr(label, field_name, None) or [])


def build_labels(annotation: FaceAnnotation, label_name: str, instance_id: str | None, kept_boxes: list, kept_points: list):
    """Return detections and keypoints, sharing one instance id."""
    fo = _import_fiftyone()
    instance = fo.Instance(id=instance_id) if instance_id else fo.Instance()
    detection = fo.Detection(
        label=label_name,
        bounding_box=list(annotation.box),
        instance=instance,
    )
    keypoint = fo.Keypoint(
        label=label_name,
        points=[tuple(point) for point in annotation.points],
        instance=instance,
    )
    keypoint["visibility"] = list(annotation.visibility)
    return (
        fo.Detections(detections=[*kept_boxes, detection]),
        fo.Keypoints(keypoints=[*kept_points, keypoint]),
    )


def match_by_sha256(
    hashed: list[tuple[FaceAnnotation, str]],
    sha_index: dict[str, list[tuple[str, str]]],
    phash_index: dict[str, list[tuple[str, str]]] | None = None,
) -> tuple[list[tuple[FaceAnnotation, str, str]], list[dict[str, str]], int]:
    """Pair each image with one sample. SHA-256 wins; exact pHash is the fallback.

    A digest that occurs more than once in the JSON, or on more than one
    sample, is skipped. A pHash fallback never takes a sample already claimed
    by SHA-256.
    """
    by_digest: dict[str, list[FaceAnnotation]] = defaultdict(list)
    for annotation, digest in hashed:
        by_digest[digest].append(annotation)
    matches: list[tuple[FaceAnnotation, str, str]] = []
    issues: list[dict[str, str]] = []
    pending: list[tuple[FaceAnnotation, str]] = []
    claimed: set[str] = set()
    for digest, group in by_digest.items():
        if len(group) > 1:
            for annotation in group:
                issues.append(issue_row(
                    "duplicate_sha256",
                    annotation.filename,
                    filepath=annotation.image_path,
                    sha256=digest,
                    detail=str(len(group)),
                ))
            continue
        annotation = group[0]
        hits = sha_index.get(digest, [])
        if len(hits) > 1:
            issues.append(issue_row(
                "ambiguous_sha256",
                annotation.filename,
                filepath=annotation.image_path,
                sha256=digest,
                detail=str(len(hits)),
            ))
            continue
        if not hits:
            pending.append((annotation, digest))
            continue
        sample_id, filepath = hits[0]
        matches.append((annotation, sample_id, filepath))
        claimed.add(sample_id)
    phash_matched = _match_by_phash(pending, phash_index, matches, issues, claimed)
    return matches, issues, phash_matched


def _match_by_phash(
    pending: list[tuple[FaceAnnotation, str]],
    phash_index: dict[str, list[tuple[str, str]]] | None,
    matches: list[tuple[FaceAnnotation, str, str]],
    issues: list[dict[str, str]],
    claimed: set[str],
) -> int:
    """Match SHA-256 misses by exact pHash. Returns how many matches were added."""
    if phash_index is None:
        for annotation, digest in pending:
            issues.append(issue_row(
                "not_in_dataset",
                annotation.filename,
                filepath=annotation.image_path,
                sha256=digest,
            ))
        return 0
    rows: list[tuple[FaceAnnotation, str, str]] = []
    total = len(pending)
    for index, (annotation, digest) in enumerate(pending, start=1):
        try:
            phash = compute_phash_hex(Path(annotation.image_path)).strip().lower()
        except (OSError, ValueError) as error:
            issues.append(issue_row(
                "phash_error",
                annotation.filename,
                filepath=annotation.image_path,
                sha256=digest,
                detail=str(error),
            ))
            continue
        if not phash:
            issues.append(issue_row(
                "phash_error",
                annotation.filename,
                filepath=annotation.image_path,
                sha256=digest,
                detail="empty phash",
            ))
            continue
        rows.append((annotation, digest, phash))
        if index % 500 == 0 or index == total:
            logger.info("pHash fallback %s/%s images", index, total)
    by_phash: dict[str, list[tuple[FaceAnnotation, str, str]]] = defaultdict(list)
    for row in rows:
        by_phash[row[2]].append(row)
    added = 0
    for phash, group in by_phash.items():
        if len(group) > 1:
            for annotation, digest, grouped_phash in group:
                issues.append(issue_row(
                    "duplicate_phash",
                    annotation.filename,
                    filepath=annotation.image_path,
                    sha256=digest,
                    phash=grouped_phash,
                    detail=str(len(group)),
                ))
            continue
        annotation, digest, grouped_phash = group[0]
        hits = phash_index.get(grouped_phash, [])
        if not hits:
            issues.append(issue_row(
                "not_in_dataset",
                annotation.filename,
                filepath=annotation.image_path,
                sha256=digest,
                phash=grouped_phash,
            ))
            continue
        if len(hits) > 1:
            issues.append(issue_row(
                "ambiguous_phash",
                annotation.filename,
                filepath=annotation.image_path,
                sha256=digest,
                phash=grouped_phash,
                detail=str(len(hits)),
            ))
            continue
        sample_id, filepath = hits[0]
        if sample_id in claimed:
            issues.append(issue_row(
                "phash_taken",
                annotation.filename,
                sample_id,
                filepath,
                sha256=digest,
                phash=grouped_phash,
            ))
            continue
        matches.append((annotation, sample_id, filepath))
        claimed.add(sample_id)
        added += 1
    return added


def plan_updates(
    matches: list[tuple[FaceAnnotation, str, str]],
    current: dict[str, dict],
    sample_tags: list[str],
    label_name: str,
) -> tuple[list[dict], int, int, list[dict[str, str]]]:
    """Decide which matched samples need a write. Does not touch the dataset."""
    updates: list[dict] = []
    matched = 0
    unchanged = 0
    issues: list[dict[str, str]] = []
    for annotation, sample_id, filepath in matches:
        filename = annotation.filename
        matched += 1
        sample = current[sample_id]
        size = image_size(sample["metadata"])
        if size is not None and size != (annotation.width, annotation.height):
            if not coordinates_fit(annotation, size[0], size[1]):
                issues.append(issue_row(
                    "size_mismatch",
                    filename,
                    sample_id,
                    filepath,
                    f"json={annotation.width}x{annotation.height} metadata={size[0]}x{size[1]}",
                ))
                continue
            annotation = annotation.for_size(size[0], size[1])
        existing_boxes = _matching_labels(
            _iter_field(sample["detections"], "detections"), label_name
        )
        locked = face_box_locked(sample.get("other_keypoints"), label_name) and len(existing_boxes) == 1
        if locked:
            annotation = replace(
                annotation,
                box=tuple(float(item) for item in existing_boxes[0].bounding_box),
            )
        tags, tags_changed = merge_tags(sample["tags"], sample_tags)
        if not tags_changed and label_is_current(
            sample["detections"], sample["keypoints"], label_name, annotation
        ):
            unchanged += 1
            continue
        kept_boxes = [
            detection
            for detection in _iter_field(sample["detections"], "detections")
            if getattr(detection, "label", None) != label_name
        ]
        kept_points = [
            keypoint
            for keypoint in _iter_field(sample["keypoints"], "keypoints")
            if getattr(keypoint, "label", None) != label_name
        ]
        instance_id = None
        for label in (
            *_matching_labels(_iter_field(sample["detections"], "detections"), label_name),
            *_matching_labels(_iter_field(sample["keypoints"], "keypoints"), label_name),
        ):
            instance_id = _instance_id(label)
            if instance_id:
                break
        detections, keypoints = build_labels(
            annotation, label_name, instance_id, kept_boxes, kept_points
        )
        updates.append({
            "sample_id": sample_id,
            "detections": detections,
            "keypoints": keypoints,
            "tags": tags,
        })
    return updates, matched, unchanged, issues


def load_current(
    dataset: object,
    sample_ids: list[str],
    label_field: str,
    keypoints_field: str,
) -> dict[str, dict]:
    """Load tags, boxes, keypoints, and metadata for the matched sample ids."""
    if not sample_ids:
        return {}
    view = dataset.select(sample_ids)
    ids = view.values("id")
    empty = [None] * len(ids)
    detections = view.values(label_field) if dataset.has_field(label_field) else empty
    keypoints = view.values(keypoints_field) if dataset.has_field(keypoints_field) else list(empty)
    metadata = view.values("metadata") if dataset.has_field("metadata") else list(empty)
    other_field = other_keypoints_field(keypoints_field)
    if other_field and dataset.has_field(other_field):
        other = view.values(other_field)
    else:
        other = list(empty)
    return {
        sample_id: {
            "tags": tags,
            "detections": detection,
            "keypoints": keypoint,
            "other_keypoints": other_keypoint,
            "metadata": meta,
        }
        for sample_id, tags, detection, keypoint, other_keypoint, meta in zip(
            ids, view.values("tags"), detections, keypoints, other, metadata, strict=True
        )
    }


def require_label_field(dataset: object, field_name: str, document_type: type) -> None:
    """Reject a field that exists but stores a different label type."""
    if not dataset.has_field(field_name):
        return
    fo = _import_fiftyone()
    field = dataset.get_field(field_name)
    if not isinstance(field, fo.EmbeddedDocumentField) or field.document_type is not document_type:
        raise ValueError(f"Field '{field_name}' must contain {document_type.__name__}")


def ensure_schema(dataset: object, label_field: str, keypoints_field: str) -> None:
    """Create the detections field, keypoints field, and per-point visibility list."""
    fo = _import_fiftyone()
    if not dataset.has_field(label_field):
        dataset.add_sample_field(
            label_field, fo.EmbeddedDocumentField, embedded_doc_type=fo.Detections
        )
    if not dataset.has_field(keypoints_field):
        dataset.add_sample_field(
            keypoints_field, fo.EmbeddedDocumentField, embedded_doc_type=fo.Keypoints
        )
    visibility_field = f"{keypoints_field}.keypoints.visibility"
    if visibility_field not in dataset.get_field_schema(flat=True):
        dataset.add_sample_field(visibility_field, fo.ListField, subfield=fo.IntField)


def ensure_skeleton_count(dataset: object, keypoints_field: str, schema: KeypointSchema) -> None:
    """Refuse to replace a skeleton that has a different point count."""
    existing = dict(dataset.skeletons or {}).get(keypoints_field)
    if existing is None:
        return
    found = len(list(existing.labels or []))
    if found != len(schema.labels):
        raise ValueError(f"Field '{keypoints_field}' already has a {found}-point skeleton")


def ensure_skeleton(dataset: object, keypoints_field: str, schema: KeypointSchema) -> None:
    """Store the category skeleton on the keypoints field without dropping other skeletons."""
    fo = _import_fiftyone()
    ensure_skeleton_count(dataset, keypoints_field, schema)
    current = dict(dataset.skeletons or {})
    existing = current.get(keypoints_field)
    edges = [list(edge) for edge in schema.edges]
    if existing is not None and list(existing.labels or []) == list(schema.labels):
        existing_edges = [list(edge) for edge in existing.edges]
        if existing_edges == edges:
            return
    current[keypoints_field] = fo.KeypointSkeleton(labels=list(schema.labels), edges=edges)
    dataset.skeletons = current
    dataset.save()


def write_updates(
    dataset: object,
    updates: list[dict],
    label_field: str,
    keypoints_field: str,
) -> None:
    """Write merged labels and tags for the planned sample ids."""
    if not updates:
        return
    by_id = {item["sample_id"]: item for item in updates}
    view = dataset.select(list(by_id))
    ordered = [by_id[sample_id] for sample_id in view.values("id")]
    view.set_values(label_field, [item["detections"] for item in ordered])
    view.set_values(keypoints_field, [item["keypoints"] for item in ordered])
    view.set_values("tags", [item["tags"] for item in ordered])


def issue_csv_path(dataset_name: str, report_dir: Path) -> Path:
    """Return ``tmp/update_face_keypoints_<dataset>.csv``."""
    safe = "".join(
        char if char.isalnum() or char in "._-" else "_" for char in dataset_name
    ).strip("._")
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"update_face_keypoints_{safe or 'dataset'}.csv"


def write_issues(path: Path, issues: list[dict[str, str]]) -> None:
    """Write issue rows. The caller skips this when there are none."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ISSUE_COLUMNS))
        writer.writeheader()
        writer.writerows(issues)


def format_result(result: UpdateResult, dry_run: bool) -> str:
    """Format the single summary line printed by a run."""
    counts = Counter(row["issue"] for row in result.issues)
    text = (
        f"mode={'dry-run' if dry_run else 'update'} "
        f"json_images={result.json_images} matched={result.matched} "
        f"to_update={result.to_update} unchanged={result.unchanged} "
        f"written={result.written} "
        f"missing_file={counts['missing_file']} "
        f"phash_matched={result.phash_matched} "
        f"not_in_dataset={counts['not_in_dataset']} "
        f"ambiguous_sha256={counts['ambiguous_sha256']} "
        f"ambiguous_phash={counts['ambiguous_phash']} "
        f"issues={len(result.issues)}"
    )
    if result.csv_path:
        text += f" csv_path={result.csv_path}"
    return text


def update_dataset(
    dataset: object,
    image_dir: Path,
    json_path: Path,
    sample_tags: list[str],
    *,
    label_field: str = DEFAULT_LABEL_FIELD,
    keypoints_field: str = DEFAULT_KEYPOINTS_FIELD,
    label: str = DEFAULT_LABEL,
    dry_run: bool = False,
    report_dir: Path | None = None,
) -> UpdateResult:
    """Parse one JSON file and attach its faces to existing samples."""
    fo = _import_fiftyone()
    with json_path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"COCO JSON must be an object: {json_path}")
    if "sha256" not in dataset.get_field_schema():
        raise ValueError(MISSING_SHA256_MSG)
    schema, labels, issues, json_images = parse_coco(document, image_dir)
    keypoints_field, label = targets_for_schema(schema, keypoints_field, label)
    require_label_field(dataset, label_field, fo.Detections)
    require_label_field(dataset, keypoints_field, fo.Keypoints)
    ensure_skeleton_count(dataset, keypoints_field, schema)
    if dataset.has_field("phash"):
        sample_ids, hashes, filepaths, phashes = dataset.values(
            ["id", "sha256", "filepath", "phash"]
        )
        phash_index = index_sha256(sample_ids, phashes, filepaths)
    else:
        sample_ids, hashes, filepaths = dataset.values(["id", "sha256", "filepath"])
        phash_index = None
    sha_index = index_sha256(sample_ids, hashes, filepaths)
    hashed, hash_issues = hash_annotations(labels)
    issues.extend(hash_issues)
    hash_matches, match_issues, phash_matched = match_by_sha256(hashed, sha_index, phash_index)
    issues.extend(match_issues)
    current = load_current(
        dataset, [sample_id for _annotation, sample_id, _filepath in hash_matches], label_field, keypoints_field
    )
    updates, matched, unchanged, size_issues = plan_updates(
        hash_matches, current, sample_tags, label
    )
    issues.extend(size_issues)
    result = UpdateResult(
        json_images=json_images,
        matched=matched,
        to_update=len(updates),
        unchanged=unchanged,
        written=0 if dry_run else len(updates),
        phash_matched=phash_matched,
        issues=issues,
    )
    if issues:
        path = issue_csv_path(dataset.name or "dataset", report_dir or Path.cwd() / REPORT_SUBDIR)
        write_issues(path, issues)
        result.csv_path = str(path)
    if not dry_run and (updates or unchanged):
        ensure_schema(dataset, label_field, keypoints_field)
        ensure_skeleton(dataset, keypoints_field, schema)
        write_updates(dataset, updates, label_field, keypoints_field)
    return result


def main(argv: list[str] | None = None) -> int:
    """Load the named dataset and attach one COCO keypoint file."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    try:
        image_dir = args.image_dir.expanduser().resolve(strict=True)
        json_path = args.json_path.expanduser().resolve(strict=True)
        if not image_dir.is_dir():
            raise ValueError(f"Not a directory: {image_dir}")
        if not json_path.is_file():
            raise ValueError(f"Not a file: {json_path}")
        fo = _import_fiftyone()
        if not fo.dataset_exists(args.dataset):
            raise ValueError(f"Dataset does not exist: {args.dataset}")
        dataset = fo.load_dataset(args.dataset)
        result = update_dataset(
            dataset,
            image_dir,
            json_path,
            args.sample_tags,
            label_field=args.label_field,
            keypoints_field=args.keypoints_field,
            label=args.label,
            dry_run=args.dry_run,
        )
        print(format_result(result, args.dry_run), flush=True)
        return 0
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except Exception as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
