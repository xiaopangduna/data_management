"""Load a CVAT annotation run and replace only the requested classes.

Boxes whose labels are outside ``--classes`` are restored from the dataset
snapshot taken before the download. Samples whose requested boxes or CVAT
attribute values changed gain ``--sample-tags`` and ``changed``. The
``cvat_<anno-key>`` tag from publish is left in place.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os

import fiftyone as fo

logger = logging.getLogger(__name__)

CHANGED_TAG = "changed"
# CVAT stores integer pixels. A few pixels of rounding, on a small image, is not an edit.
PIXEL_TOLERANCE = 5.0


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for loading a CVAT run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=nonempty)
    parser.add_argument("--anno-key", required=True, type=nonempty)
    parser.add_argument(
        "--classes",
        required=True,
        type=name_list,
        help="Detection labels replaced by the CVAT result. Other labels are kept.",
    )
    parser.add_argument(
        "--sample-tags",
        required=True,
        type=name_list,
        help="Tags added, with 'changed', on samples whose requested boxes changed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the annotation run and exit before downloading labels.",
    )
    return parser.parse_args(argv)


def cvat_credentials() -> tuple[str, str, str]:
    """Read the CVAT URL and login from the environment."""
    values = {
        "FIFTYONE_CVAT_URL": os.environ.get("FIFTYONE_CVAT_URL", "").strip(),
        "FIFTYONE_CVAT_USERNAME": os.environ.get("FIFTYONE_CVAT_USERNAME", "").strip(),
        "FIFTYONE_CVAT_PASSWORD": os.environ.get("FIFTYONE_CVAT_PASSWORD", ""),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError("Set " + ", ".join(missing))
    return values["FIFTYONE_CVAT_URL"], values["FIFTYONE_CVAT_USERNAME"], values["FIFTYONE_CVAT_PASSWORD"]


def label_schema(info: object) -> dict:
    """Return the label schema recorded on an annotation run."""
    config = getattr(info, "config", None)
    return getattr(config, "label_schema", None) or {}


def label_field_from_run(info: object) -> str:
    """Return the single label field recorded on an annotation run."""
    schema = label_schema(info)
    if len(schema) != 1:
        raise ValueError("Annotation run must contain exactly one label field")
    return str(next(iter(schema)))


def attribute_names_from_run(info: object) -> list[str]:
    """Return CVAT attribute names stored on the annotation run, in schema order."""
    names: list[str] = []
    for spec in label_schema(info).values():
        attributes = spec.get("attributes") if isinstance(spec, dict) else getattr(spec, "attributes", None)
        for name in attributes or {}:
            if name not in names:
                names.append(str(name))
    return names


def detection_list(value: object) -> list:
    """Return detections on a FiftyOne label, or an empty list."""
    detections = getattr(value, "detections", None)
    return list(detections or [])


def pixel_box(box: tuple[float, float, float, float], frame_size: tuple[float, float]) -> tuple[float, float, float, float]:
    """Convert a normalized box to pixels."""
    width, height = frame_size
    x, y, w, h = box
    return (x * width, y * height, w * width, h * height)


def pair_boxes(
    left: list[tuple[float, ...]],
    right: list[tuple[float, ...]],
    tolerance: float,
) -> list[int] | None:
    """Return each left box's match in ``right``, or None when no pairing fits."""
    if len(left) != len(right):
        return None
    candidates = []
    for box in left:
        partners = [
            index
            for index, other in enumerate(right)
            if all(abs(item - other_item) <= tolerance for item, other_item in zip(box, other))
        ]
        if not partners:
            return None
        candidates.append(partners)
    used = [False] * len(right)
    assignment = [-1] * len(left)

    def search(index: int) -> bool:
        if index == len(candidates):
            return True
        for partner in candidates[index]:
            if used[partner]:
                continue
            used[partner] = True
            assignment[index] = partner
            if search(index + 1):
                return True
            used[partner] = False
            assignment[index] = -1
        return False

    if not search(0):
        return None
    return assignment


def boxes_match(left: list[tuple[float, ...]], right: list[tuple[float, ...]], tolerance: float) -> bool:
    """Return True when the two lists can be paired with each edge within ``tolerance``."""
    return pair_boxes(left, right, tolerance) is not None


def class_detections(value: object, names: set[str]) -> dict[str, list[tuple[tuple[float, float, float, float], fo.Detection]]]:
    """Group requested detections as ``(normalized box, detection)`` by label."""
    grouped: dict[str, list[tuple[tuple[float, float, float, float], fo.Detection]]] = {}
    for detection in detection_list(value):
        label = str(detection.label)
        if label not in names:
            continue
        box = tuple(float(item) for item in detection.bounding_box)
        grouped.setdefault(label, []).append((box, detection))
    return grouped


def attribute_text(detection: fo.Detection, name: str) -> str | None:
    """Return one detection attribute as stripped text, or None when unset."""
    raw = detection.get_attribute_value(name, None)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def attributes_differ(left: fo.Detection, right: fo.Detection, attribute_names: list[str]) -> bool:
    """Return True when any named attribute differs."""
    return any(attribute_text(left, name) != attribute_text(right, name) for name in attribute_names)


def boxes_changed(
    before: object,
    after: object,
    names: set[str],
    frame_size: tuple[float, float] | None = None,
    tolerance: float = PIXEL_TOLERANCE,
    attribute_names: list[str] | None = None,
) -> bool:
    """Return True when requested boxes or their CVAT attributes changed.

    Boxes of the same label are matched to each other, not compared in sorted
    order. With ``frame_size``, ``tolerance`` is in pixels. Without it,
    coordinates stay normalized and the allowance is ``tolerance / 1000``.
    Matched boxes still count as changed when a named attribute differs.
    """
    left = class_detections(before, names)
    right = class_detections(after, names)
    if set(left) != set(right):
        return True
    limit = tolerance if frame_size else tolerance / 1000.0
    fields = list(attribute_names or [])
    for label in left:
        before_boxes = [item[0] for item in left[label]]
        after_boxes = [item[0] for item in right[label]]
        if frame_size:
            before_boxes = [pixel_box(box, frame_size) for box in before_boxes]
            after_boxes = [pixel_box(box, frame_size) for box in after_boxes]
        pairing = pair_boxes(before_boxes, after_boxes, limit)
        if pairing is None:
            return True
        for index, partner in enumerate(pairing):
            if attributes_differ(left[label][index][1], right[label][partner][1], fields):
                return True
    return False


def frame_size_of(sample: fo.Sample) -> tuple[float, float] | None:
    """Return ``(width, height)`` when the sample metadata has both."""
    metadata = sample.metadata
    if metadata is None or not metadata.width or not metadata.height:
        return None
    return (float(metadata.width), float(metadata.height))


def merge_class_detections(before: object, after: object, names: set[str]) -> fo.Detections:
    """Keep labels outside ``names`` from ``before`` and labels inside ``names`` from ``after``."""
    outside = [
        copy.deepcopy(detection)
        for detection in detection_list(before)
        if str(detection.label) not in names
    ]
    inside = [
        copy.deepcopy(detection)
        for detection in detection_list(after)
        if str(detection.label) in names
    ]
    return fo.Detections(detections=outside + inside)


def append_changed_tags(existing: object, extra: list[str]) -> list[str]:
    """Append batch tags and ``changed`` without dropping existing tags."""
    tags = [str(tag) for tag in (existing or [])]
    for tag in [*extra, CHANGED_TAG]:
        if tag not in tags:
            tags.append(tag)
    return tags


def print_report(values: dict[str, object]) -> None:
    """Print key=value lines in insertion order."""
    for key, value in values.items():
        print(f"{key}={value}")


def scaled_boxes(
    items: list[tuple[tuple[float, float, float, float], fo.Detection]],
    frame_size: tuple[float, float] | None,
) -> list[tuple[float, float, float, float]]:
    """Return boxes in pixels when the frame size is known."""
    boxes = [item[0] for item in items]
    if frame_size is None:
        return boxes
    return [pixel_box(box, frame_size) for box in boxes]


def restore_matched_tags(
    before: object,
    after: object,
    names: set[str],
    frame_size: tuple[float, float] | None,
    tolerance: float = PIXEL_TOLERANCE,
) -> None:
    """Copy label tags from each paired box onto the downloaded detection.

    CVAT returns attributes such as ``age`` and drops the ``age_*`` tags. The
    follow-up tag rewrite compares those tags with the new attribute, so the
    previous tags have to stay on boxes that still match.
    """
    left = class_detections(before, names)
    right = class_detections(after, names)
    limit = tolerance if frame_size else tolerance / 1000.0
    for label in set(left) & set(right):
        pairing = pair_boxes(scaled_boxes(left[label], frame_size), scaled_boxes(right[label], frame_size), limit)
        if pairing is None:
            continue
        for index, partner in enumerate(pairing):
            source = left[label][index][1]
            target = right[label][partner][1]
            if not list(target.tags or []) and list(source.tags or []):
                target.tags = [str(tag) for tag in source.tags]


def apply_load(
    dataset: fo.Dataset,
    sample_ids: list[str],
    before: dict[str, object],
    label_field: str,
    names: set[str],
    extra_tags: list[str],
    attribute_names: list[str] | None = None,
) -> tuple[int, int]:
    """Merge downloaded boxes and tag changed samples. Returns changed and unchanged counts."""
    changed = 0
    unchanged = 0
    fields = list(attribute_names or [])
    for sample in dataset.select(sample_ids).iter_samples(autosave=True):
        sample_id = str(sample.id)
        previous = before[sample_id]
        downloaded = sample[label_field] if dataset.has_field(label_field) else None
        frame_size = frame_size_of(sample)
        restore_matched_tags(previous, downloaded, names, frame_size)
        merged = merge_class_detections(previous, downloaded, names)
        sample[label_field] = merged
        if boxes_changed(previous, merged, names, frame_size=frame_size, attribute_names=fields):
            sample.tags = append_changed_tags(sample.tags, extra_tags)
            changed += 1
        else:
            unchanged += 1
    dataset.save()
    return changed, unchanged


def run(args: argparse.Namespace) -> int:
    """Download one annotation run and merge the requested classes."""
    if not fo.dataset_exists(args.dataset):
        raise ValueError(f"Dataset does not exist: {args.dataset}")
    dataset = fo.load_dataset(args.dataset)
    if args.anno_key not in dataset.list_annotation_runs():
        raise ValueError(f"Annotation run does not exist: {args.anno_key}")
    info = dataset.get_annotation_info(args.anno_key)
    label_field = label_field_from_run(info)
    attribute_names = attribute_names_from_run(info)
    view = dataset.load_annotation_view(args.anno_key)
    sample_ids = [str(sample_id) for sample_id in view.values("id")]
    if not sample_ids:
        raise ValueError(f"Annotation run '{args.anno_key}' has no samples.")
    print_report(
        {
            "mode": "dry-run" if args.dry_run else "load",
            "dataset": args.dataset,
            "anno_key": args.anno_key,
            "label_field": label_field,
            "classes": ",".join(args.classes),
            "attributes": ",".join(attribute_names),
            "sample_tags": ",".join(args.sample_tags),
            "samples": len(sample_ids),
        }
    )
    if args.dry_run:
        return 0

    before = {
        str(sample.id): copy.deepcopy(sample[label_field] if dataset.has_field(label_field) else None)
        for sample in view.select(sample_ids).iter_samples()
    }
    url, username, password = cvat_credentials()
    dataset.load_annotations(
        args.anno_key,
        url=url,
        username=username,
        password=password,
        unexpected="ignore",
        cleanup=False,
    )
    changed, unchanged = apply_load(
        dataset,
        sample_ids,
        before,
        label_field,
        set(args.classes),
        args.sample_tags,
        attribute_names,
    )
    print_report({"changed": changed, "unchanged": unchanged, "loaded": "true"})
    return 0


def main(argv: list[str] | None = None) -> int:
    """Load CVAT annotations back into an existing dataset."""
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
