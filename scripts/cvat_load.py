"""Load a CVAT annotation run and replace only the requested classes.

Boxes whose labels are outside ``--classes`` are restored from the dataset
snapshot taken before the download. Samples whose requested boxes changed gain
``--sample-tags`` and ``changed``. The ``cvat_<anno-key>`` tag from publish is left
in place.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os

import fiftyone as fo

logger = logging.getLogger(__name__)

CHANGED_TAG = "changed"
BOX_TOLERANCE = 0.002


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


def label_field_from_run(info: object) -> str:
    """Return the single label field recorded on an annotation run."""
    config = getattr(info, "config", None)
    schema = getattr(config, "label_schema", None) or {}
    if len(schema) != 1:
        raise ValueError("Annotation run must contain exactly one label field")
    return str(next(iter(schema)))


def detection_list(value: object) -> list:
    """Return detections on a FiftyOne label, or an empty list."""
    detections = getattr(value, "detections", None)
    return list(detections or [])


def class_boxes(value: object, names: set[str]) -> tuple:
    """Return sorted ``(label, box)`` rows for labels in ``names``."""
    rows = []
    for detection in detection_list(value):
        label = str(detection.label)
        if label not in names:
            continue
        box = tuple(float(item) for item in detection.bounding_box)
        rows.append((label, box))
    return tuple(sorted(rows))


def boxes_changed(before: object, after: object, names: set[str], tolerance: float = BOX_TOLERANCE) -> bool:
    """Return True when requested boxes differ by more than ``tolerance``."""
    left = class_boxes(before, names)
    right = class_boxes(after, names)
    if len(left) != len(right):
        return True
    for (label_before, box_before), (label_after, box_after) in zip(left, right):
        if label_before != label_after or len(box_before) != len(box_after):
            return True
        if any(abs(item_before - item_after) > tolerance for item_before, item_after in zip(box_before, box_after)):
            return True
    return False


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


def apply_load(
    dataset: fo.Dataset,
    sample_ids: list[str],
    before: dict[str, object],
    label_field: str,
    names: set[str],
    extra_tags: list[str],
) -> tuple[int, int]:
    """Merge downloaded boxes and tag changed samples. Returns changed and unchanged counts."""
    changed = 0
    unchanged = 0
    for sample in dataset.select(sample_ids).iter_samples(autosave=True):
        sample_id = str(sample.id)
        previous = before[sample_id]
        downloaded = sample[label_field] if dataset.has_field(label_field) else None
        merged = merge_class_detections(previous, downloaded, names)
        sample[label_field] = merged
        if boxes_changed(previous, merged, names):
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
    label_field = label_field_from_run(dataset.get_annotation_info(args.anno_key))
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
