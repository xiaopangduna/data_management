"""Export the union of sample tags as a single YOLO detection dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=nonempty)
    parser.add_argument("--label-field", default="ground_truth", type=nonempty)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tags", required=True, type=names,
                        help="Comma-separated sample tags; match ANY tag (union).")
    parser.add_argument("--classes", type=names,
                        help="Complete class list in class ID order; otherwise sorted from selected samples.")
    parser.add_argument("--export-media", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and print the plan without writing files.")
    return parser.parse_args(argv)


def check_output(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Output directory must not be a symlink: {path}")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output directory must be empty or absent: {path}")


def export_dataset(args: argparse.Namespace) -> dict:
    import fiftyone as fo

    output = args.output_dir.expanduser().absolute()
    check_output(output)
    dataset = fo.load_dataset(args.dataset)
    if dataset.media_type != "image":
        raise ValueError("Only image datasets are supported")
    field = dataset.get_field_schema().get(args.label_field)
    if not isinstance(field, fo.EmbeddedDocumentField) or field.document_type != fo.Detections:
        raise ValueError(f"Label field must be fo.Detections: {args.label_field}")
    missing_tags = sorted(set(args.tags) - set(dataset.distinct("tags")))
    if missing_tags:
        raise ValueError(f"Unknown sample tags: {', '.join(missing_tags)}")
    view = dataset.match_tags(args.tags, all=False)
    count = len(view)
    if not count:
        raise ValueError("No samples match the requested tags")

    detected_classes: set[str] = set()
    boxes = negatives = 0
    for sample in view.iter_samples():
        if not Path(sample.filepath).is_file():
            raise ValueError(f"Missing image for sample {sample.id}: {sample.filepath}")
        labels = sample[args.label_field]
        detections = labels.detections if labels is not None else []
        negatives += not detections
        boxes += len(detections)
        for detection in detections:
            if not isinstance(detection.label, str) or not detection.label.strip():
                raise ValueError(f"Empty class label for sample {sample.id}")
            detected_classes.add(detection.label)

    classes = args.classes if args.classes is not None else sorted(detected_classes)
    omitted = sorted(detected_classes - set(classes))
    if omitted:
        raise ValueError(f"--classes omits selected labels: {', '.join(omitted)}")
    summary = {
        "dataset": args.dataset,
        "label_field": args.label_field,
        "tags": args.tags,
        "tag_matching": "union",
        "split": "train",
        "output_dir": str(output),
        "export_media": args.export_media,
        "classes": classes,
        "images": count,
        "boxes": boxes,
        "negative_images": negatives,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry run: no files written.")
        return summary

    check_output(output)
    view.export(
        export_dir=str(output),
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=args.label_field,
        split="train",
        classes=classes,
        export_media=args.export_media,
    )
    (output / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
