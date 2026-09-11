"""Export the union of sample tags as a single YOLO detection dataset."""

from __future__ import annotations

import argparse
import json
import re
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
                        help="Classes to export in class ID order; otherwise all classes sorted from selected samples.")
    parser.add_argument("--export-media", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and print the plan without writing files.")
    return parser.parse_args(argv)


def check_output(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Output directory must not be a symlink: {path}")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output directory must be empty or absent: {path}")



def export_stem(label, classes: list[str], index: int) -> str:
    """Name a sample using its exported classes and a global sequence number."""
    present = {d.label for d in label.detections} if label is not None else set()
    ordered = [name for name in classes if name in present]
    # Bound components while preserving Unicode class names and avoiding paths.
    parts = [re.sub(r"[^\w-]+", "_", name).strip("_")[:20] or "class"
             for name in ordered[:6]]
    # Cap UTF-8 bytes as well, for filesystems with a 255-byte name limit.
    parts = [part.encode("utf-8")[:30].decode("utf-8", errors="ignore") for part in parts]
    if len(ordered) > 6:
        parts.append("more")
    prefix = "__".join(parts) if parts else "negative"
    return f"{prefix}_{index:06d}"


def make_exporter(output: Path, classes: list[str], export_media: str):
    from fiftyone.utils.yolo import YOLOv5DatasetExporter

    class NamedYOLOExporter(YOLOv5DatasetExporter):
        _sample_index = 0

        def export_sample(self, image_or_path, label, metadata=None):
            self._sample_index += 1
            stem = export_stem(label, classes, self._sample_index)
            image_path = Path(self.data_path) / (stem + Path(image_or_path).suffix)
            self._media_exporter.export(image_or_path, outpath=str(image_path))
            labels_path = Path(self.labels_path) / (stem + ".txt")
            if label is None:
                labels_path.parent.mkdir(parents=True, exist_ok=True)
                labels_path.write_text("", encoding="utf-8")
            else:
                self._writer.write(
                    label, str(labels_path), self._labels_map_rev,
                    dynamic_classes=self._dynamic_classes,
                    include_confidence=self.include_confidence,
                    use_masks=self.use_masks, tolerance=self.tolerance,
                )

    return NamedYOLOExporter(
        export_dir=str(output), split="train", classes=classes,
        export_media=True if export_media == "copy" else export_media,
    )


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

    if args.classes is not None:
        view = view.filter_labels(
            args.label_field, fo.ViewField("label").is_in(args.classes),
            only_matches=False,
        )

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
    summary = {
        "dataset": args.dataset,
        "label_field": args.label_field,
        "tags": args.tags,
        "tag_matching": "union",
        "split": "train",
        "output_dir": str(output),
        "export_media": args.export_media,
        "filename_scheme": "class_prefix_global_sequence",
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
        dataset_exporter=make_exporter(output, classes, args.export_media),
        label_field=args.label_field,
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
